import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- V8 系统提示词：CoT ---
STRICT_SYSTEM_PROMPT = """You are a smart home control agent. 
Your task is to convert User Instructions into executable Python-style API calls based on the provided Home State and Device Methods.

### PROCESS:
1. **Analyze**: Identify all devices and rooms mentioned in the User Instruction.
2. **Check**: Verify if each device exists in the <home_state> and has the requested method in <device_method>.
3. **Execute**: 
   - If a device exists and the method is valid, generate the API call.
   - If a device does NOT exist or the method is invalid, ignore it or output `error_input` only if the WHOLE instruction is invalid.
   - For Mixed Instructions (valid + invalid parts): ONLY generate API calls for the valid parts. Do NOT output `error_input` for the invalid parts in a mixed batch, just skip them.

### RULES:
1. First output your reasoning inside <Thought> tags.
2. Then output the final API calls inside <Code> tags.
3. If NO part of the instruction is valid, the code block should contain: error_input
4. Do NOT output markdown formatting (```).

### EXAMPLE:
User: Turn on the living room light and the bathroom toaster.
(Assume living room light exists, but bathroom toaster does not)
<Thought>
1. Living room light: Exists. Valid.
2. Bathroom toaster: Does not exist. Invalid.
Conclusion: Generate code for light, skip toaster.
</Thought>
<Code>
living_room.light.turn_on()
</Code>
"""

# --- IC 任务转换逻辑 ---
def transform_ic_input(case):
    if "one" not in case.get("id", ""): return None
    output_raw = case.get("output", [])
    cmd = output_raw[0] if isinstance(output_raw, list) and len(output_raw) > 0 else (output_raw if isinstance(output_raw, str) else "")
    if "." not in cmd: return None
    room_key = cmd.split(".")[0]
    room_name = room_key.replace("_", " ")
    
    # 支持 in 和 on
    pattern = re.compile(r'\b(?:in|on)\s+(?:the\s+)?' + re.escape(room_name) + r'\b', re.IGNORECASE)
    new_input = pattern.sub("here", case.get("input", ""))
    
    return {
        "new_input": new_input,
        "location_context": f"User is located in the {room_name}.",
        "room_name": room_name,
        "room_key": room_key
    }

# --- V8 清洗函数 (提取 <Code>) ---
def clean_generated_text(text):
    # 1. 尝试提取 CoT 代码块
    code_pattern = r"<Code>(.*?)</Code>"
    match = re.search(code_pattern, text, re.DOTALL)
    if match:
        content = match.group(1).strip()
        return _clean_inner_logic(content)
    # 2. Fallback
    return _clean_inner_logic(text)

def _clean_inner_logic(text):
    stop_signals = ["<User instructions:>", "-------------------------------", "<home_state>", "<device_method>", "User:", "Machine instructions:", "Example:", "<Thought>"]
    min_index = len(text)
    for signal in stop_signals:
        idx = text.find(signal)
        if idx != -1 and idx < min_index: min_index = idx
    text = text[:min_index].strip()
    
    if "." in text and "(" in text and "error_input" in text:
        lines = text.split('\n')
        valid_lines = [line.strip() for line in lines if "." in line and "(" in line]
        text = "\n".join(valid_lines) if valid_lines else "error_input"
    return text

# --- 设备索引 ---
def build_device_index(home_status):
    index = {}
    for room, devices in home_status.items():
        if room == "VacuumRobot":
            if "vacuum" not in index: index["vacuum"] = []
            index["vacuum"].append(room)
            continue
        if isinstance(devices, dict):
            for device_name in devices.keys():
                if device_name == "room_name": continue
                clean_name = device_name.replace("_", " ").lower()
                if clean_name not in index: index[clean_name] = []
                index[clean_name].append(room)
                aliases = []
                if "light" in clean_name: aliases.extend(["lamp", "lights"])
                if "air conditioner" in clean_name: aliases.extend(["ac", "a/c"])
                if "humidifier" in clean_name: aliases.extend(["humid"])
                if "curtain" in clean_name: aliases.extend(["blind"])
                for alias in aliases:
                    if alias not in index: index[alias] = []
                    index[alias].append(room)
    return index

# --- SALK 上下文提取 ---
def extract_context_smart_v6(user_input, all_rooms_list, device_index):
    user_input_lower = user_input.lower()
    explicit_rooms = set()
    
    for room in all_rooms_list:
        if room.lower() in user_input_lower: explicit_rooms.add(room)
            
    implicit_rooms = set()
    for device_kw, rooms in device_index.items():
        if re.search(r'\b' + re.escape(device_kw) + r'\b', user_input_lower):
            for room in rooms: implicit_rooms.add(room)

    final_rooms = explicit_rooms.copy()
    if not explicit_rooms: final_rooms.update(implicit_rooms)
    
    if len(final_rooms) > 3:
        priority_zones = {"living room", "master bedroom", "kitchen", "corridor"}
        keep_rooms = explicit_rooms.copy()
        for r in (final_rooms - explicit_rooms):
            if r in priority_zones: keep_rooms.add(r)
        if keep_rooms: final_rooms = keep_rooms

    return final_rooms

def chang_json2str(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if isinstance(state[room], dict):
                state_str += "  state: " + str(state[room].get("state", "N/A")) + "\n"
                if "attributes" in state[room]:
                    for k, v in state[room]["attributes"].items():
                        state_str += f"  {k}: {v.get('value', 'N/A')}\n"
        else:
            for device, val in state[room].items():
                if device == "room_name": continue
                state_str += f"  {device}\n    state: {val.get('state', 'N/A')}\n"
                if "attributes" in val:
                    for k, v in val["attributes"].items():
                        state_str += f"    {k}: {v.get('value', 'N/A')}\n"

    method_str = ""
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        method_str += f"{room_prefix}{method['device_name']}.{method['operation']}("
        if len(method["parameters"]) > 0:
            params = [f"{p['name']}:{p['type']}" for p in method["parameters"]]
            method_str += ",".join(params)
        method_str += ");"
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"

# --- Dataset: Zero-shot (SALK V8 + IC) ---
class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f: lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home: lines_home = f_home.readlines()
        
        home_status_full = {} 
        for line in lines_home:
            data = json.loads(line)
            home_status_full[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status_full: continue

                ic_data = transform_ic_input(case)
                if ic_data is None: continue

                user_input = ic_data["new_input"]
                full_state = home_status_full[case["home_id"]]["home_status"]
                full_methods = home_status_full[case["home_id"]]["method"]
                
                device_index = build_device_index(full_state)
                relevant_rooms = extract_context_smart_v6(user_input, list(full_state.keys()), device_index)

                # 强制加入 Location
                if ic_data["room_key"] in full_state: relevant_rooms.add(ic_data["room_key"])
                if not relevant_rooms: relevant_rooms = set(full_state.keys())
                else:
                    if "VacuumRobot" in full_state: relevant_rooms.add("VacuumRobot")

                filtered_state = {k:v for k,v in full_state.items() if k in relevant_rooms}
                filtered_methods = [m for m in full_methods if m["room_name"] in relevant_rooms or m["room_name"] == "None"]
                
                state_str, method_str = chang_json2str(filtered_state, filtered_methods)
                
                home_status_case = "<home_state>\n"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n"+ method_str + "\n" + "</device_method>\n"
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions you need to reply to.\n" + \
                                        "<User instructions:> \n" + \
                                        user_input + "\n" + \
                                        "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + user_instruction_case
                final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT, user_content)
                self.data.append({"input_text": final_input, "output": case["output"], "type": case.get("type", "normal")})
            except: continue
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

# --- Few-shot V8 IC ---
class home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f: lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home: lines_home = f_home.readlines()
        home_status_full = {}
        for line in lines_home:
            data = json.loads(line)
            home_status_full[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        ex_path = os.path.join(code_dir, "example1.txt")
        if not os.path.exists(ex_path): ex_path = os.path.join(code_dir, "example.txt")
        examples = open(ex_path, "r").read() if os.path.exists(ex_path) else ""

        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status_full: continue
                
                ic_data = transform_ic_input(case)
                if ic_data is None: continue

                user_input = ic_data["new_input"]
                full_state = home_status_full[case["home_id"]]["home_status"]
                full_methods = home_status_full[case["home_id"]]["method"]
                
                device_index = build_device_index(full_state)
                relevant_rooms = extract_context_smart_v6(user_input, list(full_state.keys()), device_index)
                
                if ic_data["room_key"] in full_state: relevant_rooms.add(ic_data["room_key"])
                if not relevant_rooms: relevant_rooms = set(full_state.keys())
                else: 
                    if "VacuumRobot" in full_state: relevant_rooms.add("VacuumRobot")

                filtered_state = {k:v for k,v in full_state.items() if k in relevant_rooms}
                filtered_methods = [m for m in full_methods if m["room_name"] in relevant_rooms or m["room_name"] == "None"]

                state_str, method_str = chang_json2str(filtered_state, filtered_methods)
                
                home_status_case = "<home_state>\n"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n"+ method_str + "\n" + "</device_method>\n"
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions...\n<User instructions:> \n" + \
                                        user_input + "\n" + "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + examples + user_instruction_case
                final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT, user_content)
                self.data.append({"input_text": final_input, "output": case["output"], "type": case.get("type", "normal")})
            except: continue
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

# --- RAG V8 IC ---
class rag_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, model_name_for_rag):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        filename_candidates = [f"{model_name_for_rag}_rag_test_data.json", "rag_test_data.json"]
        rag_path = next((os.path.join(dataset_dir, f) for f in filename_candidates if os.path.exists(os.path.join(dataset_dir, f))), None)
        if not rag_path:
            cand = glob.glob(os.path.join(dataset_dir, "*rag_test_data.json"))
            if cand: rag_path = cand[0]
        if not rag_path: raise FileNotFoundError("RAG dataset not found")
        print(f"Loading RAG: {rag_path}")
        with open(rag_path, "r") as f: raw_data = json.load(f)
        self.data = []
        for item in raw_data:
            ic_data = transform_ic_input(item)
            if ic_data is None: continue
            
            original_input_str = item["input"]
            pattern = re.compile(r'\b(?:in|on)\s+(?:the\s+)?' + re.escape(ic_data['room_name']) + r'\b', re.IGNORECASE)
            modified_str = pattern.sub("here", original_input_str)
            
            insert_marker = "<User instructions:>"
            if insert_marker in modified_str:
                parts = modified_str.split(insert_marker)
                final_rag_input = parts[0] + insert_marker + "\n" + ic_data['location_context'] + parts[1]
            else:
                final_rag_input = ic_data['location_context'] + "\n" + modified_str

            if "system" in item["input"]: final_input = final_rag_input
            else: final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT, final_rag_input)
            
            self.data.append({"input_text": final_input, "output": item["output"], "type": item.get("type", "normal")})
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

def model_test(model_name, use_rag=False, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=64):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_enabled = local_rank != -1
    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub_dirs = {"llama": "llama3-8b-Instruct", "qwen": "Qwen2.5-7B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3", "gemma": "Gemma-7B-Instruct-v0.3"}
    base_model_path_name = sub_dirs.get(model_name, model_name)
    base_model_dir = os.path.join(PROJECT_ROOT, "models", base_model_path_name)
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_sft")
    
    if rank == 0: print(f"Loading Model: {base_model_dir}")
    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try: tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except: tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map=None if ddp_enabled else "auto", trust_remote_code=True)
    if ddp_enabled: model.to(device)
    if use_finetuned and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_dir)
        if ddp_enabled: model.to(device)

    if rank == 0: print("Loading dataset (SALK V8 + IC)...")
    if use_rag: test_dataset = rag_home_assistant_dataset(tokenizer, base_model_path_name)
    elif use_few_shot: test_dataset = home_assistant_dataset(tokenizer)
    else: test_dataset = no_few_shot_home_assistant_dataset(tokenizer)
    
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4)
    
    res = []
    iterator = tqdm(test_loader, disable=(rank != 0))
    start_time = time.time()
    
    with torch.inference_mode():
        for inputs_str, output_texts, types in iterator:
            inputs = tokenizer(list(inputs_str), return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            # V8 使用 CoT, 需要增加 max_new_tokens
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=False, 
                repetition_penalty=1.1, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            response = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
            for i in range(len(generated_texts)):
                cleaned = clean_generated_text(generated_texts[i])
                res.append({"generated": cleaned, "expected": output_texts[i], "type": types[i]})
            
    if rank == 0: print(f"Time: {time.time() - start_time:.2f}s")
    
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_rag: mode_parts.append("rag")
    elif use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    mode_suffix = "_".join(mode_parts)
    if test_type and test_type != "normal": mode_suffix += f"_{test_type}"
    mode_suffix += "_SALK_V8_IC"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_part_{rank}.json")
    with open(part_file, "w") as f: f.write(json.dumps(res, indent=4, ensure_ascii=False))
    
    if ddp_enabled: dist.barrier()
    if rank == 0:
        final_res = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_part_*.json")):
            try:
                with open(file_path, "r") as f: final_res.extend(json.load(f))
                os.remove(file_path)
            except: pass
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_test_result.json")
        with open(final_file, "w") as f: f.write(json.dumps(final_res, indent=4, ensure_ascii=False))
        print(f"Saved: {final_file}")
    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"])
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true") 
    parser.add_argument("--test_type", type=str, default="normal")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()
    if os.environ.get("LOCAL_RANK") is None: os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    model_test(args.model_name, args.use_rag, args.use_few_shot, args.use_finetuned, args.test_type, args.batch_size)
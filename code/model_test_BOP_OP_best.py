import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2TokenizerFast
from peft import PeftModel

# --- 导入 BOP 智能体 ---
try:
    from BOP1Agent import InstructionSplitterV15 as InstructionSplitter
    from BOP2Agent import EnvironmentPerceptionAgent
except ImportError:
    print("Error: Could not import BOP Agents. Please ensure BOP1Agent.py and BOP2Agent.py are in the same directory.")
    exit(1)

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 硬编码 System Prompts (V2: 强化格式约束)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a smart home control agent. 
Your task is to convert a single atomic User Instruction into an executable Python-style API call based strictly on the provided Home State and Device Methods.

RULES:
1. **Atomic Execution**: The input is a single instruction. Output exactly ONE executable API call if valid.
2. **Grounding Verification**: 
   - If the target device or room does not exist in the provided Home State, output exactly: error_input
   - If the device attribute or method is not supported (e.g., setting temperature on a light), output exactly: error_input
   - If the parameter value is out of the valid range provided in Home State, output exactly: error_input
3. Output ONLY the code or the error keyword. Do NOT output explanations.
4. Do not include markdown code blocks.
5. **STOP generating immediately after the code.
"""

GEMMA_CUSTOM_PROMPT_BOP = """You are a smart home code generator.
Convert the atomic User Instruction into an executable Python API call.

STRICT FORMATTING:
1. Output ONLY the code line. No markdown.
2. DO NOT include parameter names (e.g. use `set_temp(20)`, not `temp=20`).
3. If invalid, output: error_input

Example:
User: Turn on the kitchen light.
Code:
kitchen.light.turn_on()
"""

# ==========================================
# 2. ICL Examples (保持原子化)
# ==========================================
BOP_FEW_SHOT_EXAMPLES = """Here are a few examples, your output format should be consistent with the results provided in the example:
<example1>
<User instructions:> Lower the air conditioner temperature in the guest bedroom by 1 degree.
<Machine instructions:> {guest_bedroom.air_conditioner.set_temperature(26)}
</example1>
<example2>
<User instructions:> Turn on the light in the living room.
<Machine instructions:> {living_room.light.turn_on()}
</example2>
<example3>
<User instructions:> Set the brightness of the light to 50 in the kitchen.
<Machine instructions:> {kitchen.light.set_brightness(50)}
</example3>
<example4>
<User instructions:> Turn on the humidifier in the garage.
<Machine instructions:> {error_input} (If there is no device or if there is no attribute or method that can be operated on the device, please output as "error_input".)
</example4>
<example5>
<User instructions:> Set the temperature of the light in the study room to 20 degrees.
<Machine instructions:> {error_input} (If there is no device or if there is no attribute or method that can be operated on the device, please output as "error_input".)
</example5>
"""

# ==========================================
# 3. 辅助函数 (V2: 增强版清洗)
# ==========================================

def chang_json2str(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if isinstance(state[room], dict):
                state_str += "  state: " + str(state[room].get("state", "N/A")) + "\n"
                if "attributes" in state[room]:
                    for attribute in state[room]["attributes"].keys():
                        val = state[room]["attributes"][attribute]
                        state_str += "  " + attribute + ": " + str(val.get("value", "N/A"))
                        if "options" in val:
                            state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val:
                            state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else:
                            state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name": continue
                device_obj = state[room][device]
                state_str += "  " + device + "\n"
                state_str += "    state: " + str(device_obj.get("state", "N/A")) + "\n"
                if "attributes" in device_obj:
                    for attribute in device_obj["attributes"].keys():
                        val = device_obj["attributes"][attribute]
                        state_str += "    " + attribute + ": " + str(val.get("value", "N/A"))
                        if "options" in val:
                            state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val:
                            state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else:
                            state_str += "\n"

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
        if "gemma" in tokenizer.name_or_path.lower():
            combined_content = f"{system}\n\n{user}"
            messages = [{"role": "user", "content": combined_content}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        return f"{system}\n\n{user}"

def clean_generated_text(text):
    """
    V2 暴力清洗：修复 'STOP', 参数名, 缺少引号等问题
    """
    # 1. 截断
    stop_signals = ["<User", "User:", "Machine", "Example", "STOP", "Note:"]
    for s in stop_signals:
        if s in text:
            text = text.split(s)[0]
    
    text = text.strip().replace("```python", "").replace("```", "")
    
    # 2. 提取有效行
    lines = text.split('\n')
    valid_code = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 优先匹配 error_input
        if "error_input" in line:
            return "error_input"
            
        # 匹配 API 调用结构: word.word.word(...)
        if re.match(r'^[\w]+\.[\w]+\.[\w]+\(.*\)', line):
            valid_code = line
            break
            
    if not valid_code:
        return "error_input" # 兜底：如果没生成代码，视为错误
        
    # 3. 参数清洗 (外科手术式修复)
    # 提取括号内的内容
    match = re.match(r'^([\w\.]+)\((.*)\)', valid_code)
    if match:
        prefix = match.group(1)
        args_str = match.group(2)
        
        # A. 移除参数名 (如 "temperature=24" -> "24", "degree:80" -> "80")
        args_str = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_str)
        
        # B. 修复缺失引号 (如 set_mode(cool) -> set_mode('cool'))
        # 逻辑：如果是纯字母且不是 true/false/none，加上引号
        new_args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg: continue
            # 如果是纯字母单词，且没有引号，且不是数字
            if re.match(r'^[a-zA-Z_]+$', arg):
                if arg.lower() not in ['true', 'false', 'none']:
                    arg = f"'{arg}'"
            new_args.append(arg)
            
        valid_code = f"{prefix}({', '.join(new_args)})"

    return valid_code

# ==========================================
# 4. BOP Dataset 类
# ==========================================
class bop_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, use_few_shot=False):
        self.tokenizer = tokenizer
        self.use_few_shot = use_few_shot
        
        print("Initializing BOP Agents...")
        self.splitter_agent = InstructionSplitter()
        self.perception_agent = EnvironmentPerceptionAgent()
        
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home:
            lines_home = f_home.readlines()
        
        self.data_store = {} 
        for line in lines_home:
            d = json.loads(line)
            self.data_store[d["home_id"]] = {"state": d["home_status"], "method": d["method"]}
            
        with open(os.path.join(dataset_dir, "test_data_copy.jsonl"), "r") as f:
            lines = f.readlines()
            
        self.samples = []
        
        print(f"Processing {len(lines)} raw samples with BOP Architecture...")
        
        for i in tqdm(range(len(lines)), desc="Building BOP Prompts"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                raw_input = case["input"]
                
                # BOP1 Split
                atoms = self.splitter_agent.process(raw_input)
                if not atoms: atoms = [raw_input]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    # BOP2 Sense
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    if p_res["perception"]["room_found"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot":
                        target_rooms.add("VacuumRobot")
                        
                    full_state = self.data_store[home_id]["state"]
                    full_methods = self.data_store[home_id]["method"]
                    
                    partial_state = {}
                    partial_methods = []
                    
                    for r in target_rooms:
                        if r in full_state: partial_state[r] = full_state[r]
                        if r == "VacuumRobot" and "VacuumRobot" in full_state:
                             partial_state["VacuumRobot"] = full_state["VacuumRobot"]

                    for m in full_methods:
                        if m["room_name"] in target_rooms or m["room_name"] == "None":
                            partial_methods.append(m)

                    state_str, method_str = chang_json2str(partial_state, partial_methods)
                    
                    home_status_case = f"<home_state>\n{state_str}</home_state>\n"
                    device_method_case = f"<device_method>\n{method_str}</device_method>\n"
                    user_instruction_case = f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
                    
                    examples = BOP_FEW_SHOT_EXAMPLES if self.use_few_shot else ""
                    user_content = home_status_case + device_method_case + examples + user_instruction_case
                    
                    if "gemma" in self.tokenizer.name_or_path.lower():
                        combined = f"{GEMMA_CUSTOM_PROMPT_BOP}\n\n{user_content}"
                        final_input = self.tokenizer.apply_chat_template([{"role": "user", "content": combined}], add_generation_prompt=True, tokenize=False)
                    else:
                        final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, user_content)
                    
                    self.samples.append({
                        "input_text": final_input,
                        "original_index": i,
                        "atom_index": atom_idx,
                        "ground_truth_full": case["output"],
                        "type": case.get("type", "normal")
                    })

            except Exception as e:
                continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 5. 主测试逻辑
# ==========================================
def model_test(model_name, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=64):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
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

    if rank == 0: print(f"Loading Model: {base_model_dir} (BOP Mode)")

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
    model.eval()

    if rank == 0: print("Preparing BOP Dataset...")
    test_dataset = bop_home_assistant_dataset(tokenizer, use_few_shot=use_few_shot)
    
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    def collate_fn(batch): return batch
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, collate_fn=collate_fn)

    temp_results = []
    iterator = tqdm(test_loader, disable=(rank != 0))
    
    with torch.inference_mode():
        for batch in iterator:
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=128, 
                do_sample=False, 
                repetition_penalty=1.1, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            response = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                cleaned_text = clean_generated_text(text)
                temp_results.append({
                    "generated": cleaned_text,
                    "original_index": batch[i]["original_index"],
                    "atom_index": batch[i]["atom_index"],
                    "ground_truth_full": batch[i]["ground_truth_full"],
                    "type": batch[i]["type"]
                })

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "bop_sft" if use_finetuned else "bop_base"
    if use_few_shot: mode_suffix += "_few_shot"
    else: mode_suffix += "_zero_shot"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        print("Merging and Reconstructing...")
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_atoms:
            grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            combined_code = ""
            for atom in atoms:
                code = atom["generated"].strip()
                code = code.replace("{", "").replace("}", "") 
                if code and not code.endswith(","): code += ","
                combined_code += code
            
            final_results.append({
                "generated": combined_code,
                "expected": atoms[0]["ground_truth_full"],
                "type": atoms[0]["type"]
            })
            
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"BOP Evaluation Complete. Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    model_test(args.model_name, use_few_shot=args.use_few_shot, use_finetuned=args.use_finetuned, batch_size=args.batch_size)
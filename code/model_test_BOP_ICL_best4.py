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
# 1. 系统提示词 (针对感知架构优化)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a smart home control agent. 
You will receive a Partial Home State (only relevant devices) and an Atomic User Instruction.

STRICT EXECUTION RULES:
1. **Grounding Only**: You MUST only use devices, attributes, and methods listed in the provided <home_state> and <device_method>.
2. **Verification**: 
   - If the device or room is MISSING in <home_state>, output: error_input
   - If the attribute (e.g., intensity) is NOT in the <home_state> for that device, output: error_input
   - If the operation is NOT in <device_method>, output: error_input
   - If the value is outside the (range) provided, output: error_input
3. **Format**: Output exactly `{room.device.method(params)}` or `{error_input}`.
4. No explanations. STOP immediately after outputting the curly braces.
"""

GEMMA_CUSTOM_PROMPT_BOP = """You are a smart home code generator.
Convert the atomic User Instruction into a single API call based ONLY on the provided state.
If invalid or missing in state, output: {error_input}
"""

# ==========================================
# 2. 像素级对齐的 ICL 示例 (基于 Home13 真实数据微调)
# ==========================================
BOP_ICL_REFINED = """Here are examples of strict state-based reasoning:

<example1>
<home_state>
master_bedroom:
  aromatherapy
    state: off
</home_state>
<device_method>
master_bedroom.aromatherapy.turn_on(); master_bedroom.aromatherapy.turn_off();
</device_method>
<User instructions:> Set the intensity of the aromatherapy in the master bedroom to 100.
<Machine instructions:> {error_input}
</example1>

<example2>
<home_state>
bathroom:
  light
    state: on
    brightness: 83 (range: 0 - 100)
</home_state>
<device_method>
bathroom.light.set_brightness(brightness:int);
</device_method>
<User instructions:> Decrease the heating temperature in the bathroom by 9 degrees.
<Machine instructions:> {error_input}
</example2>

<example3>
<home_state>
living_room:
  air_conditioner
    state: off
    temperature: 20 (range: 16 - 30)
</home_state>
<device_method>
living_room.air_conditioner.set_temperature(temperature:int);
</device_method>
<User instructions:> Set the AC in the living room to 25 degrees.
<Machine instructions:> {living_room.air_conditioner.set_temperature(25)}
</example3>

<example4>
<home_state>
study_room:
  heating
    state: off
    temperature: 83 (range: 30 - 100)
</home_state>
<device_method>
study_room.heating.set_temperature(temperature:int);
</device_method>
<User instructions:> Set the study room heating to 20 degrees.
<Machine instructions:> {error_input}
</example4>
"""

# ==========================================
# 3. 辅助函数
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
                        if "options" in val: state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val: state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else: state_str += "\n"
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
                        if "options" in val: state_str += " (options" + str(val["options"]) + ")\n"
                        elif "lowest" in val: state_str += " (range: " + str(val.get("lowest")) + " - " + str(val.get("highest")) + ")\n"
                        else: state_str += "\n"

    method_str = ""
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        method_str += f"{room_prefix}{method['device_name']}.{method['operation']}("
        if len(method["parameters"]) > 0:
            params = [f"{p['name']}:{p['type']}" for p in method["parameters"]]
            method_str += ",".join(params)
        method_str += "); "
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
    # 针对 ICL 可能产生的推理描述或花括号进行清洗
    stop_signals = ["<User", "User:", "Machine", "Example", "STOP", "Note:", "Reason:"]
    for s in stop_signals:
        if s in text: text = text.split(s)[0]
    
    text = text.strip().replace("```python", "").replace("```", "")
    
    # 提取花括号中的内容
    bracket_match = re.search(r'\{(.*?)\}', text)
    if bracket_match: text = bracket_match.group(1)

    lines = text.split('\n')
    valid_code = None
    for line in lines:
        line = line.strip()
        if not line: continue
        if "error_input" in line: return "error_input"
        if re.match(r'^[\w]+\.[\w]+\.[\w]+\(.*\)', line):
            valid_code = line
            break
    if not valid_code: return "error_input"
    
    # 参数手术式清洗
    match = re.match(r'^([\w\.]+)\((.*)\)', valid_code)
    if match:
        prefix, args_str = match.group(1), match.group(2)
        args_str = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_str)
        new_args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg: continue
            if re.match(r'^[a-zA-Z_]+$', arg) and arg.lower() not in ['true', 'false', 'none']:
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
        print(f"Building BOP Dataset (Few-shot: {use_few_shot})...")
        
        for i in tqdm(range(len(lines)), desc="Processing BOP Samples"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                # 【关键逻辑修改】：先拆分，再针对每个原子指令进行感知
                atoms = self.splitter_agent.process(case["input"])
                if not atoms: atoms = [case["input"]]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    # BOP2 Sense: 只对当前原子指令进行环境感知
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    if p_res["perception"]["room_found"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot":
                        target_rooms.add("VacuumRobot")
                    
                    full_state = self.data_store[home_id]["state"]
                    full_methods = self.data_store[home_id]["method"]
                    
                    # 构建 Partial Context
                    partial_state = {r: full_state[r] for r in target_rooms if r in full_state}
                    partial_methods = [m for m in full_methods if m["room_name"] in target_rooms or m["room_name"] == "None"]

                    state_str, method_str = chang_json2str(partial_state, partial_methods)
                    
                    user_content = f"<home_state>\n{state_str}</home_state>\n<device_method>\n{method_str}</device_method>\n"
                    if self.use_few_shot: user_content += BOP_ICL_REFINED
                    user_content += f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
                    
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
            except Exception: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 5. 主测试函数
# ==========================================
def model_test(model_name, use_few_shot=False, use_finetuned=False, batch_size=32, cuda_devices="0,1"):
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

    if rank == 0: print(f"Loading {model_name} in BOP-ICL Mode...")

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

    test_dataset = bop_home_assistant_dataset(tokenizer, use_few_shot=use_few_shot)
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    temp_results = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, disable=(rank != 0)):
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs, max_new_tokens=128, do_sample=False, 
                repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            response = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                cleaned = clean_generated_text(text)
                temp_results.append({
                    "generated": cleaned,
                    "original_index": batch[i]["original_index"],
                    "atom_index": batch[i]["atom_index"],
                    "ground_truth_full": batch[i]["ground_truth_full"],
                    "type": batch[i]["type"]
                })

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "bop_icl" if use_few_shot else "bop_zero"
    if use_finetuned: mode_str += "_sft"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_str}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_str}_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_atoms: grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            # 合并原子指令生成的代码，确保格式符合预期
            codes = [a["generated"].replace("{", "").replace("}", "") for a in atoms if a["generated"]]
            combined_code = ",".join(codes)
            if combined_code and not combined_code.endswith(","): combined_code += ","
            
            final_results.append({
                "generated": combined_code,
                "expected": atoms[0]["ground_truth_full"],
                "type": atoms[0]["type"]
            })
            
        final_file = os.path.join(output_dir, f"{model_name}_{mode_str}_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"Evaluation Complete. Final result saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    args = parser.parse_args()
    
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    model_test(args.model_name, use_few_shot=args.use_few_shot, use_finetuned=args.use_finetuned, batch_size=args.batch_size, cuda_devices=args.cuda_devices)
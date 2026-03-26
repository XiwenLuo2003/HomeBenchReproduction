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
# 1. Gemma ICL (Few-shot) Prompt Components
# ==========================================

# System Role Definition
GEMMA_ICL_SYSTEM = """### Role
You are a Python API Generator for a Smart Home.

### Task
Convert the User Instruction into a single Python API call by selecting the correct method from the "Available APIs" list.

### Constraints
1. **Strict Matching**: You must use the exact device and method names from "Available APIs".
2. **Entity Check**: If the target device or room is NOT listed in "Current State", output: error_input
3. **Attribute Check**: If the attribute is not supported, output: error_input
4. **Format**: Output ONLY the Python code line.
"""

# Few-shot Examples (Static Examples to guide the model)
# 注意：这里的示例展示了正确的格式，以及 error_input 的用法
BOP_FEW_SHOT_EXAMPLES = """### Examples

User: Lower the air conditioner temperature in the guest bedroom by 1 degree.
Response:
guest_bedroom.air_conditioner.set_temperature(26)

User: Turn on the light in the living room.
Response:
living_room.light.turn_on()

User: Set the brightness of the light to 50 in the kitchen.
Response:
kitchen.light.set_brightness(50)

User: Turn on the humidifier in the garage.
Response:
error_input

User: Set the temperature of the light in the study room to 20 degrees.
Response:
error_input
"""

# ==========================================
# 2. 辅助函数
# ==========================================

def chang_json2str(state, methods):
    """
    构建 "菜单" 形式的 Context (与 OP 模式保持一致)
    """
    # 1. 构建 Current State
    state_list = []
    for room in state.keys():
        if room == "VacuumRobot":
             state_list.append("VacuumRobot")
        else:
            for device in state[room].keys():
                if device == "room_name": continue
                state_list.append(f"{room}.{device}")
    
    if not state_list:
        state_str = "No devices found."
    else:
        state_str = "- " + "\n- ".join(state_list)

    # 2. 构建 Available APIs
    method_list = []
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        
        # 构建函数签名
        params = []
        for p in method["parameters"]:
            params.append(p["name"]) 
        param_str = ", ".join(params)
        
        signature = f"{room_prefix}{method['device_name']}.{method['operation']}({param_str})"
        method_list.append(signature)
    
    if not method_list:
        method_str = "No methods available."
    else:
        method_str = "- " + "\n- ".join(method_list)
    
    return state_str, method_str

def clean_generated_text(text):
    """
    针对 Gemma 输出的清洗函数 (复用 V5 版本的修复逻辑)
    """
    # 1. 基础清理
    text = text.replace("```python", "").replace("```", "").strip()
    
    # 2. 移除前缀干扰
    text = re.sub(r'^\**Answer:\**\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\**Code:\**\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\**Response:\**\s*', '', text, flags=re.IGNORECASE)
    
    # 3. 修复 Tokenizer 空格
    text = re.sub(r'\s*\.\s*', '.', text)
    text = re.sub(r'_\s+', '_', text)
    text = re.sub(r'\s+_', '_', text)
    text = re.sub(r'\s+\(', '(', text)

    # 4. 优先匹配 error_input
    if "error_input" in text.lower():
        return "error_input"
    
    # 5. 提取代码行
    lines = text.split('\n')
    valid_code = "error_input" 
    for line in lines:
        line = line.strip()
        if not line: continue
        if line.startswith(("#", "//", "User", "Context", "Available")): continue
        if "(" in line and ")" in line and "." in line:
            valid_code = line
            break
            
    if valid_code == "error_input":
        matches = re.findall(r'([a-z0-9_]+\.[a-z0-9_]+\.[a-z0-9_]+\(.*\))', text, re.IGNORECASE)
        if matches:
            valid_code = matches[0]
        else:
            return "error_input"

    # 6. 参数清洗
    match = re.match(r'^([\w\.]+)\((.*)\)', valid_code)
    if match:
        prefix = match.group(1)
        args_str = match.group(2)
        args_str = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_str)
        
        new_args = []
        for arg in args_str.split(','):
            arg = arg.strip()
            if not arg: continue
            if re.match(r'^[a-zA-Z_]+$', arg) and arg.lower() not in ['true', 'false', 'none', 'auto']:
                arg = f"'{arg}'"
            new_args.append(arg)
            
        valid_code = f"{prefix}({', '.join(new_args)})"

    return valid_code

# ==========================================
# 3. BOP Dataset 类 (ICL Adapter)
# ==========================================
class bop_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
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
            
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f:
            lines = f.readlines()
            
        self.samples = []
        print(f"Processing {len(lines)} raw samples with BOP Architecture (ICL Mode)...")
        
        for i in tqdm(range(len(lines)), desc="Building Prompts"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                raw_input = case["input"]
                
                atoms = self.splitter_agent.process(raw_input)
                if not atoms: atoms = [raw_input]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    
                    if p_res["extracted"]["room"]:
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
                    
                    if not partial_state:
                        state_str = "No devices found."
                        method_str = "No methods available."

                    # --- Prompt 构建 (ICL 模式) ---
                    # 结构: System -> APIs -> State -> Examples -> User -> Response
                    prompt_content = (
                        f"{GEMMA_ICL_SYSTEM}\n\n"
                        f"### Available APIs\n"
                        f"{method_str}\n\n"
                        f"### Current State\n"
                        f"{state_str}\n\n"
                        f"{BOP_FEW_SHOT_EXAMPLES}\n\n"
                        f"User: {atom_instr}\n"
                        f"Response:\n" 
                    )
                    
                    messages = [{"role": "user", "content": prompt_content}]
                    
                    final_input = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    self.samples.append({
                        "input_text": final_input,
                        "original_index": i,
                        "atom_index": atom_idx,
                        "ground_truth_full": case["output"],
                        "type": case.get("type", "normal")
                    })

            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 4. 主测试逻辑
# ==========================================
def model_test(model_name, batch_size=32):
    # DDP Setup
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

    # Model Paths
    sub_dirs = {"llama": "llama3-8b-Instruct", "qwen": "Qwen2.5-7B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3", "gemma": "Gemma-7B-Instruct-v0.3"}
    base_model_path_name = sub_dirs.get(model_name, model_name)
    base_model_dir = os.path.join(PROJECT_ROOT, "models", base_model_path_name)
    
    if rank == 0: print(f"Loading Model: {base_model_dir} (BOP Mode - Gemma ICL/Few-Shot)")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map=None if ddp_enabled else "auto", trust_remote_code=True)
    if ddp_enabled: model.to(device)
    model.eval()

    # Dataset
    test_dataset = bop_home_assistant_dataset(tokenizer)
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    
    def collate_fn(batch): return batch
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Inference
    temp_results = []
    iterator = tqdm(test_loader, disable=(rank != 0))
    
    with torch.inference_mode():
        for batch in iterator:
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64, 
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

    # Save
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    part_file = os.path.join(output_dir, f"{model_name}_bop_icl_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        print("Merging atomic results...")
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_bop_icl_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_atoms:
            grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            combined_code_list = []
            for atom in atoms:
                combined_code_list.append(atom["generated"].strip())
            
            final_code_str = ",".join(combined_code_list)
            if final_code_str and not final_code_str.endswith(","):
                final_code_str += ","
            
            final_results.append({
                "generated": final_code_str,
                "expected": atoms[0]["ground_truth_full"],
                "type": atoms[0]["type"]
            })
            
        final_file = os.path.join(output_dir, f"{model_name}_bop_icl_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"BOP ICL Evaluation Complete. Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gemma") 
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    model_test(args.model_name, batch_size=args.batch_size)
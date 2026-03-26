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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
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
# 1. 系统提示词 (极致严苛的执行指令)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a high-precision Smart Home Execution Engine. 
You transform a single atomic instruction into a Python API call based on strict physical validation.

### MANDATORY PROTOCOL ###
1. CHECK: Does the room and device exist in <home_state>?
2. CHECK: Does the device support the operation in <device_method>?
3. CHECK: Is the value within the [lowest, highest] range in <home_state>?
4. CALCULATION: If 'increase' or 'decrease' is requested, calculate the final value based on the current state.

### REJECTION RULE ###
If ANY check fails, you MUST output exactly: {error_input}
Otherwise, output: {room.device.method(final_value)}

CRITICAL: Do NOT use values or room names from the examples below. Only use them to learn the output format and validation logic.
"""

# ==========================================
# 2. 深度重构的 ICL Examples (聚焦物理冲突)
# ==========================================
BOP_FEW_SHOT_EXAMPLES = """### LOGIC EXAMPLES FOR YOUR REFERENCE ###

[Instruction] Set the brightness of the bedroom light to 120.
[Reasoning] bedroom.light exists, but max brightness is 100. 120 is out of range.
[Result] {error_input}

[Instruction] Turn on the humidifier in the garage.
[Reasoning] garage exists, but it only contains [light, fan]. humidifier is missing.
[Result] {error_input}

[Instruction] Increase the air conditioner temperature in the study by 2 degrees.
[Reasoning] study.air_conditioner temperature is 24, range [16, 30]. 24+2=26. 26 is valid.
[Result] {study.air_conditioner.set_temperature(26)}

[Instruction] Close the door in the kitchen.
[Reasoning] kitchen.door exists, but methods only support [lock, unlock]. close is invalid.
[Result] {error_input}
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"

def clean_generated_text(text):
    # 暴力截断所有解释性后缀
    stop_signals = ["###", "[Instruction]", "[Reasoning]", "[Result]", "User:", "Machine:", "Note:", "{", "}"]
    
    # 提取有效代码段
    match = re.search(r'\{(.*?)\}', text)
    if match:
        text = match.group(1).strip()
    else:
        # 如果没有花括号，尝试按行截断
        for s in stop_signals[:-2]: # 不包括花括号
            if s in text:
                text = text.split(s)[0]
    
    text = text.strip()
    
    if "error_input" in text.lower():
        return "error_input"
    
    # 清理参数名 (temperature=20 -> 20) 并处理可能的数学式残留
    def param_fixer(m):
        prefix = m.group(1)
        params = m.group(2)
        params = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', params)
        # 如果模型输出了计算式，执行最终计算
        if re.search(r'[\+\-\*\/]', params):
            try:
                # 简单清洗非数字字符后再计算
                math_expr = re.sub(r'[^\d\+\-\*\/\.]', '', params)
                params = str(int(eval(math_expr)))
            except: pass
        return f"{prefix}({params})"

    api_match = re.search(r'^([\w\.]+)\((.*)\)$', text)
    if api_match:
        return param_fixer(api_match)
    
    # 兜底正则
    final_api = re.search(r'([\w]+\.[\w]+\.[\w]+\(.*\))', text)
    return final_api.group(1) if final_api else "error_input"

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
        for i in tqdm(range(len(lines)), desc="Building ICL-BOP Prompts"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                raw_input = case["input"]
                
                atoms = self.splitter_agent.process(raw_input)
                if not atoms: atoms = [raw_input]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    if p_res["extracted"]["room"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["perception"]["room_found"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot":
                        target_rooms.add("VacuumRobot")
                    
                    full_state = self.data_store[home_id]["state"]
                    full_methods = self.data_store[home_id]["method"]
                    
                    partial_state = {}
                    partial_methods = []
                    
                    for r in target_rooms:
                        if r in full_state: 
                            partial_state[r] = full_state[r]
                        else:
                            partial_state[r] = {} 
                        if r == "VacuumRobot" and "VacuumRobot" in full_state:
                             partial_state["VacuumRobot"] = full_state["VacuumRobot"]

                    for m in full_methods:
                        if m["room_name"] in target_rooms or m["room_name"] == "None":
                            partial_methods.append(m)

                    state_str, method_str = chang_json2str(partial_state, partial_methods)
                    
                    # 重新编排 Prompt：示例作为全局引导，当前状态作为执行依据
                    user_content = ""
                    if self.use_few_shot:
                        user_content += BOP_FEW_SHOT_EXAMPLES + "\n"
                    
                    user_content += "### CURRENT PHYSICAL CONTEXT ###\n"
                    user_content += f"<home_state>\n{state_str}</home_state>\n"
                    user_content += f"<device_method>\n{method_str}</device_method>\n"
                    user_content += "-------------------------------\n"
                    user_content += f"Now process this instruction:\n"
                    user_content += f"[Instruction] {atom_instr}\n"
                    user_content += f"[Result] "
                    
                    final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, user_content)
                    
                    self.samples.append({
                        "input_text": final_input,
                        "original_index": i,
                        "atom_index": atom_idx,
                        "ground_truth_full": case["output"],
                        "type": case.get("type", "normal")
                    })
            except: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 5. 主测试逻辑 (DDP 优化版)
# ==========================================
def model_test(model_name, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=32):
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

    if rank == 0: print(f"Loading Model from: {base_model_dir}")
    
    # 优化 Tokenizer 加载，避免 dict 报错
    config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    except:
        tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(base_model_dir, "tokenizer.json"))
        
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir, 
        dtype=torch.bfloat16, 
        device_map=None if ddp_enabled else "auto", 
        trust_remote_code=True
    )
    if ddp_enabled: model.to(device)
    if use_finetuned:
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
            generated_ids = model.generate(**inputs, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            response = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                temp_results.append({
                    "generated": clean_generated_text(text),
                    "original_index": batch[i]["original_index"],
                    "atom_index": batch[i]["atom_index"],
                    "ground_truth_full": batch[i]["ground_truth_full"],
                    "type": batch[i]["type"]
                })

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_suffix = "bop_few_shot" if use_few_shot else "bop_zero_shot"
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: 
        dist.barrier(device_ids=[device.index] if device.type == 'cuda' else None)
        
    if rank == 0:
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        grouped = defaultdict(list)
        for item in all_atoms: grouped[item["original_index"]].append(item)
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            combined_code = ",".join([a["generated"].strip() for a in atoms]) + ","
            final_results.append({"generated": combined_code, "expected": atoms[0]["ground_truth_full"], "type": atoms[0]["type"]})
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"BOP ICL Evaluation Complete. Saved to: {final_file}")

    if ddp_enabled:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    model_test(args.model_name, args.use_few_shot, args.use_finetuned, batch_size=args.batch_size)
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
# 1. 系统提示词 (融合极致严苛的执行逻辑)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a high-precision Smart Home Execution Engine. 
Your task is to transform a single atomic instruction into a Python API call based on strict physical validation.

### MANDATORY VALIDATION PROTOCOL ###
1. **Entity Check**: Does the room and device exist in the provided <home_state>?
2. **Method Check**: Does the <device_method> explicitly list the requested operation for this device?
3. **Attribute Check**: If setting a value, does the device have that attribute in <home_state>?
4. **Range Check**: Is the value within the [lowest, highest] range specified in <home_state>?

### REJECTION RULE ###
If ANY of the above checks fail, you MUST output exactly: {error_input}
Otherwise, output the call in format: {room.device.method(value)}

CRITICAL: Only use the context provided in the current <home_state>. Do NOT hallucinate devices from examples.
"""

# ==========================================
# 2. 融合推理链的 ICL Examples (针对 Home13 特征校准)
# ==========================================
BOP_FEW_SHOT_EXAMPLES = """### LOGIC EXAMPLES FOR REFERENCE ###

[Instruction] Set the intensity of the aromatherapy in the master bedroom to 100.
[Reasoning] master_bedroom.aromatherapy exists, but its state only shows 'state'. It does not have an 'intensity' attribute, and 'set_intensity' is not in its methods.
[Result] {error_input}

[Instruction] Decrease the heating temperature in the bathroom by 9 degrees.
[Reasoning] bathroom exists, but it only contains [light, trash]. The device 'heating' is missing from this room's state.
[Result] {error_input}

[Instruction] Set the AC in the guest bedroom to 25 degrees.
[Reasoning] guest_bedroom.air_conditioner exists. Current temp is 18, range is [16, 30]. 25 is within the valid range.
[Result] {guest_bedroom.air_conditioner.set_temperature(25)}

[Instruction] Set the study room heating to 20 degrees.
[Reasoning] study_room.heating exists, but its valid range is [30, 100]. 20 is below the minimum threshold.
[Result] {error_input}

[Instruction] Start the vacuum robot.
[Reasoning] VacuumRobot is a global device and its methods support 'start_cleaning'.
[Result] {VacuumRobot.start_cleaning()}
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
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"

def clean_generated_text(text):
    # 截断所有推理残留
    stop_signals = ["[Instruction]", "[Reasoning]", "[Result]", "###", "Note:", "Reason:"]
    for s in stop_signals:
        if s in text: text = text.split(s)[0]
    
    # 提取花括号
    match = re.search(r'\{(.*?)\}', text)
    if match:
        text = match.group(1).strip()
    
    text = text.strip().replace("```python", "").replace("```", "")
    if "error_input" in text.lower(): return "error_input"
    
    # 处理参数格式
    api_match = re.search(r'^([\w\.]+)\((.*)\)', text)
    if api_match:
        prefix, args = api_match.group(1), api_match.group(2)
        args = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args)
        return f"{prefix}({args})"
    return text

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
        for i in tqdm(range(len(lines)), desc="Building Integrated BOP-ICL Samples"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                # BOP1: 拆分
                atoms = self.splitter_agent.process(case["input"])
                if not atoms: atoms = [case["input"]]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    # BOP2: 感知 (对原子指令进行独立感知，解决实体漂移问题)
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    if p_res["extracted"]["room"]: target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot": target_rooms.add("VacuumRobot")
                    
                    full_state = self.data_store[home_id]["state"]
                    full_methods = self.data_store[home_id]["method"]
                    
                    # 借鉴“旧代码”的 Empty State Grounding 技巧
                    partial_state = {}
                    for r in target_rooms:
                        if r in full_state: partial_state[r] = full_state[r]
                        else: partial_state[r] = {} # 显式提供空字典，增强拒绝生成的信号

                    partial_methods = [m for m in full_methods if m["room_name"] in target_rooms or m["room_name"] == "None"]
                    state_str, method_str = chang_json2str(partial_state, partial_methods)
                    
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
            except Exception: continue

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 5. 主测试逻辑 (保留全部参数)
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

    if rank == 0: print(f"Loading Model: {model_name} (BOP-ICL Optimized Mode)")

    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try: tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except: tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map=None if ddp_enabled else "auto", trust_remote_code=True)
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
            
            generated_ids = model.generate(
                **inputs, max_new_tokens=64, do_sample=False, 
                repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id
            )
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
    mode_tag = "bop_icl_final" if use_few_shot else "bop_zero_final"
    part_file = os.path.join(output_dir, f"{model_name}_{mode_tag}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_tag}_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_atoms: grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            codes = [a["generated"].strip() for a in atoms if a["generated"]]
            combined_code = ",".join(codes)
            if combined_code and not combined_code.endswith(","): combined_code += ","
            final_results.append({"generated": combined_code, "expected": atoms[0]["ground_truth_full"], "type": atoms[0]["type"]})
            
        final_file = os.path.join(output_dir, f"{model_name}_{mode_tag}_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"BOP Final Evaluation Complete. Saved to: {final_file}")

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
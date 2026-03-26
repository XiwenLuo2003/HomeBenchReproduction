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
# 1. 系统提示词 (强化物理硬约束逻辑)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a smart home control system that strictly follows physical constraints.
Your task is to convert User Instructions into Python-style API calls.

CRITICAL PROTOCOL:
1. GROUNDING: Verify room and device exist in <home_state>. If not, output: error_input
2. CAPABILITY: Verify the operation exists for the device in <device_method>. If not, output: error_input
3. RANGE: Verify parameters are within the specific [lowest, highest] range defined in <home_state>. If not, output: error_input
4. FORMAT: Output ONLY the API call or error_input. Use the format: {room.device.operation(value)}
"""

# ==========================================
# 2. 深度重构的 ICL Examples (抽象化逻辑锚定)
# ==========================================
BOP_FEW_SHOT_EXAMPLES = """### LOGIC EXAMPLES ###

[Instruction] Turn on the lamp in the kitchen.
[State Check] 'kitchen' exists, 'light' device exists, 'turn_on' method is valid.
[Output] {kitchen.light.turn_on()}

[Instruction] Set the humidifier in the living room to 50.
[State Check] 'living_room' exists, but NO 'humidifier' device is listed in the state.
[Output] {error_input}

[Instruction] Adjust the bedroom air conditioner to 15 degrees.
[State Check] 'bedroom' 'air_conditioner' exists, but temperature range is [16, 30]. 15 is out of range.
[Output] {error_input}

[Instruction] Dim the light in the hallway.
[State Check] 'hallway' exists, 'light' exists, but it only supports 'turn_on' and 'turn_off'.
[Output] {error_input}
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
    stop_signals = ["###", "User:", "[Instruction]", "Machine", "Note:", "[State Check]"]
    for s in stop_signals:
        if s in text:
            text = text.split(s)[0]
    text = text.strip()
    if "error_input" in text.lower():
        return "error_input"
    match = re.search(r'\{(.*?)\}', text)
    if match:
        content = match.group(1).strip()
        content = re.sub(r'([a-zA-Z0-9_]+)[:=]\s*', '', content)
        return content
    api_match = re.search(r'([\w]+\.[\w]+\.[\w]+\(.*\))', text)
    if api_match:
        content = api_match.group(1).strip()
        content = re.sub(r'([a-zA-Z0-9_]+)[:=]\s*', '', content)
        return content
    return "error_input"

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
        for i in tqdm(range(len(lines)), desc="Building Robust ICL Prompts"):
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
                    prompt_body = ""
                    if self.use_few_shot:
                        prompt_body += BOP_FEW_SHOT_EXAMPLES + "\n"
                    prompt_body += "### CURRENT REAL-TIME CONTEXT ###\n"
                    prompt_body += f"<home_state>\n{state_str}</home_state>\n"
                    prompt_body += f"<device_method>\n{method_str}</device_method>\n"
                    prompt_body += "-------------------------------\n"
                    prompt_body += f"[User Instruction] {atom_instr}\n"
                    prompt_body += "[Machine Result] "
                    final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, prompt_body)
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

    if rank == 0: print(f"Loading Tokenizer from: {base_model_dir}")
    
    # 强化版 Tokenizer 加载逻辑
    try:
        # 先加载 config 确保 AutoTokenizer 识别
        config = AutoConfig.from_pretrained(base_model_dir, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(base_model_dir, config=config, trust_remote_code=True)
    except Exception as e:
        if rank == 0: print(f"AutoTokenizer failed, falling back to Qwen2TokenizerFast. Error: {e}")
        tokenizer_file = os.path.join(base_model_dir, "tokenizer.json")
        tokenizer = Qwen2TokenizerFast(tokenizer_file=tokenizer_file)
        
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    if rank == 0: print(f"Loading Model from: {base_model_dir}")
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
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=64, 
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
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
        print(f"Results saved to: {final_file}")

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
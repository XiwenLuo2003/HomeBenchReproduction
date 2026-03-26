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

# --- 导入 BOP 智能体 (需确保当前目录下有对应文件) ---
try:
    from BOP1Agent import InstructionSplitterV15 as InstructionSplitter
    from BOP2Agent import EnvironmentPerceptionAgent
except ImportError:
    print("Error: 找不到 BOP 智能体文件。请确保 BOP1Agent.py 和 BOP2Agent.py 在同一目录下。")
    exit(1)

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 强化版系统提示词 (加入感知报告逻辑)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a smart home control agent. 
You will receive:
1. <home_state>: Current relevant device data.
2. <perception_report>: Feedback from the environment sensor.
3. <User instructions>: An atomic instruction.

DECISION RULES:
1. If <perception_report> mentions "missing", "not found", or "Hallucination Risk", you MUST output: {error_input}.
2. If the user asks to set an attribute that is NOT present in <home_state> for that device, output: {error_input}.
3. If <home_state> shows "No adjustable attributes" for a device, any parameter-based operation (like set_intensity) is invalid.
4. Output exactly `{room.device.method(params)}` or `{error_input}` inside curly braces.
5. STOP generation immediately after the closing brace.
"""

# ==========================================
# 2. 引入感知报告的 ICL 示例
# ==========================================
BOP_ICL_REFINED_V2 = """Here are examples of state-based reasoning with perception feedback:

<example1>
<home_state>
master_bedroom:
  aromatherapy
    state: off
    (No adjustable attributes)
</home_state>
<perception_report>
Warning: Device 'aromatherapy' in 'master_bedroom' has no adjustable attributes.
</perception_report>
<User instructions:> Set the intensity of the aromatherapy in the master bedroom to 100.
<Machine instructions:> {error_input}
</example1>

<example2>
<home_state>
bathroom:
  light
    state: on
</home_state>
<perception_report>
Error: Device 'heating' not found in 'bathroom'. Available: ['light', 'trash'].
</perception_report>
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
<perception_report>
Success: Found air_conditioner in living_room.
</perception_report>
<User instructions:> Set the AC in the living room to 25 degrees.
<Machine instructions:> {living_room.air_conditioner.set_temperature(25)}
</example3>
"""

# ==========================================
# 3. 辅助转换函数 (优化属性显示)
# ==========================================
def chang_json2str_v2(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            device_obj = state[room]
            state_str += "  state: " + str(device_obj.get("state", "N/A")) + "\n"
            if "attributes" in device_obj and device_obj["attributes"]:
                for attr, val in device_obj["attributes"].items():
                    state_str += f"    {attr}: {val.get('value', 'N/A')}\n"
            else:
                state_str += "    (No adjustable attributes)\n"
        else:
            for device in state[room].keys():
                if device == "room_name": continue
                device_obj = state[room][device]
                state_str += "  " + device + "\n"
                state_str += "    state: " + str(device_obj.get("state", "N/A")) + "\n"
                if "attributes" in device_obj and device_obj["attributes"]:
                    for attr, val in device_obj["attributes"].items():
                        line = f"    {attr}: {val.get('value', 'N/A')}"
                        if "options" in val: line += f" (options: {val['options']})"
                        elif "lowest" in val: line += f" (range: {val['lowest']} - {val['highest']})"
                        state_str += line + "\n"
                else:
                    state_str += "    (No adjustable attributes)\n"

    method_str = ""
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        params = ",".join([f"{p['name']}:{p['type']}" for p in method["parameters"]])
        method_str += f"{room_prefix}{method['device_name']}.{method['operation']}({params}); "
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        if "gemma" in tokenizer.name_or_path.lower():
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"

def clean_generated_text(text):
    bracket_match = re.search(r'\{(.*?)\}', text)
    if bracket_match: text = bracket_match.group(1)
    text = text.strip().split('\n')[0]
    if "error_input" in text: return "error_input"
    
    match = re.match(r'^([\w\.]+)\((.*)\)', text)
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
        text = f"{prefix}({', '.join(new_args)})"
    return text

# ==========================================
# 4. BOP Dataset 类
# ==========================================
class bop_home_assistant_dataset_v2(Dataset):
    def __init__(self, tokenizer, use_few_shot=False):
        self.tokenizer = tokenizer
        self.use_few_shot = use_few_shot
        self.splitter_agent = InstructionSplitter()
        self.perception_agent = EnvironmentPerceptionAgent()
        
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f:
            self.data_store = {json.loads(line)["home_id"]: json.loads(line) for line in f}
            
        with open(os.path.join(dataset_dir, "test_data_copy.jsonl"), "r") as f:
            lines = f.readlines()
            
        self.samples = []
        print(f"正在构建 BOP-ICL-V2 数据集 (Few-shot: {use_few_shot})...")
        
        for i in tqdm(range(len(lines))):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                atoms = self.splitter_agent.process(case["input"])
                if not atoms: atoms = [case["input"]]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    report = f"Perception Status: {p_res['message']}\n"
                    if percept.issues:
                        report += "Warnings: " + "; ".join(percept.issues) + "\n"

                    target_rooms = set()
                    if p_res["perception"]["room_found"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot":
                        target_rooms.add("VacuumRobot")
                    
                    full_data = self.data_store[home_id]
                    p_state = {r: full_data["home_status"][r] for r in target_rooms if r in full_data["home_status"]}
                    p_methods = [m for m in full_data["method"] if m["room_name"] in target_rooms or m["room_name"] == "None"]

                    state_str, method_str = chang_json2str_v2(p_state, p_methods)
                    
                    user_content = f"<home_state>\n{state_str}</home_state>\n"
                    user_content += f"<device_method>\n{method_str}</device_method>\n"
                    user_content += f"<perception_report>\n{report}</perception_report>\n"
                    
                    if self.use_few_shot:
                        user_content += BOP_ICL_REFINED_V2
                    
                    user_content += f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
                    
                    final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, user_content)
                    
                    self.samples.append({
                        "input_text": final_input,
                        "original_index": i,
                        "atom_index": atom_idx,
                        "ground_truth_full": case["output"],
                        "type": case.get("type", "normal")
                    })
            except Exception:
                continue

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

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    load_device_map = None if ddp_enabled else "auto"
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map=load_device_map, trust_remote_code=True)
    if ddp_enabled: model.to(device)
    if use_finetuned:
        model = PeftModel.from_pretrained(model, adapter_dir)
        if ddp_enabled: model.to(device)
    model.eval()

    test_dataset = bop_home_assistant_dataset_v2(tokenizer, use_few_shot=use_few_shot)
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    temp_results = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, disable=(rank != 0)):
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
            generated_texts = tokenizer.batch_decode(gen_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
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
    mode_str = "bop_v2_icl" if use_few_shot else "bop_v2_zero"
    if use_finetuned: mode_str += "_sft"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_str}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        print("正在合并各进程结果...")
        all_atoms = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_str}_part_*.json")):
            with open(file_path, "r") as f: all_atoms.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_atoms: grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
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
        print(f"评估完成，最终结果已保存至: {final_file}")

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
    
    model_test(args.model_name, args.use_few_shot, args.use_finetuned, args.batch_size, args.cuda_devices)
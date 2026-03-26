import argparse
import torch
import os
import json
import re
import time
import glob
import gc
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
    print("Error: Could not import BOP Agents. Please check BOP1Agent.py and BOP2Agent.py.")
    exit(1)

# --- 路径与环境设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 系统提示词矩阵 (基于 ICL-V2 优化)
# ==========================================

# 【Agent 4 开启】严格决策逻辑
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

# 【Agent 4 消融】宽松翻译逻辑
LOOSE_SYSTEM_PROMPT_BOP = """You are a smart home assistant. 
Translate User Instructions into Python-style API calls inside curly braces {}. 
Try your best to fulfill the user's request based on the provided environment data.
"""

# ==========================================
# 2. ICL 示例库 (ICL-V2 精炼版)
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
# 3. 辅助函数
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

def apply_chat_template(tokenizer, system, user):
    if "gemma" in tokenizer.name_or_path.lower():
        return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ==========================================
# 4. Ablation Dataset 类
# ==========================================

class bop_ablation_dataset_icl(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.splitter = InstructionSplitter()
        self.perception = EnvironmentPerceptionAgent()
        
        with open(os.path.join(PROJECT_ROOT, "dataset", "home_status_method.jsonl"), "r") as f:
            self.data_store = {json.loads(l)["home_id"]: json.loads(l) for l in f}
        with open(os.path.join(PROJECT_ROOT, "dataset", "test_data_copy.jsonl"), "r") as f:
            self.lines = f.readlines()

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        case = json.loads(self.lines[idx])
        home_id = case["home_id"]
        sample_id = case.get("id", "")
        
        # 1. A1 Ablation: Instruction Splitting
        if self.args.no_splitter:
            atoms = [case["input"]]
        else:
            if "multi" in sample_id.lower():
                atoms = self.splitter.process(case["input"])
            else:
                atoms = [case["input"]]
        
        if not atoms: atoms = [case["input"]]
        
        # 2. A4 Ablation: Verifier Rules
        system_p = STRICT_SYSTEM_PROMPT_BOP if not self.args.no_verifier else LOOSE_SYSTEM_PROMPT_BOP
        
        atom_samples = []
        for atom_idx, atom_instr in enumerate(atoms):
            full_data = self.data_store[home_id]
            
            # 3. A2 Ablation: Perception Report & Context Filter
            if not self.args.no_perception:
                percept = self.perception.run(atom_instr, home_id)
                p_res = percept.result
                report = f"Perception Status: {p_res['message']}\n"
                if percept.issues:
                    report += "Warnings: " + "; ".join(percept.issues) + "\n"
                
                target_rooms = set()
                if p_res["perception"]["room_found"]: target_rooms.add(p_res["extracted"]["room"])
                if p_res["extracted"]["device"] == "vacuum_robot": target_rooms.add("VacuumRobot")
                
                p_state = {r: full_data["home_status"][r] for r in target_rooms if r in full_data["home_status"]}
                p_methods = [m for m in full_data["method"] if m["room_name"] in target_rooms or m["room_name"] == "None"]
                state_str, method_str = chang_json2str_v2(p_state, p_methods)
            else:
                # Perception 消融：全量状态，无报告
                state_str, method_str = chang_json2str_v2(full_data["home_status"], full_data["method"])
                report = "Perception Status: Service Unavailable. Full house state provided.\n"

            user_content = f"<home_state>\n{state_str}</home_state>\n"
            user_content += f"<device_method>\n{method_str}</device_method>\n"
            user_content += f"<perception_report>\n{report}</perception_report>\n"
            
            if self.args.use_few_shot:
                user_content += BOP_ICL_REFINED_V2
            
            user_content += f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
            
            atom_samples.append({
                "input_text": apply_chat_template(self.tokenizer, system_p, user_content),
                "original_index": idx,
                "atom_index": atom_idx,
                "ground_truth_full": case["output"],
                "type": case.get("type", "normal")
            })
        return atom_samples

# ==========================================
# 5. 推理与汇总程序
# ==========================================

def run_test(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    ddp = "WORLD_SIZE" in os.environ
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    sub_dirs = {"llama": "llama3-8b-Instruct", "qwen": "Qwen2.5-7B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3", "gemma": "Gemma-7B-Instruct-v0.3"}
    model_dir = os.path.join(PROJECT_ROOT, "models", sub_dirs.get(args.model_name, args.model_name))
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output", f"{args.model_name}_sft")

    if local_rank == 0: print(f"🚀 Loading ICL-Ablation Framework: {args.model_name}...")

    # Tokenizer
    tk_src = adapter_dir if args.use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else model_dir
    try: tokenizer = AutoTokenizer.from_pretrained(tk_src, trust_remote_code=True)
    except: tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tk_src, "tokenizer.json"))
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=None, trust_remote_code=True).to(device)
    if args.use_finetuned:
        model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()

    # Dataset
    ds = bop_ablation_dataset_icl(tokenizer, args)
    def collate_fn(batch): return [item for sublist in batch for item in sublist]
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=DistributedSampler(ds, shuffle=False) if ddp else None, collate_fn=collate_fn)

    results = []
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(loader, disable=(local_rank != 0))):
            inputs = tokenizer([b["input_text"] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id)
            gen_texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            for j, text in enumerate(gen_texts):
                results.append({
                    "generated": clean_generated_text(text),
                    "original_index": batch[j]["original_index"],
                    "atom_index": batch[j]["atom_index"],
                    "ground_truth_full": batch[j]["ground_truth_full"],
                    "type": batch[j]["type"]
                })
            if i % 10 == 0: torch.cuda.empty_cache()

    # 清理显存准备同步
    del model
    gc.collect()
    torch.cuda.empty_cache()

    out_dir = os.path.join(PROJECT_ROOT, "output", "ablation_icl")
    os.makedirs(out_dir, exist_ok=True)
    exp_id = f"ICL_{args.model_name}_S{int(not args.no_splitter)}P{int(not args.no_perception)}V{int(not args.no_verifier)}"
    
    with open(os.path.join(out_dir, f"{exp_id}_part_{local_rank}.json"), "w") as f: json.dump(results, f)

    if ddp: dist.barrier()
    if local_rank == 0:
        all_data = []
        for f in glob.glob(os.path.join(out_dir, f"{exp_id}_part_*.json")):
            with open(f, "r") as r: all_data.extend(json.load(r))
            os.remove(f)
        
        grouped = defaultdict(list)
        for item in all_data: grouped[item["original_index"]].append(item)
        final_res = []
        for idx in sorted(grouped.keys()):
            atoms = sorted(grouped[idx], key=lambda x: x["atom_index"])
            code = ",".join([a["generated"] for a in atoms if a["generated"]])
            if code and not code.endswith(","): code += ","
            final_res.append({"generated": code, "expected": atoms[0]["ground_truth_full"], "type": atoms[0]["type"]})
        
        final_path = os.path.join(out_dir, f"{exp_id}_result.json")
        with open(final_path, "w") as f: json.dump(final_res, f, indent=4)
        print(f"\n✅ Ablation Test Complete: {exp_id}\n📄 Result: {os.path.abspath(final_path)}\n")

    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_splitter", action="store_true")
    parser.add_argument("--no_perception", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")
    args = parser.parse_args()
    run_test(args)
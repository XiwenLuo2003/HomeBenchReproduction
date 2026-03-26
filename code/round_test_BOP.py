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
    print("Error: Could not find BOP agent files.")
    exit(1)

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 系统提示词与 ICL 示例 (沿用 V2 最优版)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a smart home control agent. 
You will receive:
1. <home_state>: Current relevant device data.
2. <perception_report>: Feedback from the environment sensor.
3. <User instructions>: An atomic instruction.

DECISION RULES:
1. If <perception_report> mentions "missing", "not found", or "Hallucination Risk", you MUST output: {error_input}.
2. If the user asks to set an attribute that is NOT present in <home_state> for that device, output: {error_input}.
3. If <home_state> shows "No adjustable attributes" for a device, any parameter-based operation is invalid.
4. Output exactly `{room.device.method(params)}` or `{error_input}` inside curly braces.
5. STOP generation immediately after the closing brace.
"""

BOP_ICL_REFINED_V2 = """Here are examples of strict state-based reasoning with perception feedback:

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
# 2. 辅助函数
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
# 3. BOP Round Dataset
# ==========================================
class bop_round_dataset_v2(Dataset):
    def __init__(self, tokenizer, home_id_file, use_few_shot=False):
        self.tokenizer = tokenizer
        self.use_few_shot = use_few_shot
        self.splitter_agent = InstructionSplitter()
        self.perception_agent = EnvironmentPerceptionAgent()
        
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f:
            self.home_data_store = {json.loads(line)["home_id"]: json.loads(line) for line in f}
            
        filepath = os.path.join(dataset_dir, home_id_file)
        with open(filepath, "r", encoding="utf-8") as f:
            self.rounds_data = json.load(f)

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        for i, round_case in enumerate(self.rounds_data):
            home_id = round_case["home_id"]
            if home_id not in self.home_data_store: continue
            
            # BOP1: 拆分
            atoms = self.splitter_agent.process(round_case["input"])
            if not atoms: atoms = [round_case["input"]]
            
            atom_prompts = []
            for atom_idx, atom_instr in enumerate(atoms):
                percept = self.perception_agent.run(atom_instr, home_id)
                p_res = percept.result
                report = f"Perception Status: {p_res['message']}\n"
                if percept.issues: report += "Warnings: " + "; ".join(percept.issues) + "\n"

                target_rooms = set()
                if p_res["perception"]["room_found"]: target_rooms.add(p_res["extracted"]["room"])
                if p_res["extracted"]["device"] == "vacuum_robot": target_rooms.add("VacuumRobot")
                
                full_data = self.home_data_store[home_id]
                p_state = {r: full_data["home_status"][r] for r in target_rooms if r in full_data["home_status"]}
                p_methods = [m for m in full_data["method"] if m["room_name"] in target_rooms or m["room_name"] == "None"]

                state_str, method_str = chang_json2str_v2(p_state, p_methods)
                
                user_content = f"<home_state>\n{state_str}</home_state>\n<device_method>\n{method_str}</device_method>\n<perception_report>\n{report}</perception_report>\n"
                if self.use_few_shot: user_content += BOP_ICL_REFINED_V2
                user_content += f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
                
                final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, user_content)
                atom_prompts.append(final_input)

            self.samples.append({
                "original_index": i,
                "id": round_case["id"],
                "atom_prompts": atom_prompts,
                "expected": round_case["output"],
                "type": round_case.get("type", "normal")
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# ==========================================
# 4. 主测试逻辑 (DDP 适配)
# ==========================================
def round_test_ddp(model_name, home_id_file, use_few_shot=False, use_finetuned=False, batch_size=1):
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
    base_model_dir = os.path.join(PROJECT_ROOT, "models", sub_dirs.get(model_name, model_name))
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_sft")

    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # DDP 模式下显存管理：不使用 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    if use_finetuned: model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()

    dataset = bop_round_dataset_v2(tokenizer, home_id_file, use_few_shot)
    sampler = DistributedSampler(dataset, shuffle=False) if ddp_enabled else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x)

    temp_results = []
    with torch.inference_mode():
        for batch in tqdm(loader, disable=(rank != 0)):
            for case in batch:
                round_codes = []
                # 由于 round 内可能有多个 atom，我们这里进行 mini-batch 推理或顺序推理
                # 为保证代码健壮性，这里对单 round 内的 atoms 进行处理
                inputs = tokenizer(case["atom_prompts"], return_tensors="pt", padding=True).to(device)
                gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.1)
                responses = tokenizer.batch_decode(gen_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                
                for res_text in responses:
                    round_codes.append(clean_generated_text(res_text).replace("{", "").replace("}", ""))
                
                combined = ",".join([c for c in round_codes if c])
                if combined and not combined.endswith(","): combined += ","
                
                temp_results.append({
                    "original_index": case["original_index"],
                    "id": case["id"],
                    "generated": combined,
                    "expected": case["expected"],
                    "type": case["type"]
                })

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "bop_round_v2"
    part_file = os.path.join(output_dir, f"{model_name}_{mode_str}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        all_data = []
        for p in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_str}_part_*.json")):
            with open(p, "r") as f: all_data.extend(json.load(f))
            os.remove(p)
        
        all_data.sort(key=lambda x: x["original_index"])
        home_tag = os.path.basename(home_id_file).replace("all_rounds_of_", "").replace(".json", "")
        final_file = os.path.join(output_dir, f"{model_name}_{mode_str}_ICL_{home_tag}.json")
        
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"DDP Round Test Complete. Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--home_id_file", type=str, required=True)
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    round_test_ddp(args.model_name, args.home_id_file, args.use_few_shot, args.use_finetuned, args.batch_size)
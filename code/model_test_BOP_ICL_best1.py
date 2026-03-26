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

# --- 导入 BOP 智能体 (请确保 BOP1Agent.py 和 BOP2Agent.py 在同一目录下) ---
try:
    from BOP1Agent import InstructionSplitterV15 as InstructionSplitter
    from BOP2Agent import EnvironmentPerceptionAgent
except ImportError:
    print("Error: 找不到 BOP 智能体文件。请检查 BOP1Agent.py 和 BOP2Agent.py 是否在当前目录。")
    exit(1)

# --- 全局路径与环境设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 系统提示词 (针对 ICL 优化的宪法级约束)
# ==========================================
STRICT_SYSTEM_PROMPT_BOP = """You are a specialized Smart Home Control Agent. 
Your ONLY task is to convert a single User Instruction into EXACTLY ONE executable Python API call based on the provided Home State.

CRITICAL MATHEMATICAL LAWS:
1. **Simple Arithmetic ONLY**: For all 0-100 attributes (brightness, intensity, degree), "percent" refers to a DIRECT ADDITION or SUBTRACTION of the number.
   - FORMULA: TargetValue = CurrentValue + ChangeNumber (or - ChangeNumber).
   - EXAMPLE: If current is 80, "decrease by 20 percent" means 80 - 20 = 60. 
   - NEVER MULTIPLY. 10% change always means a 10-unit change.

GROUNDING RULES:
1. Use the provided <home_state> to verify device existence and current values.
2. Output 'error_input' ONLY if the room/device is explicitly missing or the action is physically impossible.
3. If the instruction is valid but the device is 'off', still output the requested 'set_' or 'turn_' command.
4. Output ONLY the code string. NO explanations. NO markdown.
"""

# ==========================================
# 2. ICL 示例 (V5：模拟局部环境推理)
# ==========================================
BOP_FEW_SHOT_EXAMPLES = """Below are pedagogical examples. Notice how we use the partial state to calculate values:

<example1>
<Context Snippet:>
living_room:
  light:
    state: on
    attributes:
      brightness: 50
<User instructions:> Turn off the light in the living room.
<Machine instructions:> living_room.light.turn_off()
</example1>

<example2>
<Context Snippet:>
living_room:
  curtain:
    state: open
    attributes:
      degree: 10
<User instructions:> Increase the curtain degree by 20 percent in the living room.
<Calculation:> Current(10) + 20 = 30.
<Machine instructions:> living_room.curtain.set_degree(30)
</example2>

<example3>
<Context Snippet:>
balcony:
  light:
    state: on
    attributes:
      brightness: 83
<User instructions:> Decrease the brightness of the light in the balcony by 43 percent.
<Calculation:> Current(83) - 43 = 40.
<Machine instructions:> balcony.light.set_brightness(40)
</example3>

<example4>
<Context Snippet:>
study_room:
  light:
    state: off
<User instructions:> Set the heating temperature to 29 in the study room.
<Logic:> State shows 'light' but NO 'heating' device.
<Machine instructions:> error_input
</example4>

<example5>
<Context Snippet:>
master_bedroom:
  light:
    state: on
    attributes:
      brightness: 65
<User instructions:> Decrease the light brightness by 45 percent in the master bedroom.
<Calculation:> Current(65) - 45 = 20.
<Machine instructions:> master_bedroom.light.set_brightness(20)
</example5>
"""

# ==========================================
# 3. 辅助函数 (V4: 激光手术级清洗)
# ==========================================

def chang_json2str(state, methods):
    """将环境 JSON 状态和方法库转换为字符串"""
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if isinstance(state[room], dict):
                state_str += "  state: " + str(state[room].get("state", "N/A")) + "\n"
                if "attributes" in state[room]:
                    for attr in state[room]["attributes"].keys():
                        val = state[room]["attributes"][attr]
                        state_str += f"  {attr}: {val.get('value', 'N/A')}"
                        if "options" in val: state_str += f" (options{val['options']})"
                        elif "lowest" in val: state_str += f" (range: {val['lowest']} - {val['highest']})"
                        state_str += "\n"
        else:
            for dev in state[room].keys():
                if dev == "room_name": continue
                dev_obj = state[room][dev]
                state_str += f"  {dev}\n    state: {dev_obj.get('state', 'N/A')}\n"
                if "attributes" in dev_obj:
                    for attr in dev_obj["attributes"].keys():
                        val = dev_obj["attributes"][attr]
                        state_str += f"    {attr}: {val.get('value', 'N/A')}"
                        if "options" in val: state_str += f" (options{val['options']})"
                        elif "lowest" in val: state_str += f" (range: {val['lowest']} - {val['highest']})"
                        state_str += "\n"

    method_str = ""
    for m in methods:
        prefix = f"{m['room_name']}." if m["room_name"] != "None" else ""
        params = ",".join([f"{p['name']}:{p['type']}" for p in m["parameters"]])
        method_str += f"{prefix}{m['device_name']}.{m['operation']}({params});"
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    """适配模型对话模板"""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        return f"{system}\n\n{user}"

def clean_generated_text(text):
    """
    V4 强化清洗逻辑：解决 ICL 带来的重复、幻觉和多余解释。
    """
    # 1. 基础截断
    stop_signals = ["<User", "User:", "Machine", "<Logic", "<Calculation", "Example", "STOP", "Note:", "Ref:"]
    for s in stop_signals:
        if s in text:
            text = text.split(s)[0]
    
    text = text.strip().replace("```python", "").replace("```", "")
    
    # 2. 优先检索 error_input
    if re.search(r'\berror_input\b', text.lower()):
        return "error_input"

    # 3. 激光提取：只抓取第一个符合 room.device.op(...) 格式的代码
    code_match = re.search(r'([\w]+\.[\w]+\.[\w]+\([^\)]*\))', text)
    
    if not code_match:
        return "error_input"
    
    valid_code = code_match.group(1)

    # 4. 参数精修
    try:
        prefix = valid_code.split('(')[0]
        args_part = valid_code.split('(')[1].rstrip(')')
        
        args_part = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_part)
        
        new_args = []
        for arg in args_part.split(','):
            arg = arg.strip()
            if not arg: continue
            if re.match(r'^[a-zA-Z_]+$', arg) and arg.lower() not in ['true', 'false', 'none']:
                arg = f"'{arg}'"
            new_args.append(arg)
        
        return f"{prefix}({', '.join(new_args)})"
    except Exception:
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
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f:
            self.data_store = {}
            for line in f:
                d = json.loads(line)
                self.data_store[d["home_id"]] = d
            
        with open(os.path.join(dataset_dir, "test_data_copy.jsonl"), "r") as f:
            lines = f.readlines()
            
        self.samples = []
        print(f"BOP ICL 系统：正在处理 {len(lines)} 条原始测试数据...")
        
        for i in tqdm(range(len(lines)), desc="构建推理 Prompt"):
            try:
                case = json.loads(lines[i])
                home_id = case["home_id"]
                if home_id not in self.data_store: continue
                
                atoms = self.splitter_agent.process(case["input"])
                if not atoms: atoms = [case["input"]]
                
                for atom_idx, atom_instr in enumerate(atoms):
                    percept = self.perception_agent.run(atom_instr, home_id)
                    p_res = percept.result
                    
                    target_rooms = set()
                    if p_res["perception"]["room_found"]:
                        target_rooms.add(p_res["extracted"]["room"])
                    if p_res["extracted"]["device"] == "vacuum_robot":
                        target_rooms.add("VacuumRobot")
                        
                    full_state = self.data_store[home_id]["home_status"]
                    full_methods = self.data_store[home_id]["method"]
                    
                    partial_state = {r: full_state[r] for r in target_rooms if r in full_state}
                    if "VacuumRobot" in target_rooms:
                        partial_state["VacuumRobot"] = full_state.get("VacuumRobot", {})

                    partial_methods = [m for m in full_methods if m["room_name"] in target_rooms or m["room_name"] == "None"]

                    state_str, method_str = chang_json2str(partial_state, partial_methods)
                    
                    prompt_body = f"<home_state>\n{state_str}</home_state>\n<device_method>\n{method_str}</device_method>\n"
                    if self.use_few_shot:
                        prompt_body += BOP_FEW_SHOT_EXAMPLES
                    
                    query = f"-------------------------------\n<User instructions:>\n{atom_instr}\n<Machine instructions:>"
                    
                    final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_BOP, prompt_body + query)
                    
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
# 5. 测试主逻辑 (支持 DDP)
# ==========================================
def model_test(model_name, use_few_shot=False, use_finetuned=False, batch_size=32):
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

    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception:
        tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4, collate_fn=lambda b: b)

    temp_results = []
    with torch.inference_mode():
        for batch in tqdm(test_loader, disable=(rank != 0)):
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
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
    suffix = "bop_base_few_shot" if use_few_shot else "bop_base_zero_shot"
    part_file = os.path.join(output_dir, f"{model_name}_{suffix}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()
    if rank == 0:
        all_data = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{suffix}_part_*.json")):
            with open(file_path, "r") as f: all_data.extend(json.load(f))
            os.remove(file_path)
        
        grouped = defaultdict(list)
        for item in all_data: grouped[item["original_index"]].append(item)
        
        final_results = []
        for idx in sorted(grouped.keys()):
            items = sorted(grouped[idx], key=lambda x: x["atom_index"])
            combined = ",".join([it["generated"].strip() for it in items]) + ","
            final_results.append({"generated": combined, "expected": items[0]["ground_truth_full"], "type": items[0]["type"]})
            
        final_file = os.path.join(output_dir, f"{model_name}_{suffix}_test_result.json")
        with open(final_file, "w") as f: json.dump(final_results, f, indent=4)
        print(f"\n[Done] BOP 实验完成。结果已保存至: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    model_test(args.model_name, use_few_shot=args.use_few_shot, use_finetuned=args.use_finetuned, batch_size=args.batch_size)
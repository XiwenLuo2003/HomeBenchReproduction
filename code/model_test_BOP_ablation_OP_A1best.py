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

# --- 导入 BOP 核心智能体工具 ---
try:
    from BOP1Agent import InstructionSplitterV15 as InstructionSplitter
    from BOP2Agent import EnvironmentPerceptionAgent
except ImportError:
    print("Error: Could not import BOP Agents. Please ensure BOP1Agent.py and BOP2Agent.py are in the same directory.")
    exit(1)

# --- 环境设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ==========================================
# 1. 解耦的 System Prompts (对应消融状态)
# ==========================================

# [完整规则] 对应完整 BOP (Agent 4 开启)
PROMPT_STRICT = """You are a smart home control agent. 
RULES:
1. Grounding Verification: If device/room not found, method unsupported, or parameter out of range, output: error_input
2. Output ONLY the code. No explanations. STOP generating immediately after code.
"""

# [无校验规则] 对应 Agent 4 消融 (no_verifier=True)
PROMPT_LOOSE = """You are a smart home assistant. 
Convert the User Instructions into Python-style API calls. You can use math expressions if needed.
"""

# [原子指令约束] 对应 Agent 1 开启
DESC_ATOMIC = "Your task is to convert a single atomic User Instruction into exactly ONE executable API call."

# [批量指令允许] 对应 Agent 1 消融 (no_splitter=True)
DESC_BULK = "The user may provide multiple instructions. Provide all corresponding API calls."

# ==========================================
# 2. 强化版提取与清洗函数
# ==========================================

def clean_generated_text(text, is_ablation_a4=False):
    """
    针对消融实验结果的特殊清洗逻辑：
    1. 提取所有 API 或 error_input。
    2. 解决 A4 消融后的循环冗余 (Deduplication)。
    3. 处理数学表达式 (Math Stripping)。
    """
    # 截断
    stop_signals = ["<User", "User:", "Machine", "Example", "Note:", "The current", "To "]
    for s in stop_signals:
        if s in text: text = text.split(s)[0]

    # 正则提取：匹配 API 格式 或 error_input
    pattern = r'([\w]+\.[\w]+\.[\w]+\(.*\)|error_input)'
    raw_matches = re.findall(pattern, text)
    
    if not raw_matches: return "error_input"
    
    cleaned_results = []
    seen = set() # 用于彻底消融 A4 时的严重重复
    last_item = None
    
    for match in raw_matches:
        # 内部清洗：修复参数名和处理简单的数学运算（消融 A4 时常见）
        if match != "error_input":
            prefix = match.split('(')[0]
            args_search = re.search(r'\((.*)\)', match)
            if args_search:
                args_str = args_search.group(1)
                # 移除参数引导符 (e.g., temp=20 -> 20)
                args_str = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_str)
                # 针对 A4 消融产生的数学公式进行简单截断（只保留第一个数字部分）
                # 比如 set(52+52*0.08) -> set(56) 这种逻辑在 eval.py 才能过
                if is_ablation_a4:
                    # 尝试寻找数字并替换
                    math_match = re.search(r'(\d+\.?\d*)', args_str)
                    if math_match: args_str = math_match.group(1)

                # 修复引号
                parts = [p.strip() for p in args_str.split(',') if p.strip()]
                final_parts = []
                for p in parts:
                    if re.match(r'^[a-zA-Z_]+$', p) and p.lower() not in ['true', 'false', 'none']:
                        p = f"'{p}'"
                    final_parts.append(p)
                match = f"{prefix}({', '.join(final_parts)})"

        # 去重逻辑：如果消融了 A4，模型容易复读，我们保留第一次出现的条目
        # 如果没消融 A4，我们只过滤连续重复
        if is_ablation_a4:
            if match not in seen:
                cleaned_results.append(match)
                seen.add(match)
        else:
            if match != last_item:
                cleaned_results.append(match)
                last_item = match

    # 长度硬截断：防止 A4 消融产生的幻觉无限延伸
    return ",".join(cleaned_results[:12])

# ==========================================
# 3. 消融实验 Dataset
# ==========================================

class bop_ablation_dataset(Dataset):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args
        self.splitter = InstructionSplitter()
        self.perception = EnvironmentPerceptionAgent()
        
        with open(os.path.join(PROJECT_ROOT, "dataset", "home_status_method.jsonl"), "r") as f:
            self.data_store = {json.loads(l)["home_id"]: {"state": json.loads(l)["home_status"], "method": json.loads(l)["method"]} for l in f}
        with open(os.path.join(PROJECT_ROOT, "dataset", "test_data_copy.jsonl"), "r") as f:
            self.lines = f.readlines()

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        case = json.loads(self.lines[idx])
        home_id = case["home_id"]
        raw_input = case["input"]
        
        # 1. A1 Ablation Logic
        if not self.args.no_splitter:
            atoms = self.splitter.process(raw_input)
            desc = DESC_ATOMIC
        else:
            atoms = [raw_input]
            desc = DESC_BULK
        
        if not atoms: atoms = [raw_input]
        
        # 2. System Prompt Selection
        rules = PROMPT_STRICT if not self.args.no_verifier else PROMPT_LOOSE
        system_p = f"{rules}\n{desc}"
        
        atom_samples = []
        for atom_idx, atom_instr in enumerate(atoms):
            full_st = self.data_store[home_id]["state"]
            full_mt = self.data_store[home_id]["method"]
            
            # 3. A2 Ablation Logic (Perception)
            if not self.args.no_perception:
                p_res = self.perception.run(atom_instr, home_id).result
                rooms = {p_res["extracted"]["room"]} if p_res["perception"]["room_found"] else set()
                if p_res["extracted"]["device"] == "vacuum_robot": rooms.add("VacuumRobot")
                
                st = {r: full_st[r] for r in rooms if r in full_st}
                mt = [m for m in full_mt if m["room_name"] in rooms or m["room_name"] == "None"]
                state_str, method_str = chang_json2str(st, mt)
            else:
                state_str, method_str = chang_json2str(full_st, full_mt)

            user_content = f"<home_state>\n{state_str}</home_state>\n<device_method>\n{method_str}</device_method>\n"
            user_content += f"-------------------------------\n<User instructions:> \n{atom_instr}\n<Machine instructions:>"
            
            atom_samples.append({
                "input_text": apply_chat_template(self.tokenizer, system_p, user_content),
                "original_index": idx,
                "atom_index": atom_idx,
                "ground_truth_full": case["output"],
                "type": case.get("type", "normal")
            })
        return atom_samples

def chang_json2str(state, methods):
    # 此处保持与 model_test_BOP_OP_best.py 一致的转换逻辑
    state_str = ""
    for room, data in state.items():
        state_str += f"{room}:\n"
        if room == "VacuumRobot":
            state_str += f"  state: {data.get('state','N/A')}\n"
            for k, v in data.get("attributes", {}).items():
                state_str += f"  {k}: {v.get('value','N/A')}\n"
        else:
            for dev, obj in data.items():
                if dev == "room_name": continue
                state_str += f"  {dev}\n    state: {obj.get('state','N/A')}\n"
                for k, v in obj.get("attributes", {}).items():
                    state_str += f"    {k}: {v.get('value','N/A')}\n"
    method_str = "".join([f"{m['room_name']}.{m['device_name']}.{m['operation']}({','.join([p['name']+':'+p['type'] for p in m['parameters']])});\n" for m in methods])
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    if "gemma" in tokenizer.name_or_path.lower():
        return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ==========================================
# 4. 主运行逻辑
# ==========================================

def run_test(args):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    ddp = local_rank != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{max(0, local_rank)}")

    sub_dirs = {"llama": "llama3-8b-Instruct", "qwen": "Qwen2.5-7B-Instruct", "mistral": "Mistral-7B-Instruct-v0.3", "gemma": "Gemma-7B-Instruct-v0.3"}
    model_dir = os.path.join(PROJECT_ROOT, "models", sub_dirs.get(args.model_name, args.model_name))
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16, device_map=None if ddp else "auto", trust_remote_code=True)
    if ddp: model.to(device)
    model.eval()

    ds = bop_ablation_dataset(tokenizer, args)
    # 此处 DataLoader 需要处理嵌套列表
    def collate_fn(batch): return [item for sublist in batch for item in sublist]
    loader = DataLoader(ds, batch_size=args.batch_size // 4 if args.no_splitter else args.batch_size, sampler=DistributedSampler(ds, shuffle=False) if ddp else None, collate_fn=collate_fn)

    results = []
    with torch.inference_mode():
        for batch in tqdm(loader, disable=(ddp and local_rank != 0)):
            inputs = tokenizer([b["input_text"] for b in batch], return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False, repetition_penalty=1.1, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            gen_texts = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            for i, text in enumerate(gen_texts):
                results.append({
                    "generated": clean_generated_text(text, is_ablation_a4=args.no_verifier),
                    "original_index": batch[i]["original_index"],
                    "atom_index": batch[i]["atom_index"],
                    "ground_truth_full": batch[i]["ground_truth_full"],
                    "type": batch[i]["type"]
                })

    out_dir = os.path.join(PROJECT_ROOT, "output", "ablation")
    os.makedirs(out_dir, exist_ok=True)
    exp_id = f"{args.model_name}_S{int(not args.no_splitter)}P{int(not args.no_perception)}V{int(not args.no_verifier)}"
    with open(os.path.join(out_dir, f"{exp_id}_part_{max(0, local_rank)}.json"), "w") as f: json.dump(results, f)

    if ddp: dist.barrier()
    if local_rank in [-1, 0]:
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
        
        with open(os.path.join(out_dir, f"{exp_id}_result.json"), "w") as f: json.dump(final_res, f, indent=4)
        print(f"Ablation {exp_id} Finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_splitter", action="store_true")
    parser.add_argument("--no_perception", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")
    args = parser.parse_args()
    run_test(args)
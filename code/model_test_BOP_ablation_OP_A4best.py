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
# 1. 更加科学的解耦 System Prompts
# ==========================================

# 【Agent 4 开启】强约束验证模式
PROMPT_STRICT_VERIFIER = """You are a smart home control agent. 
STRICT RULES:
1. Grounding Verification: You MUST check the provided Home State. If a device/room is missing, an attribute is unsupported, or a value is out of range, you MUST output: error_input
2. Conciseness: Output ONLY the code or 'error_input'. No explanations. STOP immediately.
"""

# 【Agent 4 消融】普通助手模式 (去掉了硬性的 Grounding 要求)
PROMPT_ABLATED_VERIFIER = """You are a smart home assistant. 
Your task is to convert User Instructions into Python-style API calls based on the provided environment information. 
Try your best to fulfill the user's request.
"""

# 任务描述部分
DESC_ATOMIC = "This is a single atomic instruction. Output ONE API call."
DESC_BULK = "This may contain multiple instructions. Output all relevant API calls."

# ==========================================
# 2. 清洗函数 (保持鲁棒性)
# ==========================================

def clean_generated_text(text, is_ablation_a4=False):
    stop_signals = ["<User", "User:", "Machine", "Example", "Note:", "The current", "To ", "Calculation:"]
    for s in stop_signals:
        if s in text: text = text.split(s)[0]

    # 提取所有 API 调用或 error_input
    pattern = r'([\w]+\.[\w]+\.[\w]+\(.*\)|error_input)'
    raw_matches = re.findall(pattern, text)
    
    if not raw_matches: return "error_input"
    
    cleaned_results = []
    seen = set()
    last_item = None
    
    for match in raw_matches:
        # 基础格式清洗 (处理引号和参数名)
        if match != "error_input":
            prefix = match.split('(')[0]
            args_search = re.search(r'\((.*)\)', match)
            if args_search:
                args_str = args_search.group(1)
                args_str = re.sub(r'[a-zA-Z0-9_]+[:=]\s*', '', args_str)
                # A4 消融时，如果出现数学表达式，尝试保留最终数值
                if is_ablation_a4:
                    # 简单匹配计算式中的最后一个数字
                    nums = re.findall(r'(\d+\.?\d*)', args_str)
                    if nums: args_str = nums[-1]

                parts = [p.strip() for p in args_str.split(',') if p.strip()]
                final_parts = []
                for p in parts:
                    if re.match(r'^[a-zA-Z_]+$', p) and p.lower() not in ['true', 'false', 'none']:
                        p = f"'{p}'"
                    final_parts.append(p)
                match = f"{prefix}({', '.join(final_parts)})"

        # 去重：处理消融后的重复输出
        if is_ablation_a4:
            if match not in seen:
                cleaned_results.append(match)
                seen.add(match)
        else:
            if match != last_item:
                cleaned_results.append(match)
                last_item = match

    return ",".join(cleaned_results[:12]) # 限制长度防止幻觉溢出

# ==========================================
# 3. 辅助转换函数 (保持一致)
# ==========================================

def chang_json2str(state, methods):
    state_str = ""
    for room, data in state.items():
        state_str += f"{room}:\n"
        if room == "VacuumRobot":
            state_str += f"  state: {data.get('state','N/A')}\n"
            if "attributes" in data:
                for k, v in data["attributes"].items():
                    state_str += f"  {k}: {v.get('value','N/A')}\n"
        else:
            for dev, obj in data.items():
                if dev == "room_name": continue
                state_str += f"  {dev}\n    state: {obj.get('state','N/A')}\n"
                if "attributes" in obj:
                    for k, v in obj["attributes"].items():
                        state_str += f"    {k}: {v.get('value','N/A')}\n"
    method_str = "".join([f"{m['room_name']}.{m['device_name']}.{m['operation']}({','.join([p['name']+':'+p['type'] for p in m['parameters']])});\n" for m in methods])
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    if "gemma" in tokenizer.name_or_path.lower():
        return f"<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ==========================================
# 4. Dataset 与主逻辑
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
        
        # A1 消融逻辑
        if not self.args.no_splitter:
            atoms = self.splitter.process(case["input"])
            desc = DESC_ATOMIC
        else:
            atoms = [case["input"]]
            desc = DESC_BULK
        
        # A4 消融逻辑 (切换规则)
        rules = PROMPT_STRICT_VERIFIER if not self.args.no_verifier else PROMPT_ABLATED_VERIFIER
        system_p = f"{rules}\n{desc}"
        
        atom_samples = []
        for atom_idx, atom_instr in enumerate(atoms or [case["input"]]):
            full_st = self.data_store[home_id]["state"]
            full_mt = self.data_store[home_id]["method"]
            
            # A2 消融逻辑
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
    def collate_fn(batch): return [item for sublist in batch for item in sublist]
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=DistributedSampler(ds, shuffle=False) if ddp else None, collate_fn=collate_fn)

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
        
        save_path = os.path.join(out_dir, f"{exp_id}_result.json")
        with open(save_path, "w") as f: json.dump(final_res, f, indent=4)
        
        print("\n" + "="*60)
        print(f"✅ 实验完成! 标识: {exp_id}")
        print(f"📄 结果文件: {os.path.abspath(save_path)}")
        print("="*60 + "\n")

    if ddp: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_splitter", action="store_true")
    parser.add_argument("--no_perception", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")
    args = parser.parse_args()
    run_test(args)
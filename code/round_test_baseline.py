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

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 辅助函数：将 JSON 状态转换为字符串 (Baseline 完整版) ---
def chang_json2str(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if isinstance(state[room], dict):
                state_str += "  state: " + str(state[room].get("state", "N/A")) + "\n"
                if "attributes" in state[room]:
                    for attribute, val in state[room]["attributes"].items():
                        state_str += f"  {attribute}: {val.get('value', 'N/A')}"
                        if "options" in val: state_str += f" (options{val['options']})\n"
                        elif "lowest" in val: state_str += f" (range: {val.get('lowest')} - {val.get('highest')})\n"
                        else: state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name": continue
                device_obj = state[room][device]
                state_str += "  " + device + "\n"
                state_str += "    state: " + str(device_obj.get("state", "N/A")) + "\n"
                if "attributes" in device_obj:
                    for attribute, val in device_obj["attributes"].items():
                        state_str += f"    {attribute}: {val.get('value', 'N/A')}"
                        if "options" in val: state_str += f" (options{val['options']})\n"
                        elif "lowest" in val: state_str += f" (range: {val.get('lowest')} - {val.get('highest')})\n"
                        else: state_str += "\n"
    method_str = ""
    for method in methods:
        room_prefix = method["room_name"] + "." if method["room_name"] != "None" else ""
        params = ",".join([f"{p['name']}:{p['type']}" for p in method["parameters"]])
        method_str += f"{room_prefix}{method['device_name']}.{method['operation']}({params});"
    return state_str, method_str

def apply_chat_template(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        if "gemma" in tokenizer.name_or_path.lower():
            messages = [{"role": "user", "content": f"{system}\n\n{user}"}]
        else:
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return f"{system}\n\n{user}"

# --- 适配 all_rounds 的 Baseline Dataset ---
class baseline_round_dataset(Dataset):
    def __init__(self, tokenizer, home_id_file, use_few_shot=False):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        # 1. 加载家庭背景
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f:
            self.home_map = {json.loads(line)["home_id"]: json.loads(line) for line in f}
            
        # 2. 加载多轮测试文件
        filepath = os.path.join(dataset_dir, home_id_file)
        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        # 3. 加载 Prompts
        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            self.system_prompt = f.read()
            
        self.few_shot_prompt = ""
        if use_few_shot:
            ex_path = os.path.join(code_dir, "example1.txt")
            if os.path.exists(ex_path):
                with open(ex_path, "r") as f: self.few_shot_prompt = f.read()

        self.samples = []
        self._build_samples()

    def _build_samples(self):
        for i, case in enumerate(self.raw_data):
            home_id = case["home_id"]
            if home_id not in self.home_map: continue
            
            info = self.home_map[home_id]
            state_str, method_str = chang_json2str(info["home_status"], info["method"])
            
            home_status_case = f"<home_state>\n{state_str}</home_state>\n"
            device_method_case = f"<device_method>\n{method_str}</device_method>\n"
            user_instr = f"-------------------------------\n<User instructions:> \n{case['input']}\n<Machine instructions:>"
            
            user_content = home_status_case + device_method_case + self.few_shot_prompt + user_instr
            final_input = apply_chat_template(self.tokenizer, self.system_prompt, user_content)
            
            self.samples.append({
                "original_index": i,
                "input_text": final_input,
                "expected": case["output"],
                "type": case.get("type", "normal"),
                "id": case["id"]
            })

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]

# --- 核心分布式测试函数 ---
def round_test_baseline_ddp(model_name, home_id_file, use_few_shot=False, use_finetuned=False, batch_size=1):
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

    # 关键点：每张卡加载一个独立的副本，不使用 device_map="auto"
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    if use_finetuned: model = PeftModel.from_pretrained(model, adapter_dir).to(device)
    model.eval()

    dataset = baseline_round_dataset(tokenizer, home_id_file, use_few_shot)
    sampler = DistributedSampler(dataset, shuffle=False) if ddp_enabled else None
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=lambda x: x)

    temp_results = []
    with torch.inference_mode():
        for batch in tqdm(loader, disable=(rank != 0)):
            input_texts = [b["input_text"] for b in batch]
            inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            gen_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, repetition_penalty=1.1)
            generated_texts = tokenizer.batch_decode(gen_ids[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            for i, text in enumerate(generated_texts):
                temp_results.append({
                    "original_index": batch[i]["original_index"],
                    "generated": text.strip(),
                    "expected": batch[i]["expected"],
                    "type": batch[i]["type"],
                    "id": batch[i]["id"]
                })

    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_str = "baseline_round_icl" if use_few_shot else "baseline_round_zero"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_str}_part_{rank}.json")
    with open(part_file, "w") as f: json.dump(temp_results, f)
    
    if ddp_enabled: dist.barrier()

    if rank == 0:
        print("Merging results...")
        all_data = []
        for p in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_str}_part_*.json")):
            with open(p, "r") as f: all_data.extend(json.load(f))
            os.remove(p)
        
        all_data.sort(key=lambda x: x["original_index"])
        home_tag = os.path.basename(home_id_file).replace("all_rounds_of_", "").replace(".json", "")
        final_file = os.path.join(output_dir, f"{model_name}_{mode_str}_{home_tag}_result.json")
        
        with open(final_file, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)
        print(f"Baseline Round Test Complete. Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--home_id_file", type=str, required=True)
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    round_test_baseline_ddp(args.model_name, args.home_id_file, args.use_few_shot, args.use_finetuned, args.batch_size)
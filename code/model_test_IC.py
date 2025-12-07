import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Tokenizers 并行，防止 DataLoader 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 辅助函数：IC任务核心转换逻辑 ---
def transform_ic_input(case):
    """
    1. 筛选: 检查 id 是否包含 "one"
    2. 提取: 从 output 中获取房间名
    3. 改造: 将 input 中的房间名替换为 "here"
    4. 生成: 返回 location 提示词
    """
    # 1. 筛选单指令数据
    if "one" not in case.get("id", ""):
        return None

    # 2. 从标准答案中提取真实房间名 (Ground Truth Room)
    # output 格式通常为 ["room.device.method(...)"] 或 "room.device..."
    output_raw = case.get("output", [])
    if isinstance(output_raw, list) and len(output_raw) > 0:
        cmd = output_raw[0]
    elif isinstance(output_raw, str):
        cmd = output_raw
    else:
        return None
    
    # 提取 "living_room" from "living_room.light.turn_on()"
    if "." not in cmd:
        return None
    
    room_key = cmd.split(".")[0]
    room_name = room_key.replace("_", " ") # "living room"

    # 3. 改造 Input: 替换显式房间名为 "here"
    original_input = case.get("input", "")
    
    # 关键修复：正则同时匹配 "in" 和 "on" (适配 "on the balcony")
    # 模式: 单词边界 + (in 或 on) + 空格 + 可选的the + 空格 + 房间名 + 单词边界
    pattern = re.compile(r'\b(?:in|on)\s+(?:the\s+)?' + re.escape(room_name) + r'\b', re.IGNORECASE)
    
    # 如果指令中确实提到了该房间，则替换为 here
    # (如果没提到，保持原样，但也注入位置信息，视为纯隐式)
    new_input = pattern.sub("here", original_input)

    # 4. 生成位置提示
    location_context = f"User is located in the {room_name}."
    
    return {
        "new_input": new_input,
        "location_context": location_context,
        "room_name": room_name
    }

# --- 辅助函数：将 JSON 状态转换为字符串 ---
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

# --- 辅助函数：应用 Chat 模板 ---
def apply_chat_template(tokenizer, system, user):
    """如果 Tokenizer 支持 Chat 模板则使用，否则使用默认拼接"""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # Fallback format
        return f"{system}\n\n{user}"

# --- Dataset 类 1: Zero-shot (IC Modified) ---
class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home:
            lines_home = f_home.readlines()
        
        home_status = {}
        for line in lines_home:
            data = json.loads(line)
            home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            self.system_prompt = f.read()
        
        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status: continue
                
                # --- IC 任务转换 ---
                ic_data = transform_ic_input(case)
                if ic_data is None: continue # 跳过非单指令数据
                
                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                
                home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                
                # 注入位置信息和修改后的指令
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions you need to reply to.\n" + \
                                        "<User instructions:> \n" + \
                                        ic_data['new_input'] + "\n" + \
                                        "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + user_instruction_case
                
                final_input = apply_chat_template(self.tokenizer, self.system_prompt, user_content)
                
                self.data.append({
                    "input_text": final_input,
                    "output": case["output"],
                    "type": case.get("type", "normal")
                })
            except Exception as e:
                print(f"Error processing line {i}: {e}")
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_text"], item["output"], item["type"]

# --- Dataset 类 2: Few-shot (IC Modified) ---
class home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home:
            lines_home = f_home.readlines()
        
        home_status = {}
        for line in lines_home:
            data = json.loads(line)
            home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        ex_path1 = os.path.join(code_dir, "example1.txt")
        ex_path2 = os.path.join(code_dir, "example.txt")
        if os.path.exists(ex_path1):
            with open(ex_path1, "r") as f: examples = f.read()
        elif os.path.exists(ex_path2):
            with open(ex_path2, "r") as f: examples = f.read()
        else:
            examples = ""

        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            self.system_prompt = f.read()
        
        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status: continue
                
                # --- IC 任务转换 ---
                ic_data = transform_ic_input(case)
                if ic_data is None: continue 

                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
                
                home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
                device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
                
                # 注入位置信息
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions you need to reply to.\n" + \
                                        "<User instructions:> \n" + \
                                        ic_data['new_input'] + "\n" + \
                                        "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + examples + user_instruction_case
                
                final_input = apply_chat_template(self.tokenizer, self.system_prompt, user_content)

                self.data.append({
                    "input_text": final_input,
                    "output": case["output"],
                    "type": case.get("type", "normal")
                })
            except Exception as e:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_text"], item["output"], item["type"]

# --- Dataset 类 3: RAG Mode (IC Modified) ---
class rag_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, model_name_for_rag):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        # 尝试加载 System Prompt
        sys_path = os.path.join(code_dir, "system.txt")
        if os.path.exists(sys_path):
             with open(sys_path, "r") as f: self.system_prompt = f.read()
        else:
             self.system_prompt = "You are a smart home assistant."

        filename_candidates = [f"{model_name_for_rag}_rag_test_data.json", "rag_test_data.json"]
        rag_path = None
        for fn in filename_candidates:
            p = os.path.join(dataset_dir, fn)
            if os.path.exists(p):
                rag_path = p
                break
        
        if not rag_path:
            candidates = glob.glob(os.path.join(dataset_dir, "*rag_test_data.json"))
            if candidates: rag_path = candidates[0]
        
        if not rag_path:
            raise FileNotFoundError(f"Could not find RAG dataset.")
            
        print(f"Loading RAG dataset from: {rag_path}")
        with open(rag_path, "r") as f:
            raw_data = json.load(f)
            
        self.data = []
        for item in raw_data:
            # --- IC 任务转换 ---
            ic_data = transform_ic_input(item)
            if ic_data is None: continue 

            original_input_str = item["input"]
            
            # 1. 替换 'in living room' 或 'on the balcony' -> 'here'
            # 修正：同样使用更新后的正则
            pattern = re.compile(r'\b(?:in|on)\s+(?:the\s+)?' + re.escape(ic_data['room_name']) + r'\b', re.IGNORECASE)
            modified_str = pattern.sub("here", original_input_str)
            
            # 2. 注入位置信息
            insert_marker = "<User instructions:>"
            if insert_marker in modified_str:
                parts = modified_str.split(insert_marker)
                final_rag_input = parts[0] + insert_marker + "\n" + ic_data['location_context'] + parts[1]
            else:
                final_rag_input = ic_data['location_context'] + "\n" + modified_str

            # 3. 再次应用 Template
            if "system" in item["input"]:
                final_input = final_rag_input
            else:
                final_input = apply_chat_template(self.tokenizer, self.system_prompt, final_rag_input)

            self.data.append({
                "input_text": final_input,
                "output": item["output"],
                "type": item.get("type", "normal")
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_text"], item["output"], item["type"]

# --- 主测试函数 ---
def model_test(model_name, use_rag=False, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=64):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp_enabled = local_rank != -1

    if ddp_enabled:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        if rank == 0: print(f"DDP Enabled. World Size: {world_size}")
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Running in Single Process Mode (No DDP).")

    sub_dirs = {
        "llama": "llama3-8b-Instruct",
        "qwen": "Qwen2.5-7B-Instruct",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "gemma": "Gemma-7B-Instruct-v0.3"
    }
    
    base_model_path_name = sub_dirs.get(model_name, model_name)
    base_model_dir = os.path.join(PROJECT_ROOT, "models", base_model_path_name)
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_sft")
    
    if rank == 0: 
        print(f"Loading Base Model from: {base_model_dir}")

    # Tokenizer
    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except:
        tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    load_device_map = None if ddp_enabled else "auto"
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch.bfloat16,
        device_map=load_device_map, 
        trust_remote_code=True
    )
    if ddp_enabled: model.to(device)

    # Load Adapter
    if use_finetuned:
        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            if rank == 0: print(f"Loading SFT Adapter from {adapter_dir}...")
            model = PeftModel.from_pretrained(model, adapter_dir)
            if ddp_enabled: model.to(device)
        else:
            if rank == 0: print(f"Warning: --use_finetuned is set but no adapter found. Running with Base Model.")

    model.eval()

    # Select Dataset
    if rank == 0: print("Loading test dataset (IC Mode)...")
    
    if use_rag:
        test_dataset = rag_home_assistant_dataset(tokenizer, base_model_path_name)
    elif use_few_shot:
        test_dataset = home_assistant_dataset(tokenizer)
    else:
        test_dataset = no_few_shot_home_assistant_dataset(tokenizer)
        
    if rank == 0: print(f"Dataset size: {len(test_dataset)}")
    
    sampler = DistributedSampler(test_dataset, shuffle=False) if ddp_enabled else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=4)
    
    res = []
    iterator = tqdm(test_loader, disable=(rank != 0))
    start_time = time.time()
    
    with torch.inference_mode():
        for inputs_str, output_texts, types in iterator:
            inputs = tokenizer(list(inputs_str), return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            for i in range(len(generated_texts)):
                res.append({
                    "generated": generated_texts[i].strip(),
                    "expected": output_texts[i],
                    "type": types[i]
                })
            
    if rank == 0: print(f"Inference Time: {time.time() - start_time:.2f}s")
    
    # Save Results
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_rag: mode_parts.append("rag")
    elif use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    if test_type and test_type != "normal": mode_suffix += f"_{test_type}"
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_part_{rank}.json")
    with open(part_file, "w") as f:
        f.write(json.dumps(res, indent=4, ensure_ascii=False))
    
    if ddp_enabled: dist.barrier()
    
    if rank == 0:
        print("Merging results...")
        final_res = []
        pattern = os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_part_*.json")
        for file_path in glob.glob(pattern):
            try:
                with open(file_path, "r") as f:
                    final_res.extend(json.load(f))
                os.remove(file_path)
            except: pass
        
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_IC_test_result.json")
        with open(final_file, "w") as f:
            f.write(json.dumps(final_res, indent=4, ensure_ascii=False))
        print(f"Saved to: {final_file}")

    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"])
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") 
    parser.add_argument("--test_type", type=str, default="normal")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    parser.add_argument("--batch_size", type=int, default=16) 
    args = parser.parse_args()
    
    if os.environ.get("LOCAL_RANK") is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    
    model_test(args.model_name, args.use_rag, args.use_few_shot, args.use_finetuned, args.test_type, args.batch_size)
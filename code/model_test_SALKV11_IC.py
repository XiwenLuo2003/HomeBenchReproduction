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
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- SALKV7 系统提示词 (直接指令生成) ---
STRICT_SYSTEM_PROMPT_SALKV7 = """You are a smart home control agent. 
Your task is to convert User Instructions into executable Python-style API calls based on the provided Home State and Device Methods.

RULES:
1. ONLY output the API calls. Do NOT output explanations.
2. **Partial Execution**: If the user instruction contains multiple parts, execute the valid parts and output `error_input` for the invalid parts.
3. If the entire instruction is invalid, output exactly: error_input
4. Separate multiple commands with newlines.
5. Do not include markdown code blocks.
"""

# --- IC 任务转换逻辑 ---
def transform_ic_input(case):
    if "one" not in case.get("id", ""): return None
    output_raw = case.get("output", [])
    cmd = output_raw[0] if isinstance(output_raw, list) and len(output_raw) > 0 else (output_raw if isinstance(output_raw, str) else "")
    if "." not in cmd: return None
    room_key = cmd.split(".")[0]
    room_name = room_key.replace("_", " ")
    
    # 支持 in 和 on
    pattern = re.compile(r'\\b(?:in|on)\\s+(?:the\\s+)?' + re.escape(room_name) + r'\\b', re.IGNORECASE)
    new_input = pattern.sub("here", case.get("input", ""))
    
    return {
        "new_input": new_input,
        "location_context": f"User is located in the {room_name}.",
        "room_name": room_name,
        "room_key": room_key # 用于 SALK 强制召回
    }

# --- SALKV7 核心：生成结果清洗 (解决幻觉复读) ---
def clean_generated_text_salkv7(text):
    """
    清洗模型生成结果，截断幻觉内容。(与 SALKV7 的逻辑完全一致)
    """
    # 1. 强力截断标记：一旦出现这些词，后面全部丢弃
    stop_signals = [
        "<User instructions:>", 
        "-------------------------------\n", 
        "<home_state>", 
        "<device_method>",
        "User:",
        "Machine instructions:",
        "Example:",
    ]
    
    min_index = len(text)
    for signal in stop_signals:
        idx = text.find(signal)
        if idx != -1 and idx < min_index:
            min_index = idx
            
    text = text[:min_index].strip()
            
    # 2. 清洗多余的 error_input
    if "." in text and "(" in text and "error_input" in text:
        lines = text.split('\n')
        valid_lines = []
        for line in lines:
            line = line.strip()
            if not line: continue
            if "." in line and "(" in line:
                valid_lines.append(line)
            elif "error_input" in line:
                pass
        
        if valid_lines:
            text = "\n".join(valid_lines)
        else:
            text = "error_input"
            
    lines = text.split('\n')
    normalized_commands = []
    for cmd in lines:
        cmd = cmd.strip()
        if not cmd: continue

        if cmd == "error_input":
            normalized_commands.append(cmd)
            continue

        def remove_quotes_from_params_inner(match):
            params_str = match.group(1).strip()
            # First, remove outer quotes if any
            cleaned_params_str = re.sub(r"""^['"](.*?)[\\\\\'\\\"]$""", r"\\1", params_str)

            # Attempt to evaluate simple arithmetic expressions if it contains operators
            if any(op in cleaned_params_str for op in ['+', '-', '*', '/']):
                try:
                    # Check if the string consists only of numbers, operators, and whitespace
                    if re.fullmatch(r'[\d\s\+\-\*\/]+', cleaned_params_str):
                        evaluated_result = eval(cleaned_params_str)
                        return f"({evaluated_result})"
                except (SyntaxError, TypeError, NameError):
                    # If evaluation fails, return the original cleaned string
                    pass # Fallback to original string

            return f"({cleaned_params_str})"

        cleaned_cmd = re.sub(r'\\((.*?)\\)', remove_quotes_from_params_inner, cmd)
        normalized_commands.append(cleaned_cmd)
    
    if not normalized_commands:
        return "error_input"
    return "\n".join(normalized_commands)

# --- 辅助函数：构建设备索引 (SALKV11_IC 中不再直接用于过滤，但保留以防其他模块调用) ---
def build_device_index(home_status):
    index = {}
    for room, devices in home_status.items():
        if room == "VacuumRobot":
            key = "vacuum"
            if key not in index: index[key] = []
            index[key].append(room)
            continue
            
        if isinstance(devices, dict):
            for device_name in devices.keys():
                if device_name == "room_name": continue
                clean_name = device_name.replace("_", " ").lower()
                
                if clean_name not in index: index[clean_name] = []
                index[clean_name].append(room)
                
                # 别名映射
                aliases = []
                if "light" in clean_name: aliases.extend(["lamp", "lights", "lamps"])
                if "air conditioner" in clean_name: aliases.extend(["ac", "a/c", "air condition", "aircon"])
                if "humidifier" in clean_name: aliases.extend(["humid", "humidifiers"])
                if "dehumidifier" in clean_name: aliases.extend(["dehumid", "dehumidifiers"])
                if "curtain" in clean_name: aliases.extend(["blind", "curtains", "blinds"])
                if "fan" in clean_name: aliases.extend(["fans"])
                if "player" in clean_name: aliases.extend(["music", "song", "media"])
                
                for alias in aliases:
                    if alias not in index: index[alias] = []
                    index[alias].append(room)
    return index

# --- SALK 上下文提取 (SALKV11_IC 中不再直接用于过滤，但保留以防其他模块调用) ---
def extract_context_smart_v6(user_input, all_rooms_list, device_index):
    user_input_lower = user_input.lower()
    explicit_rooms = set()
    
    multi_intent_keywords = ["and", ",", "all", "every", "both", ";", "also", "then"]
    is_multi_intent = any(kw in user_input_lower for kw in multi_intent_keywords)
    
    room_keyword_map = {
        "master": ["master bedroom"],
        "guest": ["guest bedroom"],
        "bed": ["master bedroom", "guest bedroom"],
        "living": ["living room"],
        "dining": ["dining room", "ding room"],
        "ding": ["dining room", "ding room"],
        "study": ["study room"],
        "bath": ["bathroom"],
        "kitchen": ["kitchen"],
        "garage": ["garage"],
        "balcony": ["balcony"],
        "foyer": ["foyer"],
        "corridor": ["corridor"],
        "store": ["store room"],
        "storage": ["store room"]
    }

    for room in all_rooms_list:
        if room.lower() in user_input_lower:
            explicit_rooms.add(room)
            
    for kw, targets in room_keyword_map.items():
        pattern = r'\\b' + re.escape(kw) + r'\\b'
        if re.search(pattern, user_input_lower):
            for target in targets:
                if target in all_rooms_list:
                    explicit_rooms.add(target)

    implicit_rooms = set()
    for device_kw, rooms in device_index.items():
        pattern = r'\\b' + re.escape(device_kw) + r'\\b'
        if re.search(pattern, user_input_lower):
            for room in rooms:
                implicit_rooms.add(room)

    final_rooms = explicit_rooms.copy()
    
    if is_multi_intent or not explicit_rooms:
        final_rooms.update(implicit_rooms)
    
    threshold = 6 if is_multi_intent else 3
    
    if len(final_rooms) > threshold:
        priority_zones = {"living room", "master bedroom", "kitchen", "corridor"}
        keep_rooms = explicit_rooms.copy()
        remaining = final_rooms - explicit_rooms
        for r in remaining:
            if r in priority_zones:
                keep_rooms.add(r)
        
        if is_multi_intent and len(keep_rooms) < 2:
             for r in list(remaining)[:2]:
                 keep_rooms.add(r)
                 
        if keep_rooms:
            final_rooms = keep_rooms

    return final_rooms

# --- 辅助函数：将 JSON 状态转换为结构化知识图谱字符串 (SALKV10/V11 新增) ---
def chang_json2structured_kg_str(state, methods):
    state_facts = []
    for room_name, devices_in_room in state.items():
        if room_name == "VacuumRobot":
            if isinstance(devices_in_room, dict):
                fact = f"- VacuumRobot state: {devices_in_room.get('state', 'N/A')}"
                if "attributes" in devices_in_room:
                    for attr_name, attr_val in devices_in_room["attributes"].items():
                        value = attr_val.get("value", "N/A")
                        options = attr_val.get("options")
                        lowest = attr_val.get("lowest")
                        highest = attr_val.get("highest")
                        if options:
                            fact += f", {attr_name}: {value} (options: {options})"
                        elif lowest is not None and highest is not None:
                            fact += f", {attr_name}: {value} (range: {lowest}-{highest})"
                        else:
                            fact += f", {attr_name}: {value}"
                state_facts.append(fact + ".")
        else:
            for device_name, device_obj in devices_in_room.items():
                if device_name == "room_name": continue
                
                fact = f"- {room_name} {device_name} state: {device_obj.get('state', 'N/A')}"
                if "attributes" in device_obj:
                    for attr_name, attr_val in device_obj["attributes"].items():
                        value = attr_val.get("value", "N/A")
                        options = attr_val.get("options")
                        lowest = attr_val.get("lowest")
                        highest = attr_val.get("highest")
                        if options:
                            fact += f", {attr_name}: {value} (options: {options})"
                        elif lowest is not None and highest is not None:
                            fact += f", {attr_name}: {value} (range: {lowest}-{highest})"
                        else:
                            fact += f", {attr_name}: {value}"
                state_facts.append(fact + ".")

    method_facts = []
    for method_entry in methods:
        room_prefix = method_entry["room_name"] + "." if method_entry["room_name"] != "None" else ""
        method_signature = f"{room_prefix}{method_entry['device_name']}.{method_entry['operation']}("
        if method_entry["parameters"]:
            params = [f"{p['name']}:{p['type']}" for p in method_entry["parameters"]]
            method_signature += ", ".join(params)
        method_signature += ");"
        method_facts.append(f"- Can perform: {method_signature}")

    structured_state_str = "<HomeStateKnowledgeGraph>\n" + "\n".join(state_facts) + "\n</HomeStateKnowledgeGraph>"
    structured_method_str = "<DeviceMethodKnowledgeGraph>\n" + "\n".join(method_facts) + "\n</DeviceMethodKnowledgeGraph>"

    return structured_state_str, structured_method_str

# --- 辅助函数：应用 Chat 模板 (支持 Gemma) ---
def apply_chat_template(tokenizer, system, user):
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        if "gemma" in tokenizer.name_or_path.lower():
            combined_user_content = f"{system}\n\n{user}"
            messages = [
                {"role": "user", "content": combined_user_content}
            ]
        else:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        return f"{system}\n\n{user}"

# --- Dataset: Zero-shot (SALK V11 IC) ---
class no_few_shot_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f: lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home: lines_home = f_home.readlines()
        
        home_status_full = {} 
        for line in lines_home:
            data = json.loads(line)
            home_status_full[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status_full: continue

                # 1. IC 任务转换
                ic_data = transform_ic_input(case)
                if ic_data is None: continue

                user_input = ic_data["new_input"] # 使用替换为 "here" 后的输入
                current_home_data = home_status_full[case["home_id"]]
                full_state = current_home_data["home_status"]
                full_methods = current_home_data["method"]
                
                # 2. SALKV11: 使用全量上下文 (无智能上下文提取)
                state_str_kg, method_str_kg = chang_json2structured_kg_str(full_state, full_methods)
                
                # 3. 构建 Prompt (注入位置)
                home_status_case = state_str_kg + "\n"
                device_method_case = method_str_kg + "\n"
                
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions you need to reply to.\n" + \
                                        "<User instructions:> \n" + \
                                        user_input + "\n" + \
                                        "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + user_instruction_case
                final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_SALKV7, user_content)
                
                self.data.append({
                    "input_text": final_input,
                    "output": case["output"],
                    "type": case.get("type", "normal")
                })
            except Exception as e:
                continue

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

# --- Dataset: Few-shot (SALK V11 IC) ---
class home_assistant_dataset(Dataset):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        with open(os.path.join(dataset_dir, "test_data.jsonl"), "r") as f: lines = f.readlines()
        with open(os.path.join(dataset_dir, "home_status_method.jsonl"), "r") as f_home: lines_home = f_home.readlines()
        home_status_full = {}
        for line in lines_home:
            data = json.loads(line)
            home_status_full[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        ex_path = os.path.join(code_dir, "example1.txt")
        if not os.path.exists(ex_path): ex_path = os.path.join(code_dir, "example.txt")
        examples = open(ex_path, "r").read() if os.path.exists(ex_path) else ""

        self.data = []
        for i in range(len(lines)):
            try:
                case = json.loads(lines[i])
                if case["home_id"] not in home_status_full: continue
                
                ic_data = transform_ic_input(case)
                if ic_data is None: continue

                user_input = ic_data["new_input"]
                full_state = home_status_full[case["home_id"]]["home_status"]
                full_methods = home_status_full[case["home_id"]]["method"]
                
                # SALKV11: 使用全量上下文 (无智能上下文提取)
                state_str_kg, method_str_kg = chang_json2structured_kg_str(full_state, full_methods)
                home_status_case = state_str_kg + "\n"
                device_method_case = method_str_kg + "\n"
                user_instruction_case = "-------------------------------\n" + \
                                        f"{ic_data['location_context']}\n" + \
                                        "Here are the user instructions you need to reply to.\n" + \
                                        "<User instructions:> \n" + \
                                        user_input + "\n" + "<Machine instructions:>"
                
                user_content = home_status_case + device_method_case + examples + user_instruction_case
                final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_SALKV7, user_content)
                self.data.append({"input_text": final_input, "output": case["output"], "type": case.get("type", "normal")})
            except: continue
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

# --- RAG Class (SALK V11 IC) ---
class rag_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, model_name_for_rag):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        filename_candidates = [f"{model_name_for_rag}_rag_test_data.json", "rag_test_data.json"]
        rag_path = next((os.path.join(dataset_dir, f) for f in filename_candidates if os.path.exists(os.path.join(dataset_dir, f))), None)
        if not rag_path:
            cand = glob.glob(os.path.join(dataset_dir, "*rag_test_data.json"))
            if cand: rag_path = cand[0]
        if not rag_path: raise FileNotFoundError("RAG dataset not found")
        print(f"Loading RAG: {rag_path}")
        with open(rag_path, "r") as f: raw_data = json.load(f)
        self.data = []
        for item in raw_data:
            ic_data = transform_ic_input(item)
            if ic_data is None: continue
            
            # RAG Input Modification: 这里我们直接构建 SALKV11 的 Prompt 结构
            # 不再假设 item["input"] 已经是 RAG 格式，而是从头构建，以确保结构化知识图谱的注入
            
            current_home_data = home_status_full[item["home_id"]]
            full_state = current_home_data["home_status"]
            full_methods = current_home_data["method"]
            
            # SALKV11: 使用全量上下文 (无智能上下文提取)
            state_str_kg, method_str_kg = chang_json2structured_kg_str(full_state, full_methods)

            # 构造 RAG prompt 的 User content 部分
            rag_user_content_parts = [
                state_str_kg + "\n",
                method_str_kg + "\n"
            ]
            
            # 检查是否有 RAG 检索到的额外上下文（虽然 SALKV11 是全量，但 RAG 可能会添加额外信息）
            # 假设 RAG 原始数据中的 "input" 已经包含了 RAG 的提示信息，我们将其直接合并
            # 重要的是，我们要确保我们的结构化知识图谱信息优先，并且位置上下文被正确注入
            
            # 原始 RAG input 通常会包含 system / user messages 或者直接是一个长字符串
            # 这里的策略是，将原始 RAG input 中 User Instruction 之前的部分作为 RAG Context
            # 然后在构建的结构化知识图谱之后，注入这个 RAG Context

            # 提取原始 RAG input 中的 "<User instructions:>" 之前的部分作为 RAG 上下文
            # 假设 RAG 的 input 已经是一个包含 system/user 格式的字符串，或者是一个纯文本
            original_rag_input_text = item["input"]
            user_instr_marker = "<User instructions:>"
            rag_context_before_user_instr = ""
            if user_instr_marker in original_rag_input_text:
                rag_context_before_user_instr = original_rag_input_text.split(user_instr_marker)[0].strip()
                if "<system>" in rag_context_before_user_instr.lower(): # 移除 system 部分，只保留 RAG context
                    rag_context_before_user_instr = rag_context_before_user_instr.split("<system>")[0].strip()
                elif "<user>" in rag_context_before_user_instr.lower(): # 移除 user 部分
                     rag_context_before_user_instr = rag_context_before_user_instr.split("<user>")[0].strip()
            
            if rag_context_before_user_instr: # 如果有 RAG 额外上下文，则加入
                rag_user_content_parts.append(rag_context_before_user_instr + "\n")

            # 构建最终的 User Instruction 部分，包含 IC 的位置上下文
            final_user_instruction_case = "-------------------------------\n" + \
                                            f"{ic_data['location_context']}\n" + \
                                            "Here are the user instructions you need to reply to.\n" + \
                                            "<User instructions:> \n" + \
                                            ic_data["new_input"] + "\n" + \
                                            "<Machine instructions:>"
            
            rag_user_content_parts.append(final_user_instruction_case)
            final_rag_user_content = "".join(rag_user_content_parts)

            final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_SALKV7, final_rag_user_content)
            
            self.data.append({"input_text": final_input, "output": item["output"], "type": item.get("type", "normal")})
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]["input_text"], self.data[idx]["output"], self.data[idx]["type"]

def model_test(model_name, use_rag=False, use_few_shot=False, use_finetuned=False, test_type=None, batch_size=64):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
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
    
    if rank == 0: print(f"Loading Model: {base_model_dir}")
    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try: tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except: tokenizer = Qwen2TokenizerFast(tokenizer_file=os.path.join(tokenizer_source, "tokenizer.json"))
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.bfloat16, device_map=None if ddp_enabled else "auto", trust_remote_code=True)
    if ddp_enabled: model.to(device)
    if use_finetuned and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
        model = PeftModel.from_pretrained(model, adapter_dir)
        if ddp_enabled: model.to(device)

    if rank == 0: print("Loading dataset (SALK V11 + IC)...")
    if use_rag: test_dataset = rag_home_assistant_dataset(tokenizer, base_model_path_name)
    elif use_few_shot: test_dataset = home_assistant_dataset(tokenizer)
    else: test_dataset = no_few_shot_home_assistant_dataset(tokenizer)
    
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
                max_new_tokens=256, 
                do_sample=False, 
                repetition_penalty=1.1, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            response = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response, skip_special_tokens=True)
            for i in range(len(generated_texts)):
                cleaned = clean_generated_text_salkv7(generated_texts[i])
                res.append({"generated": cleaned, "expected": output_texts[i], "type": types[i]})
            
    if rank == 0: print(f"Time: {time.time() - start_time:.2f}s")
    
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_rag: mode_parts.append("rag")
    elif use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    mode_suffix = "_".join(mode_parts)
    if test_type and test_type != "normal": mode_suffix += f"_{test_type}"
    mode_suffix += "_SALK_V11_IC" # 后缀
    
    part_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_{rank}.json")
    with open(part_file, "w") as f: f.write(json.dumps(res, indent=4, ensure_ascii=False))
    
    if ddp_enabled: dist.barrier()
    if rank == 0:
        final_res = []
        for file_path in glob.glob(os.path.join(output_dir, f"{model_name}_{mode_suffix}_part_*.json")):
            try:
                with open(file_path, "r") as f: final_res.extend(json.load(f))
                os.remove(file_path)
            except: pass
        final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_test_result.json")
        with open(final_file, "w") as f: f.write(json.dumps(final_res, indent=4, ensure_ascii=False))
        print(f"Saved: {final_file}")
    if ddp_enabled: dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IC-specific tests for HomeBench using SALKV11 architecture (Structured Knowledge + Full Context).")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"])
    parser.add_argument("--use_rag", action="store_true")
    parser.add_argument("--use_few_shot", action="store_true")
    parser.add_argument("--use_finetuned", action="store_true") 
    parser.add_argument("--test_type", type=str, default="normal")
    parser.add_argument("--cuda_devices", type=str, default="0,1")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    if os.environ.get("LOCAL_RANK") is None: os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    model_test(args.model_name, args.use_rag, args.use_few_shot, args.use_finetuned, args.test_type, args.batch_size)

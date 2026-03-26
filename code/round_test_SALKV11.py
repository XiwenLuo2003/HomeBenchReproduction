import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- SALKV7 系统提示词 (直接指令生成) ---
# 这个系统提示词与 round_test.py 类似，但更明确
STRICT_SYSTEM_PROMPT_SALKV7 = """You are a smart home control agent. 
Your task is to convert User Instructions into executable Python-style API calls based on the provided Home State and Device Methods.

RULES:
1. ONLY output the API calls. Do NOT output explanations.
2. **Partial Execution**: If the user instruction contains multiple parts, execute the valid parts and output `error_input` for the invalid parts.
3. If the entire instruction is invalid, output exactly: error_input
4. Separate multiple commands with newlines.
5. Do not include markdown code blocks.
"""

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

# --- 辅助函数：应用 Chat 模板 ---
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
                    if re.fullmatch(r'[\d\s\+\-\*\/\.]+', cleaned_params_str):
                        evaluated_result = eval(cleaned_params_str)
                        return f"({evaluated_result})"
                except (SyntaxError, TypeError, NameError):
                    # If evaluation fails, return the original cleaned string
                    pass # Fallback to original string

            return f"({cleaned_params_str})"

        cleaned_cmd = re.sub(r'\((.*?)\)', remove_quotes_from_params_inner, cmd)
        normalized_commands.append(cleaned_cmd)
    
    if not normalized_commands:
        return "error_input"
    return "\n".join(normalized_commands)


# --- Dataset 类 11: SALKV11 (全量上下文 + 结构化知识表示) ---
class round_test_SALKV11_dataset(Dataset):
    def __init__(self, tokenizer, home_id_file, use_few_shot=False):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")

        if home_id_file.startswith("dataset/"):
            home_id_file = home_id_file[len("dataset/"):]
        filepath = os.path.join(dataset_dir, home_id_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Specified home ID data file not found: {filepath}")
        
        print(f"Loading round test data from: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        home_status_path = os.path.join(dataset_dir, "home_status_method.jsonl")
        with open(home_status_path, "r") as f_home:
            lines_home = f_home.readlines()
        
        self.home_status_full_map = {}
        for line in lines_home:
            data = json.loads(line)
            self.home_status_full_map[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}

        self.examples_prompt = ""
        if use_few_shot:
            ex_path1 = os.path.join(code_dir, "example1.txt")
            ex_path2 = os.path.join(code_dir, "example.txt")
            if os.path.exists(ex_path1):
                with open(ex_path1, "r") as f: self.examples_prompt = f.read()
            elif os.path.exists(ex_path2):
                with open(ex_path2, "r") as f: self.examples_prompt = f.read()

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        case = self.raw_data[idx]
        home_id = case["home_id"]
        user_input = case["input"]

        if home_id not in self.home_status_full_map:
            raise ValueError(f"Home ID {home_id} not found in home_status_method.jsonl")

        full_home_info = self.home_status_full_map[home_id]
        # --- SALKV11 使用全量上下文 (无智能上下文提取) ---
        full_state = full_home_info["home_status"]
        full_methods = full_home_info["method"]
        # all_rooms_in_home = list(full_state.keys()) # 不再需要，因为不进行过滤

        # --- SALKV11 使用结构化知识表示 (处理全量上下文) ---
        state_str, method_str = chang_json2structured_kg_str(full_state, full_methods)

        user_instruction_case = (
            "-------------------------------\n" +
            "Here are the user instructions you need to reply to.\n" +
            "<User instructions:> \n" +
            user_input + "\n" +
            "<Machine instructions:>"
        )
        
        home_status_case = state_str # 直接使用结构化知识字符串
        device_method_case = method_str # 直接使用结构化知识字符串

        user_content_parts = [
            home_status_case,
            device_method_case
        ]
        if self.examples_prompt:
            user_content_parts.append(self.examples_prompt)
        user_content_parts.append(user_instruction_case)
        user_content = "".join(user_content_parts)

        final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT_SALKV7, user_content)
        
        return final_input, case["output"], case.get("type", "normal"), case["id"]

# --- 核心的 round_test 函数 ---
def round_test_salkv11(model_name, home_id_file, use_few_shot=False, use_finetuned=False, batch_size=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running in Single Process Mode for round testing.")
    
    sub_dirs = {
        "llama": "llama3-8b-Instruct",
        "qwen": "Qwen2.5-7B-Instruct",
        "mistral": "Mistral-7B-Instruct-v0.3",
        "gemma": "Gemma-7B-Instruct-v0.3"
    }
    
    base_model_path_name = sub_dirs.get(model_name, model_name)
    base_model_dir = os.path.join(PROJECT_ROOT, "models", base_model_path_name)
    adapter_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_sft")
    
    print(f"Loading Base Model from: {base_model_dir}")

    # Tokenizer
    tokenizer_source = adapter_dir if use_finetuned and os.path.exists(os.path.join(adapter_dir, "tokenizer.json")) else base_model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except:
        tokenizer_file_path = os.path.join(tokenizer_source, "tokenizer.json")
        if os.path.exists(tokenizer_file_path):
            tokenizer = Qwen2TokenizerFast(tokenizer_file=tokenizer_file_path)
        else:
            raise FileNotFoundError(f"Tokenizer files not found in {tokenizer_source}")
    
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        dtype=torch.bfloat16,
        device_map="auto", 
        trust_remote_code=True
    ).to(device)

    # Load Adapter
    if use_finetuned:
        if os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            print(f"Loading SFT Adapter from {adapter_dir}...")
            model = PeftModel.from_pretrained(model, adapter_dir).to(device)
        else:
            print(f"Warning: --use_finetuned is set but no adapter found. Running with Base Model.")

    model.eval()

    print("Loading round test dataset (SALKV11 - Structured Knowledge + Full Context)...")
    test_dataset = round_test_SALKV11_dataset(tokenizer, home_id_file, use_few_shot)
        
    print(f"Dataset size for round test: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    accumulated_results = []

    iterator = tqdm(test_loader)
    start_time = time.time()
    
    with torch.inference_mode():
        for batch_idx, (inputs_str, output_texts, types, ids) in enumerate(iterator):
            inputs = tokenizer(list(inputs_str), return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=256, # 与基线保持一致
                do_sample=False, 
                repetition_penalty=1.1, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            for i in range(len(generated_texts)):
                cleaned_generated_text = clean_generated_text_salkv7(generated_texts[i]) # 沿用 SALKV7 的清洗逻辑
                accumulated_results.append({
                    "id": ids[i],
                    "generated": cleaned_generated_text,
                    "expected": output_texts[i],
                    "type": types[i]
                })
            
    print(f"Inference Time for {len(accumulated_results)} instructions: {time.time() - start_time:.2f}s")
    
    # Save Final Results
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    mode_suffix += "_SALKV11"
    
    base_home_id_filename = os.path.basename(home_id_file)
    home_id = base_home_id_filename.replace("multi_rounds_of_", "").replace(".json", "")
    final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_round_test_{home_id}.json")
    
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(accumulated_results, f, ensure_ascii=False, indent=4)
    print(f"Saved accumulated round test results to: {final_file}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run round-based tests for HomeBench using SALKV11 architecture (Structured Knowledge + Full Context).")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"], help="Name of the model to test.")
    parser.add_argument("--home_id_file", type=str, required=True, help="Filename of the specific home ID dataset (e.g., multi_rounds_of_Home_59.json).")
    parser.add_argument("--use_few_shot", action="store_true", help="Whether to use Few-Shot Learning.")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") 
    parser.add_argument("--cuda_devices", type=str, default="0", help="Comma-separated list of CUDA device IDs to use. E.g., \"0,1\" or \"0\".")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference. Recommended to be 1 for round-based testing.")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    round_test_salkv11(args.model_name, args.home_id_file, args.use_few_shot, args.use_finetuned, args.batch_size)
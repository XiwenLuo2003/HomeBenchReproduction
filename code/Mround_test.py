import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 辅助函数：将 JSON 状态转换为字符串 (与 round_test.py 保持一致) ---
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

# --- 辅助函数：应用 Chat 模板 (与 round_test.py 保持一致) ---
def apply_chat_template(tokenizer, messages):
    # 检查是否支持 chat_template
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # `add_generation_prompt=True` 会在末尾添加 `assistant` 角色，期待模型生成
        return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    else:
        # Fallback 格式：简单拼接
        formatted_text = ""
        for msg in messages:
            if msg["role"] == "system":
                formatted_text += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                formatted_text += "User: " + msg["content"] + "\n"
            elif msg["role"] == "assistant":
                formatted_text += "Assistant: " + msg["content"] + "\n"
        return formatted_text

# --- 辅助函数：加载 System Prompt (新增) ---
def load_system_prompt(code_dir):
    with open(os.path.join(code_dir, "system.txt"), "r") as f:
        return f.read()

# --- 辅助函数：加载 Few-shot Examples (新增) ---
def load_examples_prompt(code_dir, use_few_shot):
    if not use_few_shot:
        return ""
    ex_path1 = os.path.join(code_dir, "example1.txt")
    ex_path2 = os.path.join(code_dir, "example.txt")
    if os.path.exists(ex_path1):
        with open(ex_path1, "r") as f: return f.read()
    elif os.path.exists(ex_path2):
        with open(ex_path2, "r") as f: return f.read()
    return ""

# --- 辅助函数：通用结果清洗 (从 eval.py 提取核心逻辑进行简化，用于去除幻觉等) ---
def clean_generated_text(text):
    """
    清洗模型生成结果，截断幻觉内容。
    """
    # 1. 强力截断标记：一旦出现这些词，后面全部丢弃
    # 这些是常见于原始提示词中的部分，模型可能会幻觉复读
    stop_signals = [
        "<User instructions:>", 
        "-------------------------------\n", 
        "<home_state>", 
        "<device_method>",
        "User:",
        "Machine instructions:",
        "Example:",
        "Thought:", # 包含CoT标签，以防万一
        "Code:",
    ]
    
    min_index = len(text)
    for signal in stop_signals:
        idx = text.find(signal)
        if idx != -1 and idx < min_index:
            min_index = idx
            
    text = text[:min_index].strip()

    # 2. 清洗多余的 error_input (如果有效指令和 error_input 同时出现)
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
            
    # 3. 对识别出的命令进行二次标准化，移除参数中的引号并评估算术表达式
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
            cleaned_params_str = re.sub(r"""^['"](.*?)[\\'\"]$""", r"\\1", params_str)

            # Attempt to evaluate simple arithmetic expressions if it contains operators
            if any(op in cleaned_params_str for op in ['+', '-', '*', '/']):
                try:
                    # Check if the string consists only of numbers, operators, and whitespace
                    if re.fullmatch(r'[\d\s\+\-\*\/\.]+', cleaned_params_str):
                        evaluated_result = eval(cleaned_params_str)
                        return f"({evaluated_result})"
                except (SyntaxError, TypeError, NameError):
                    pass 

            return f"({cleaned_params_str})"

        cleaned_cmd = re.sub(r'\((.*?)\)', remove_quotes_from_params_inner, cmd) # 注意这里，移除了反斜杠，与Python正则匹配一致
        normalized_commands.append(cleaned_cmd)
    
    if not normalized_commands:
        return "error_input"
    return "\n".join(normalized_commands)

# --- 核心的 Mround_test 函数 (Baseline 版本) ---
def Mround_test(model_name, home_id_file, use_few_shot=False, use_finetuned=False, batch_size=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices # 从命令行获取cuda_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running in Single Process Mode for multi-round testing.")
    
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
        torch_dtype=torch.bfloat16,
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

    # --- 1. 加载所有原始数据并按 home_id 分组 ---
    dataset_filepath = os.path.join(PROJECT_ROOT, "dataset", home_id_file)
    if not os.path.exists(dataset_filepath):
        raise FileNotFoundError(f"Specified home ID data file not found: {dataset_filepath}")
    with open(dataset_filepath, "r", encoding="utf-8") as f:
        raw_data_list = json.load(f)

    grouped_data = {}
    for item in raw_data_list:
        home_id = item["home_id"]
        if home_id not in grouped_data:
            grouped_data[home_id] = []
        grouped_data[home_id].append(item)

    # --- 2. 加载 home_status_method.jsonl 以获取家庭设备信息 (静态) ---
    home_status_path = os.path.join(PROJECT_ROOT, "dataset", "home_status_method.jsonl")
    with open(home_status_path, "r") as f_home:
        lines_home = f_home.readlines()
    home_status_map = {}
    for line in lines_home:
        data = json.loads(line)
        home_status_map[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}

    # --- 3. 加载系统提示词和 Few-shot 示例 ---
    code_dir = os.path.join(PROJECT_ROOT, "code")
    system_prompt = load_system_prompt(code_dir)
    examples_prompt = load_examples_prompt(code_dir, use_few_shot)

    accumulated_results = []
    total_instructions = sum(len(v) for v in grouped_data.values())
    pbar = tqdm(total=total_instructions, desc="Processing multi-round interactions")

    start_time = time.time()
    with torch.inference_mode():
        for home_id, instructions_for_home in grouped_data.items():
            dialogue_history = [] # Reset dialogue history for each new home
            home_info = home_status_map[home_id] # Get initial static info for this home

            for case in instructions_for_home:
                pbar.update(1)
                
                # Construct current home_status_block and device_method_block (these are static for the home)
                state_str, method_str = chang_json2str(home_info["home_status"], home_info["method"])
                
                home_status_block = f"<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:{state_str}\n</home_state>\n"
                device_method_block = f"<device_method>\n     The following provides the methods to control each device in the current household:{method_str}\n</device_method>\n"
                
                # Build messages list for apply_chat_template
                messages = []
                
                # Add system prompt as the first message (handled for Gemma inside apply_chat_template)
                messages.append({"role": "system", "content": system_prompt})

                # Add initial context (home state, methods, few-shot examples if any)
                initial_user_context = home_status_block + device_method_block
                if examples_prompt:
                    initial_user_context += examples_prompt
                messages.append({"role": "user", "content": initial_user_context.strip()})

                # Add dialogue history (previous user queries and model responses)
                for turn in dialogue_history:
                    if turn.startswith("User:"):
                        messages.append({"role": "user", "content": turn[len("User:"):].strip()})
                    elif turn.startswith("Machine:"):
                        messages.append({"role": "assistant", "content": turn[len("Machine:"):].strip()})
                
                # Add current user instruction
                current_user_instruction_content = f"-------------------------------\nHere are the user instructions you need to reply to.\n<User instructions:> \n{case['input']}\n<Machine instructions:>
"
                messages.append({"role": "user", "content": current_user_instruction_content.strip()})

                # Apply chat template to get the final input string
                final_input = apply_chat_template(tokenizer, messages)
                
                inputs = tokenizer(final_input, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
                
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=512, # 默认值，与 model_test.py 保持一致
                    do_sample=False, 
                    repetition_penalty=1.1, 
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                response_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
                generated_text = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0] # batch size is 1 here, so [0]
                
                # Clean generated text
                cleaned_generated_text = clean_generated_text(generated_text)

                # Store results for evaluation
                accumulated_results.append({
                    "id": case["id"],
                    "generated": cleaned_generated_text,
                    "expected": case["output"],
                    "type": case.get("type", "normal")
                })

                # Update dialogue history for the next round in this home
                dialogue_history.append(f"User: {case['input']}")
                dialogue_history.append(f"Machine: {cleaned_generated_text}")
                
    pbar.close()
    print(f"Inference Time for {total_instructions} instructions: {time.time() - start_time:.2f}s")
    
    # Save Final Results
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    mode_suffix += "_Mround_test" # 明确表示是多轮测试
    
    base_home_id_filename = os.path.basename(home_id_file)
    home_id_clean = base_home_id_filename.replace("multi_rounds_of_", "").replace(".json", "")
    final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_{home_id_clean}.json")
    
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(accumulated_results, f, ensure_ascii=False, indent=4)
    print(f"Saved accumulated multi-round test results to: {final_file}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-round interaction tests for HomeBench.")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"], help="Name of the model to test.")
    parser.add_argument("--home_id_file", type=str, required=True, help="Filename of the specific home ID dataset (e.g., multi_rounds_of_Home_59.json). This file contains a sequence of instructions for one home.")
    parser.add_argument("--use_few_shot", action="store_true", help="Whether to use Few-Shot Learning.")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") 
    parser.add_argument("--cuda_devices", type=str, default="0", help="Comma-separated list of CUDA device IDs to use. E.g., \"0,1\" or \"0\".")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference. For multi-round testing, it processes one instruction at a time, so this should ideally be 1.") # batch_size for inner loop (instruction processing)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    Mround_test(args.model_name, args.home_id_file, args.use_few_shot, args.use_finetuned, args.batch_size)






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
# 禁用 Tokenizers 并行，防止 DataLoader 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- V9 系统提示词：保持 V8 的 CoT 逻辑，控制变量 ---
STRICT_SYSTEM_PROMPT = """You are a smart home control agent. 
Your task is to convert User Instructions into executable Python-style API calls based on the provided Home State and Device Methods.

### PROCESS:
1. **Analyze**: Identify all devices and rooms mentioned in the User Instruction.
2. **Check**: Verify if each device exists in the <home_state> and has the requested method in <device_method>.
3. **Execute**: 
   - If a device exists and the method is valid, generate the API call.
   - If a device does NOT exist or the method is invalid, ignore it or output `error_input` only if the WHOLE instruction is invalid.
   - For Mixed Instructions (valid + invalid parts): ONLY generate API calls for the valid parts. Do NOT output `error_input` for the invalid parts in a mixed batch, just skip them.

### RULES:
1. First output your reasoning inside <Thought> tags.
2. Then output the final API calls inside <Code> tags.
3. If NO part of the instruction is valid, the code block should contain: error_input
4. Do NOT output markdown formatting (```).

### EXAMPLE:
User: Turn on the living room light and the bathroom toaster.
(Assume living room light exists, but bathroom toaster does not)
<Thought>
1. Living room light: Exists. Valid.
2. Bathroom toaster: Does not exist. Invalid.
Conclusion: Generate code for light, skip toaster.
</Thought>
<Code>
living_room.light.turn_on()
</Code>
"""

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

# --- V9 清洗函数：保持 V8 的清洗逻辑 --- (严格遵循 model_test_SALKV9.py)
def clean_generated_text(text):
    """
    从模型输出中提取 <Code>...</Code> 之间的内容。
    """
    # 1. 尝试提取 CoT 代码块
    code_pattern = r"<Code>(.*?)</Code>"
    match = re.search(code_pattern, text, re.DOTALL)
    
    if match:
        content = match.group(1).strip()
        return _clean_inner_logic(content)
    
    # 2. Fallback
    return _clean_inner_logic(text)

def _clean_inner_logic(text):
    stop_signals = ["<User instructions:>", "-------------------------------", "<home_state>", "<device_method>", "User:", "Machine instructions:", "Example:", "<Thought>"]
    min_index = len(text)
    for signal in stop_signals:
        idx = text.find(signal)
        if idx != -1 and idx < min_index:
            min_index = idx
    text = text[:min_index].strip()
    
    # 鲁棒性修改：确保最终输出是可解析的命令或 'error_input'
    lines = text.split('\n')
    processed_commands = []
    has_valid_command = False

    for line in lines:
        line = line.strip()
        if not line: continue

        if "." in line and "(" in line: # 看起来像一个有效的命令
            processed_commands.append(line)
            has_valid_command = True
        elif line == "error_input": # 明确的 error_input
            processed_commands.append(line)
        # 任何其他非命令文本或噪声都被忽略

    # 新增：对识别出的命令进行二次标准化，移除参数中的引号
    normalized_commands = []
    for cmd in processed_commands:
        # 仅对非 'error_input' 的命令进行处理
        if cmd == "error_input":
            normalized_commands.append(cmd)
            continue

        def remove_quotes_from_params_inner(match):
            params_str = match.group(1).strip() # Content inside the parentheses
            # 移除参数值最外层的单引号或双引号
            # 使用多行原始字符串字面量和明确的反向引用，提高解析鲁棒性
            cleaned_params_str = re.sub(r"""^['"](.*?)[\\'\"]$""", r"\\1", params_str)
            return f"({cleaned_params_str})"

        # 应用这个去除引号的逻辑到命令的所有括号部分
        cleaned_cmd = re.sub(r'\((.*?)\)', remove_quotes_from_params_inner, cmd)
        normalized_commands.append(cleaned_cmd)

    if not normalized_commands: # 如果没有找到任何命令或明确的 error_input
        return "error_input"
    elif has_valid_command: # 如果找到了有效命令
        # 按照 V8/V9 的 CoT 规则，如果存在有效指令，则移除纯粹的 'error_input' 行
        final_output_commands = [cmd for cmd in normalized_commands if cmd != "error_input"]
        if not final_output_commands: # 过滤后可能变空，此时也应为 error_input
            return "error_input"
        return "\n".join(final_output_commands)
    else: # 此时 normalized_commands 只包含纯粹的 "error_input" (没有有效命令)
        return "error_input"

# --- Dataset 类 1: Zero-shot (Adapted for round test with SALKV9 ablation: full context) ---
class round_test_SALKV9_dataset(Dataset):
    def __init__(self, tokenizer, home_id_file, use_few_shot=False): # SALKV9 默认不使用 RAG
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")

        # 加载指定 home_id 的数据文件
        # 确保 home_id_file 只是文件名，不包含 dataset/ 前缀
        if home_id_file.startswith("dataset/"):
            home_id_file = home_id_file[len("dataset/"):]
        filepath = os.path.join(dataset_dir, home_id_file)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Specified home ID data file not found: {filepath}")
        
        print(f"Loading round test data from: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f) # 假设这些文件是 JSON 数组

        # 加载 home_status_method.jsonl 以获取家庭设备信息
        home_status_path = os.path.join(dataset_dir, "home_status_method.jsonl")
        with open(home_status_path, "r") as f_home:
            lines_home = f_home.readlines()
        
        self.home_status_full_map = {} # 存储完整的家庭状态和方法
        for line in lines_home:
            data = json.loads(line)
            self.home_status_full_map[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}

        # 加载 few-shot examples (如果使用 few-shot 模式)
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
        # --- V9 Ablation: 直接使用全量状态，不做检索过滤 ---
        full_state = full_home_info["home_status"]
        full_methods = full_home_info["method"]
        # all_rooms_in_home = list(full_state.keys())

        # # SALKV9 移除智能上下文提取
        # device_index = build_device_index(full_state)
        # relevant_rooms = extract_context_smart_v6(user_input, all_rooms_in_home, device_index)

        # # 根据 relevant_rooms 过滤 home_status 和 methods
        # filtered_state = {}
        # for room_name, room_data in full_state.items():
        #     if room_name in relevant_rooms:
        #         filtered_state[room_name] = room_data

        # filtered_methods = []
        # for method_entry in full_methods:
        #     if method_entry["room_name"] in relevant_rooms or method_entry["room_name"] == "None":
        #         filtered_methods.append(method_entry)

        state_str, method_str = chang_json2str(full_state, full_methods)

        user_instruction_case = (
            "-------------------------------\n" +
            "Here are the user instructions you need to reply to.\n" +
            "<User instructions:> \n" +
            user_input + "\n" +
            "<Machine instructions:>"
        )
        
        home_status_case = ("<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n")
        device_method_case = ("<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n")

        user_content_parts = [
            home_status_case,
            device_method_case
        ]
        if self.examples_prompt: # Few-shot examples只在few-shot模式下添加
            user_content_parts.append(self.examples_prompt)
        user_content_parts.append(user_instruction_case)
        user_content = "".join(user_content_parts)

        # 使用 SALKV9 的 STRICT_SYSTEM_PROMPT
        final_input = apply_chat_template(self.tokenizer, STRICT_SYSTEM_PROMPT, user_content)
        
        return final_input, case["output"], case.get("type", "normal"), case["id"]

# --- 核心的 round_test 函数 ---
def round_test_salkv9(model_name, home_id_file, use_few_shot=False, use_finetuned=False, batch_size=1):
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
        # 对于 Qwen，可能需要 Qwen2TokenizerFast
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
        dtype=torch.bfloat16, # 使用 dtype 替代 torch_dtype
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

    # Select Dataset (使用 SALKV9 对应的数据集类)
    print("Loading round test dataset (SALKV9)...")
    test_dataset = round_test_SALKV9_dataset(tokenizer, home_id_file, use_few_shot)
        
    print(f"Dataset size for round test: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    accumulated_results = [] # 存储每条指令的生成结果，用于后续累计评估

    iterator = tqdm(test_loader)
    start_time = time.time()
    
    with torch.inference_mode():
        for batch_idx, (inputs_str, output_texts, types, ids) in enumerate(iterator):
            inputs = tokenizer(list(inputs_str), return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
            
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=512, # 与 model_test_SALKV9.py 保持一致，CoT 需要更多 token
                do_sample=False, 
                repetition_penalty=1.1, 
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
            generated_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            # 存储当前批次的生成结果，并应用 V9 清洗逻辑
            for i in range(len(generated_texts)):
                cleaned_generated_text = clean_generated_text(generated_texts[i])
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
    if use_few_shot: mode_parts.append("few_shot") # RAG 模式在这里是禁用状态
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    mode_suffix += "_SALKV9" # 添加 V9 特有后缀
    
    # 命名规则：model_name_mode_suffix_round_test_HomeID.json
    # 确保 home_id 不包含任何目录信息
    base_home_id_filename = os.path.basename(home_id_file) # 获取文件名，例如 'multi_rounds_of_Home_13_test.json'
    home_id = base_home_id_filename.replace("multi_rounds_of_", "").replace(".json", "") # 提取纯粹的 Home ID 部分
    final_file = os.path.join(output_dir, f"{model_name}_{mode_suffix}_round_test_{home_id}.json")
    
    with open(final_file, "w", encoding="utf-8") as f:
        json.dump(accumulated_results, f, ensure_ascii=False, indent=4)
    print(f"Saved accumulated round test results to: {final_file}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run round-based tests for HomeBench using SALKV9 architecture (Ablation: Full Context).")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"], help="Name of the model to test.")
    parser.add_argument("--home_id_file", type=str, required=True, help="Filename of the specific home ID dataset (e.g., multi_rounds_of_Home_59.json).")
    parser.add_argument("--use_few_shot", action="store_true", help="Whether to use Few-Shot Learning.")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") 
    parser.add_argument("--cuda_devices", type=str, default="0", help="Comma-separated list of CUDA device IDs to use. E.g., \"0,1\" or \"0\".")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference. Recommended to be 1 for round-based testing.") # 默认 batch size 为 1
    args = parser.parse_args()

    # 设置 CUDA 设备环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # 运行 round_test_salkv9
    # 注意：SALKV9 版本明确不使用 RAG 参数，因此这里不传递 use_rag
    round_test_salkv9(args.model_name, args.home_id_file, args.use_few_shot, args.use_finetuned, args.batch_size)

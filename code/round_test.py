import argparse
import torch
import os
import json
import re
import time
import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.distributed import DistributedSampler # 移除DDP相关
# import torch.distributed as dist # 移除DDP相关
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from peft import PeftModel

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 禁用 Tokenizers 并行，防止 DataLoader 死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

# --- Dataset 类 1: Zero-shot (Adapted for round test) ---
class round_test_dataset(Dataset):
    def __init__(self, tokenizer, home_id_file, use_rag=False, use_few_shot=False):
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
        
        self.home_status_map = {}
        for line in lines_home:
            data = json.loads(line)
            self.home_status_map[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}

        # 加载 system prompt
        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            self.system_prompt = f.read()

        # 加载 few-shot examples (如果使用 few-shot 模式)
        self.examples_prompt = ""
        if use_few_shot:
            ex_path1 = os.path.join(code_dir, "example1.txt")
            ex_path2 = os.path.join(code_dir, "example.txt")
            if os.path.exists(ex_path1):
                with open(ex_path1, "r") as f: self.examples_prompt = f.read()
            elif os.path.exists(ex_path2):
                with open(ex_path2, "r") as f: self.examples_prompt = f.read()

        self.use_rag = use_rag
        if use_rag:
            # RAG 模式需要预先生成 RAG 数据集，这里只做简单的检查和加载
            # 实际的 RAG 检索逻辑在 generate_rag_dataset 中完成
            # 这里我们假设 home_id_file 已经是 RAG 处理过的文件
            print("RAG mode selected for round test. Assuming input data is pre-RAG processed.")
            # 如果 home_id_file 已经是 RAG 处理过的 json，直接使用
            # 否则，需要额外逻辑来生成或加载 RAG 增强数据

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        case = self.raw_data[idx]
        home_id = case["home_id"]
        
        # 获取家庭状态和方法
        if home_id not in self.home_status_map:
            raise ValueError(f"Home ID {home_id} not found in home_status_method.jsonl")
        
        home_info = self.home_status_map[home_id]
        state_str, method_str = chang_json2str(home_info["home_status"], home_info["method"])

        # 构建用户指令部分
        user_instruction_case = (
            "-------------------------------\n" +
            "Here are the user instructions you need to reply to.\n" +
            "<User instructions:> \n" +
            case["input"] + "\n" +
            "<Machine instructions:>"
        )
        
        home_status_case = ("<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n")
        device_method_case = ("<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n")

        # 组合所有提示词元素
        user_content_parts = [
            home_status_case,
            device_method_case
        ]
        if self.examples_prompt: # Few-shot examples只在few-shot模式下添加
            user_content_parts.append(self.examples_prompt)
        user_content_parts.append(user_instruction_case)
        user_content = "".join(user_content_parts)
        
        # 最终输入通过 chat 模板处理
        final_input = apply_chat_template(self.tokenizer, self.system_prompt, user_content)
        
        return final_input, case["output"], case.get("type", "normal"), case["id"]

# --- RAG Dataset 类 (用于加载预生成的 RAG 数据) ---
class round_rag_dataset(Dataset):
    def __init__(self, tokenizer, home_id_file, model_name_for_rag):
        self.tokenizer = tokenizer
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        # 尝试加载 System Prompt
        sys_path = os.path.join(code_dir, "system.txt")
        if os.path.exists(sys_path):
             with open(sys_path, "r") as f: self.system_prompt = f.read()
        else:
             self.system_prompt = "You are a smart home assistant."

        # 根据 model_name_for_rag 和 home_id_file 来确定 RAG 数据文件名
        # 假设 RAG 数据生成脚本已经处理了这些文件并保存了
        # 例如：qwen_rag_multi_rounds_of_Home_59.json
        rag_filename = f"{model_name_for_rag}_rag_{home_id_file}"
        rag_path = os.path.join(dataset_dir, rag_filename)

        if not os.path.exists(rag_path):
            raise FileNotFoundError(f"RAG data file not found: {rag_path}. Please run rag_dataset_generation.py first.")
            
        print(f"Loading RAG dataset for round test from: {rag_path}")
        with open(rag_path, "r") as f:
            raw_data = json.load(f)
            
        self.data = []
        for item in raw_data:
            # 确保 RAG 数据也有 System Prompt 包裹
            # RAG数据生成时已经包含了system和examples
            final_input = item["input"]

            self.data.append({
                "input_text": final_input,
                "output": item["output"],
                "type": item.get("type", "normal"),
                "id": item.get("id") # 确保 ID 也被传递
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item["input_text"], item["output"], item["type"], item["id"]

# --- 核心的 round_test 函数 ---
def round_test(model_name, home_id_file, use_rag=False, use_few_shot=False, use_finetuned=False, batch_size=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices # 从命令行获取cuda_devices
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

    # Select Dataset
    print("Loading round test dataset...")
    if use_rag:
        test_dataset = round_rag_dataset(tokenizer, home_id_file, base_model_path_name)
    else:
        test_dataset = round_test_dataset(tokenizer, home_id_file, use_rag, use_few_shot)
        
    print(f"Dataset size for round test: {len(test_dataset)}")
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    accumulated_results = [] # 存储每条指令的生成结果，用于后续累计评估
    iteration_results = [] # 存储每轮迭代的累计结果

    iterator = tqdm(test_loader)
    start_time = time.time()
    
    with torch.inference_mode():
        for batch_idx, (inputs_str, output_texts, types, ids) in enumerate(iterator):
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
            
            # 存储当前批次的生成结果
            for i in range(len(generated_texts)):
                accumulated_results.append({
                    "id": ids[i],
                    "generated": generated_texts[i].strip(),
                    "expected": output_texts[i],
                    "type": types[i]
                })
            
            # # 在这里，您可以选择在每个批次结束后进行评估
            # # 但由于您要求最后再新建文件专门评估，这里只保存结果
            # # 假设您希望每次迭代都保存累计结果
            # iteration_results.append({
            #     "round": batch_idx + 1,
            #     "cumulative_results": list(accumulated_results) # 复制一份，防止后续修改
            # })
            # 或者，只在所有指令处理完成后保存最终累计结果
            
    print(f"Inference Time for {len(accumulated_results)} instructions: {time.time() - start_time:.2f}s")
    
    # Save Final Results
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    mode_parts = []
    if use_finetuned: mode_parts.append("sft")
    if use_rag: mode_parts.append("rag")
    elif use_few_shot: mode_parts.append("few_shot")
    else: mode_parts.append("zero_shot")
    
    mode_suffix = "_".join(mode_parts)
    
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
    parser = argparse.ArgumentParser(description="Run round-based tests for HomeBench.")
    parser.add_argument("--model_name", type=str, default="qwen", choices=["llama", "qwen", "mistral", "gemma"], help="Name of the model to test.")
    parser.add_argument("--home_id_file", type=str, required=True, help="Filename of the specific home ID dataset (e.g., multi_rounds_of_Home_59.json).")
    parser.add_argument("--use_rag", action="store_true", help="Whether to use Retrieval-Augmented Generation (RAG).")
    parser.add_argument("--use_few_shot", action="store_true", help="Whether to use Few-Shot Learning.")
    parser.add_argument("--use_finetuned", action="store_true", help="Load fine-tuned LoRA adapter.") 
    parser.add_argument("--cuda_devices", type=str, default="0", help="Comma-separated list of CUDA device IDs to use. E.g., \"0,1\" or \"0\".")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference. Recommended to be 1 for round-based testing.") # 默认 batch size 为 1
    args = parser.parse_args()

    # 设置 CUDA 设备环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices

    # 运行 round_test
    round_test(args.model_name, args.home_id_file, args.use_rag, args.use_few_shot, args.use_finetuned, args.batch_size)

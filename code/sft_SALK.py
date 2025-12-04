import argparse
import os
import sys
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import DataCollatorForSeq2Seq # 关键修改：SFT推荐使用Seq2Seq Collator
from transformers import Qwen2TokenizerFast
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import Dataset as hf_Dataset, IterableDataset

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- 全局房间列表 (SALK 核心) ---
ALL_ROOMS = [
    "master bedroom", "guest bedroom", "living room", "dining room", "ding room",
    "study room", "kitchen", "bathroom", "foyer", "corridor",
    "balcony", "garage", "store room"
]

# --- 论文原始 Prompt (Table 8) ---
PAPER_SYSTEM_PROMPT = """You are 'AI', a helpful AI Assistant that controls the devices in a house. Complete the following task as instructed or answer the following question with the information provided only. The current status of the device and the methods it possesses are provided below, please only use the methods provided. Output error_input when operating non-existent attributes and devices. Only output machine instructions and enclose them in {}."""

# --- 辅助函数：从用户输入中提取相关房间 ---
def extract_rooms_from_input(user_input, all_rooms_list):
    found_rooms = set()
    for room in all_rooms_list:
        if re.search(r'\b' + re.escape(room) + r'\b', user_input, re.IGNORECASE):
            found_rooms.add(room)
    return found_rooms

# --- 辅助函数：转换 JSON 为易读字符串 (严格对齐 model_test_SALK.py) ---
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

# --- 数据加载逻辑 (无Tokenizer依赖，纯数据准备) ---
class HomeAssistantDataProcessor:
    def __init__(self, dataset_type="train"):
        self.dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        self.dataset_type = dataset_type
        
        # 兼容不同的文件名
        if dataset_type == "test":
            self.file_path = os.path.join(self.dataset_dir, "test_data.jsonl")
        elif dataset_type == "train":
            # 优先尝试 part1_copy
            self.file_path = os.path.join(self.dataset_dir, "train_data_part1.jsonl")
            if not os.path.exists(self.file_path):
                 self.file_path = os.path.join(self.dataset_dir, "train_data_part1.jsonl")
        elif dataset_type == "val":
            self.file_path = os.path.join(self.dataset_dir, "valid_data.jsonl")
        
        # 预加载 Home Status
        self.home_status_full = {}
        home_status_path = os.path.join(self.dataset_dir, "home_status_method.jsonl")
        if os.path.exists(home_status_path):
            with open(home_status_path, "r") as f_home:
                for line in f_home:
                    data = json.loads(line)
                    self.home_status_full[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        self.data_lines = []
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as f:
                self.data_lines = f.readlines()
            print(f"Loaded {len(self.data_lines)} lines from {self.file_path}")
        else:
            print(f"Warning: Dataset file not found: {self.file_path}")

    def __len__(self):
        return len(self.data_lines)

    def process_item(self, line):
        try:
            case = json.loads(line)
        except json.JSONDecodeError:
            return None

        if case["home_id"] not in self.home_status_full:
            return None

        # --- SALK 核心逻辑：数据预处理 ---
        user_input = case["input"]
        relevant_rooms = extract_rooms_from_input(user_input, ALL_ROOMS)

        # 兜底逻辑：如果没找到房间，使用全量数据
        if not relevant_rooms:
            relevant_rooms = set(self.home_status_full[case["home_id"]]["home_status"].keys())

        current_home_data = self.home_status_full[case["home_id"]]
        full_state = current_home_data["home_status"]
        full_methods = current_home_data["method"]

        filtered_state = {}
        for room_name, room_data in full_state.items():
            if room_name in relevant_rooms or room_name == "VacuumRobot": 
                filtered_state[room_name] = room_data

        filtered_methods = []
        for method_entry in full_methods:
            if method_entry["room_name"] in relevant_rooms or method_entry["room_name"] == "None": 
                filtered_methods.append(method_entry)
        
        # 使用严格对齐的转换函数
        state_str, method_str = chang_json2str(filtered_state, filtered_methods)
        
        # --- 构造 Prompt 内容 ---
        home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
        device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
        user_instruction_case = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + user_input + "\n" + "<Machine instructions:>"
        
        user_content = home_status_case + device_method_case + user_instruction_case
        
        output = case["output"]
        output = output.replace("\'\'\'", "").replace(" ", "")
        
        return {
            "system": PAPER_SYSTEM_PROMPT,
            "user": user_content,
            "output": output
        }

    def generator(self):
        for line in self.data_lines:
            item = self.process_item(line)
            if item:
                yield item

# --- LoRA Config ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

def sft_model(model_name, batch_size=2, grad_accum=16):
    # --- 路径 ---
    models_dir = os.path.join(PROJECT_ROOT, "models")
    if model_name == "qwen":
        model_id = os.path.join(models_dir, "Qwen2.5-7B-Instruct")
    elif model_name == "llama":
        model_id = os.path.join(models_dir, "llama3-8b-Instruct")
    else:
        model_id = os.path.join(models_dir, "Qwen2.5-7B-Instruct")

    print(f"Loading model from: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="right")
    except:
        print("Using fallback tokenizer loading...")
        tokenizer = Qwen2TokenizerFast(vocab_file=os.path.join(model_id, "vocab.json"), merges_file=os.path.join(model_id, "merges.txt"), tokenizer_file=os.path.join(model_id, "tokenizer.json"), padding_side="right")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")
    
    # --- 修复：更鲁棒的 Chat Template 设置 ---
    if not tokenizer.chat_template:
        if "llama" in model_name.lower():
            print("Warning: Chat template not found. Setting default Llama-3 template.")
            tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
        else:
            print("Warning: Chat template not found. Setting default Qwen ChatML template.")
            tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    tokenizer.model_max_length = 4096 

    print("Initializing Data Processor...")
    processor = HomeAssistantDataProcessor(dataset_type="train")
    
    # --- 核心修复：Tokenization 函数 ---
    def tokenize_function(example):
        messages = [
            {"role": "system", "content": example["system"]},
            {"role": "user", "content": example["user"]}
        ]
        
        # 1. Tokenize Context
        try:
            context_tokens = tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                tokenize=True
            )
        except Exception as e:
            # Fallback specifically for weird template issues
            print(f"Template error: {e}, using basic concat")
            text = f"<|im_start|>system\n{example['system']}<|im_end|>\n<|im_start|>user\n{example['user']}<|im_end|>\n<|im_start|>assistant\n"
            context_tokens = tokenizer.encode(text)

        # 2. Tokenize Output
        output_tokens = tokenizer.encode(example["output"], add_special_tokens=False)
        output_tokens += [tokenizer.eos_token_id]
        
        # 3. 拼接
        input_ids = context_tokens + output_tokens
        
        # 4. 构建 labels (Masking user input)
        labels = [-100] * len(context_tokens) + output_tokens
        
        # 5. 截断
        max_len = tokenizer.model_max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
            
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

    print("Creating HF Dataset from generator (Memory Efficient)...")
    # 使用 from_generator 防止大规模数据导致 OOM
    hf_train_dataset = hf_Dataset.from_generator(processor.generator)
    
    print("Applying Tokenization...")
    # 移除不需要的列
    tokenized_dataset = hf_train_dataset.map(tokenize_function, remove_columns=["system", "user", "output"], num_proc=4)
    
    print(f"Train dataset size: {len(tokenized_dataset)}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    output_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_salk_sft")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        remove_unused_columns=True, # 使用 map 处理后可以开启这个
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=True,
        num_train_epochs=2,
        logging_steps=10,
        report_to="tensorboard",
        save_steps=100,
        logging_dir=os.path.join(output_dir, "logs"),
        packing=False, # 关闭 packing，因为我们手动处理了
    )
    
    # 关键修改：使用 DataCollatorForSeq2Seq
    # 它会自动处理 input_ids 的 padding (pad_token_id)
    # 并自动处理 labels 的 padding (-100)，这是 SFT 的标准做法
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, 
        padding=True,
        pad_to_multiple_of=8 
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    print("Starting SALK SFT Training...")
    trainer.train()
    trainer.save_model()
    print(f"SALK SFT Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    args = parser.parse_args()
    
    sft_model(args.model_name, args.batch_size, args.grad_accum)
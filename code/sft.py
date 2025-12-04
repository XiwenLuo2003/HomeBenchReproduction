import argparse
import os
import sys

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# 使用通用的 DataCollator，不再使用 CompletionOnly
from transformers import DataCollatorForLanguageModeling
from transformers import Qwen2TokenizerFast, PreTrainedTokenizerFast
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
from datasets import Dataset as hf_Dataset
from typing import Any, Dict, List, Union

# --- 辅助函数 (保持原样) ---
def chang_json2str(state, methods):
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if "state" in state[room]:
                state_str += "  state: " + str(state[room]["state"]) + "\n"
            if "attributes" in state[room] and isinstance(state[room]["attributes"], dict):
                for attribute in state[room]["attributes"].keys():
                    attr_obj = state[room]["attributes"][attribute]
                    state_str += "  " + attribute + ": " + str(attr_obj.get("value", "N/A"))
                    if "options" in attr_obj:
                        state_str += " (options" + str(attr_obj["options"]) + ")\n"
                    elif "lowest" in attr_obj:
                        low = attr_obj.get("lowest", "N/A")
                        high = attr_obj.get("highest", "N/A")
                        state_str += " (range: " + str(low) + " - " + str(high) + ")\n"
                    else:
                        state_str += "\n"
        else:
            for device in state[room].keys():
                if device == "room_name":
                    continue
                else:
                    device_obj = state[room][device]
                    if not isinstance(device_obj, dict):
                        continue
                    state_str += "  " + device + "\n"
                    if "state" in device_obj:
                        state_str += "    state: " + str(device_obj["state"]) + "\n"
                    if "attributes" in device_obj and isinstance(device_obj["attributes"], dict):
                        for attribute in device_obj["attributes"].keys():
                            attr_obj = device_obj["attributes"][attribute]
                            state_str += "    " + attribute + ": " + str(attr_obj.get("value", "N/A"))
                            if "options" in attr_obj:
                                state_str += " (options" + str(attr_obj["options"]) + ")\n"
                            elif "lowest" in attr_obj:
                                low = attr_obj.get("lowest", "N/A")
                                high = attr_obj.get("highest", "N/A")
                                state_str += " (range: " + str(low) + " - " + str(high) + ")\n"
                            else:
                                state_str += "\n"

    method_str = ""
    for method in methods:
        if method["room_name"] == "None":
            method_str += method["device_name"] + "." + method["operation"] + "("
        else:
            method_str += method["room_name"] + "." + method["device_name"] + "." + method["operation"] + "("
        if len(method["parameters"]) > 0:
            for parameter in method["parameters"]:
                method_str += parameter["name"] + ":" + parameter["type"] + ","
            method_str = method_str[:-1]
        method_str += ");"
    return state_str, method_str

# --- 重写的 Dataset 类 (核心修复) ---
class train_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, use_rag=False, dataset_type="train"):
        self.tokenizer = tokenizer
        self._data = []
        
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        # 兼容不同的文件名
        if dataset_type == "test":
            file_path = os.path.join(dataset_dir, "test_data.jsonl")
        elif dataset_type == "train":
            # 优先尝试 part1_copy，如果不存在则尝试 part1
            file_path = os.path.join(dataset_dir, "train_data_part1.jsonl")
            if not os.path.exists(file_path):
                 file_path = os.path.join(dataset_dir, "train_data_part1.jsonl")
        elif dataset_type == "val":
            file_path = os.path.join(dataset_dir, "valid_data.jsonl")
        
        if not os.path.exists(file_path):
            print(f"Warning: Dataset file not found: {file_path}")
            return

        with open(file_path, "r") as f:
            lines = f.readlines()
        
        home_status_path = os.path.join(dataset_dir, "home_status_method.jsonl")
        with open(home_status_path, "r") as f_home:
            lines_home = f_home.readlines()
        
        home_status = {}
        for line in lines_home:
            data = json.loads(line)
            home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
        
        with open(os.path.join(code_dir, "example.txt"), "r") as f:
            examples = f.read()
        with open(os.path.join(code_dir, "system.txt"), "r") as f:
            system = f.read()
        
        print(f"Processing {len(lines)} examples for {dataset_type}...")
        for i in range(len(lines)):
            case = lines[i]
            try:
                case = json.loads(case) 
            except json.JSONDecodeError:
                continue

            try:
                if case["home_id"] not in home_status:
                    continue
                state_str, method_str = chang_json2str(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
            except Exception as e:
                continue
            
            case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n<Machine instructions:>\n"
            
            home_status_case = "<home_state>\n  The following provides the status of all devices in each room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ state_str + "\n" + "</home_state>\n"
            
            device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ method_str + "\n" + "</device_method>\n"
            
            # 原始论文逻辑：把所有 context 拼接到 system prompt 中
            full_input = system + home_status_case + device_method_case + examples + case_input
            
            output = case["output"]
            output = output.replace("\'\'\'", "")
            output = output.replace(" ", "")
            output = "{" + output + "}"
            
            self._data.append({"input": full_input, "output": output})

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        
        # 1. 构建对话格式
        messages = [{"role": "system", "content": item["input"]}]
        
        # 2. Tokenize Context (不包含 output)
        # add_generation_prompt=True 会加上 <|im_start|>assistant\n
        context_tokens = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True
        )
        
        # 3. Tokenize Output (Response)
        output_tokens = self.tokenizer.encode(item["output"], add_special_tokens=False)
        # 添加 EOS
        output_tokens += [self.tokenizer.eos_token_id]
        
        # 4. 拼接 input_ids
        input_ids = context_tokens + output_tokens
        
        # 5. 手动构建 labels (核心修复点)
        # Context 部分设为 -100 (不计算 loss)
        # Output 部分设为 output_tokens (计算 loss)
        labels = [-100] * len(context_tokens) + output_tokens
        
        # 截断 (防止显存溢出，Qwen 32k context 很长，但训练通常不需要那么长)
        max_len = 8192 # 或者 4096，根据显存调整
        if len(input_ids) > max_len:
            # 简单的截断策略：保留最后的 max_len
            input_ids = input_ids[-max_len:]
            labels = labels[-max_len:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels
        }

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
        model_id = os.path.join(models_dir, "Qwen2.5-7B-Instruct") # Default

    print(f"Loading model from: {model_id}")
    
    # 简单的 Tokenizer 加载逻辑
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except:
        # Fallback for Qwen if auto fails
        print("Using fallback tokenizer loading...")
        tokenizer = Qwen2TokenizerFast(vocab_file=os.path.join(model_id, "vocab.json"), merges_file=os.path.join(model_id, "merges.txt"), tokenizer_file=os.path.join(model_id, "tokenizer.json"))

    # 确保有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 强制设置 model_max_length 避免警告
    tokenizer.model_max_length = 8192

    # 设置 Chat Template (如果 tokenizer 没带的话，Qwen2.5 通常自带)
    if not tokenizer.chat_template:
        print("Warning: Setting default Qwen chat template")
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    print("Loading datasets...")
    # 注意：这里会触发我们重写的 __getitem__，完成 Tokenization 和 Labels 构建
    train_dataset = train_home_assistant_dataset(tokenizer, dataset_type="train")
    
    # 为了避免 multiprocessing 问题，这里直接转换
    print("Converting to HF Dataset...")
    hf_train_dataset = hf_Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    
    print(f"Train dataset size: {len(hf_train_dataset)}")
    # 打印一条数据检查 labels 是否全为 -100
    sample = hf_train_dataset[0]
    print("Sample Label Check (First 10):", sample['labels'][:10])
    print("Sample Label Check (Last 10):", sample['labels'][-10:])
    # 检查是否有非 -100 的 label
    has_valid_label = any(l != -100 for l in sample['labels'])
    print(f"Has valid labels? {has_valid_label}")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    output_dir = os.path.join(PROJECT_ROOT, "model_output")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        remove_unused_columns=False, # 关键：设为 False，因为我们手动构建了 labels
        save_strategy="steps",
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        fp16=True,
        num_train_epochs=2,
        logging_steps=1,
        report_to="tensorboard",
        save_steps=100,
    )
    
    # 使用通用的 DataCollator，而不是 CompletionOnly
    # mlm=False 表示这是 Causal LM 任务 (padding 会被设为 -100)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator, # 使用通用 collator
        processing_class=tokenizer, # 关键修复：显式传递 tokenizer，防止 Trainer 自动加载错误的 Processor
    )
    
    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    args = parser.parse_args()
    
    sft_model(args.model_name, args.batch_size, args.grad_accum)
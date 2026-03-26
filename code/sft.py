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

# --- Dataset 类 ---
class train_home_assistant_dataset(Dataset):
    def __init__(self, tokenizer, use_rag=False, dataset_type="train"):
        self.tokenizer = tokenizer
        self._data = []
        
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        code_dir = os.path.join(PROJECT_ROOT, "code")
        
        if use_rag:
            rag_filename = f"rag_{dataset_type}_data.json" # 假设的命名规则
            # 实际情况可能需要更复杂的匹配，或者直接由 rag_dataset_generation.py 生成特定名字
            # 这里简化处理，如果找不到文件就报错或回退
            rag_path = os.path.join(dataset_dir, rag_filename)
            # 为了兼容我们生成的 rag 数据集，尝试匹配 model_name (这里 dataset 类没传 model_name，暂略)
            # 简单起见，如果 use_rag 为真，尝试读取通用的 rag 文件，或者由用户保证文件存在
            pass 

        # --- 普通模式逻辑 ---
        if dataset_type == "test":
            file_path = os.path.join(dataset_dir, "test_data.jsonl")
        elif dataset_type == "train":
            file_path = os.path.join(dataset_dir, "train_data_part1.jsonl")
            if not os.path.exists(file_path):
                 file_path = os.path.join(dataset_dir, "train_data_part1.jsonl")
        elif dataset_type == "val":
            file_path = os.path.join(dataset_dir, "valid_data.jsonl")
        else:
             file_path = os.path.join(dataset_dir, "train_data_part1.jsonl")
        
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
        
        messages = [{"role": "system", "content": item["input"]}]
        
        context_tokens = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=True
        )
        
        output_tokens = self.tokenizer.encode(item["output"], add_special_tokens=False)
        output_tokens += [self.tokenizer.eos_token_id]
        
        input_ids = context_tokens + output_tokens
        labels = [-100] * len(context_tokens) + output_tokens
        
        max_len = 8192 
        if len(input_ids) > max_len:
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

def sft_model(model_name, batch_size=2, grad_accum=16, use_rag=False):
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
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except:
        print("Using fallback tokenizer loading...")
        tokenizer = Qwen2TokenizerFast(vocab_file=os.path.join(model_id, "vocab.json"), merges_file=os.path.join(model_id, "merges.txt"), tokenizer_file=os.path.join(model_id, "tokenizer.json"))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = 8192

    if not tokenizer.chat_template:
        print("Warning: Setting default Qwen chat template")
        tokenizer.chat_template = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    print("Loading datasets...")
    train_dataset = train_home_assistant_dataset(tokenizer, use_rag=use_rag, dataset_type="train")
    
    print("Converting to HF Dataset...")
    hf_train_dataset = hf_Dataset.from_list([train_dataset[i] for i in range(len(train_dataset))])
    
    print(f"Train dataset size: {len(hf_train_dataset)}")
    
    if len(hf_train_dataset) > 0:
        sample = hf_train_dataset[0]
        print("Sample Label Check (First 10):", sample['labels'][:10])
        print(f"Has valid labels? {any(l != -100 for l in sample['labels'])}")
    else:
        print("Error: Dataset is empty!")
        return

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # --- 关键修改：为每个模型创建独立的输出子目录 ---
    # 格式: model_output/qwen_sft
    output_dir = os.path.join(PROJECT_ROOT, "model_output", f"{model_name}_sft")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    print(f"Model output directory set to: {output_dir}")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        remove_unused_columns=False,
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
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=hf_train_dataset,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    trainer.train()
    trainer.save_model() # 默认保存到 output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="qwen")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--use_rag", action="store_true")
    args = parser.parse_args()
    
    sft_model(args.model_name, args.batch_size, args.grad_accum, args.use_rag)
import torch
import os
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import Qwen2TokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerFast
import re
from tqdm import tqdm

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 设置 HF 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def calculate_similarity(query_vector, vector_list, threshold):
    """计算余弦相似度并筛选超过阈值的索引"""
    query_vector = query_vector / query_vector.norm(p=2)
    vector_list = vector_list / vector_list.norm(p=2, dim=1, keepdim=True)
    similarities = torch.matmul(vector_list, query_vector)
    indices = (similarities > threshold).nonzero(as_tuple=True)[0]
    return indices.tolist()

def calculate_topk_similarity(query_vector, vector_list, topk):
    """计算余弦相似度并返回 Top-K 索引"""
    query_vector = query_vector / query_vector.norm(p=2)
    vector_list = vector_list / vector_list.norm(p=2, dim=1, keepdim=True)
    similarities = torch.matmul(vector_list, query_vector)
    topk_indices = torch.topk(similarities, topk).indices
    return topk_indices.tolist()

def chang_json2strchunk(state, methods):
    """将设备状态和方法转换为按房间切分的 chunk 字符串列表"""
    state_str = ""
    for room in state.keys():
        state_str += room + ":\n"
        if room == "VacuumRobot":
            if "state" in state[room]:
                state_str += "  state: " + str(state[room]["state"]) + "\n"
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
                if "state" in device_obj:
                    state_str += "    state: " + str(device_obj["state"]) + "\n"
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
        state_str += "<chunk>"

    method_str = ""
    if not methods: return state_str, ""
        
    tmp_room_name = methods[0].get("room_name", "None")
    for method in methods:
        if method.get("room_name") != tmp_room_name:
            method_str += "<chunk>"
            tmp_room_name = method.get("room_name")
        
        if method.get("room_name") == "None":
            method_str += method["device_name"] + "." + method["operation"] + "("
        else:
            method_str += method.get("room_name", "") + "." + method["device_name"] + "." + method["operation"] + "("
        
        if len(method.get("parameters", [])) > 0:
            for parameter in method["parameters"]:
                method_str += parameter["name"] + ":" + parameter["type"] + ","
            method_str = method_str[:-1]
        method_str += "),"
    
    return state_str, method_str

def generate_rag_dataset(model_name, cuda_devices="0"):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
    print(f"Initializing RAG generation with model: {model_name}")
    
    # 1. 稳健的模型/Tokenizer 加载逻辑
    model_paths = {
        "qwen": os.path.join(PROJECT_ROOT, "models", "Qwen2.5-7B-Instruct"),
        "llama": os.path.join(PROJECT_ROOT, "models", "llama3-8b-Instruct"),
    }
    model_path = model_paths.get(model_name, model_name)
    
    if not os.path.exists(model_path) and model_name in model_paths:
         print(f"Warning: Local path {model_path} not found. Will try to load as HF repo ID or check path.")
    else:
         print(f"Found local model at: {model_path}")

    tokenizer = None
    config = None
    
    # --- 步骤 A: 加载 Config ---
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        # 显式设置 config 的 output_hidden_states 为 True
        config.output_hidden_states = True
    except Exception as e:
        print(f"Warning: Failed to load config with AutoConfig: {e}")

    # --- 步骤 B: 加载 Tokenizer (保持之前的修复) ---
    print(f"Loading tokenizer from: {model_path}")
    
    if "qwen" in model_name.lower():
        try:
            tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, trust_remote_code=True)
            print("Loaded using Qwen2TokenizerFast.")
        except:
            pass
    
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, config=config, trust_remote_code=True)
            print("Loaded using AutoTokenizer with config object.")
        except Exception as e:
            print(f"AutoTokenizer load failed: {e}")
            
    if tokenizer is None:
        print("Attempting manual tokenizer load...")
        tokenizer_json = os.path.join(model_path, "tokenizer.json")
        if os.path.exists(tokenizer_json):
            try:
                if "qwen" in model_name.lower():
                    tokenizer = Qwen2TokenizerFast(tokenizer_file=tokenizer_json)
                else:
                    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
                
                if hasattr(config, "pad_token_id") and config.pad_token_id is not None:
                    tokenizer.pad_token_id = config.pad_token_id
                if hasattr(config, "eos_token_id"):
                    tokenizer.eos_token_id = config.eos_token_id
                print("Loaded manually from tokenizer.json")
            except Exception as e:
                print(f"Manual load failed: {e}")
                
    if tokenizer is None:
        print("CRITICAL ERROR: Could not load tokenizer. Exiting.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # --- 步骤 C: 加载模型 ---
    print("Loading model for embeddings...")
    try:
        # 关键修复：
        # 1. 移除 output_hidden_states=True 参数，因为它可能不被 __init__ 支持
        # 2. 我们通过 config.output_hidden_states = True (上面已设置) 来控制
        # 3. 或者在 forward 时传递 output_hidden_states=True
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            config=config, 
            device_map="auto", 
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 加载数据
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    code_dir = os.path.join(PROJECT_ROOT, "code")
    test_data_path = os.path.join(dataset_dir, "test_data.jsonl")
    home_status_path = os.path.join(dataset_dir, "home_status_method.jsonl")
    system_path = os.path.join(code_dir, "system.txt")

    if not os.path.exists(test_data_path):
        print(f"Error: {test_data_path} not found.")
        return

    print("Loading datasets...")
    with open(test_data_path, "r") as f:
        lines = f.readlines()
    
    # --- 数据缩减 (如需全量，请注释掉) ---
    # lines = lines[:100] 
    # print(f"DEBUG MODE: Limiting to first {len(lines)} examples.")

    with open(home_status_path, "r") as f_home:
        lines_home = f_home.readlines()
    
    home_status = {}
    for line in lines_home:
        data = json.loads(line)
        home_status[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
    
    examples = ""
    if os.path.exists(os.path.join(code_dir, "example1.txt")):
        with open(os.path.join(code_dir, "example1.txt"), "r") as f: examples = f.read()
    elif os.path.exists(os.path.join(code_dir, "example.txt")):
        with open(os.path.join(code_dir, "example.txt"), "r") as f: examples = f.read()

    with open(system_path, "r") as f:
        system = f.read()

    rag_data = []
    print(f"Processing {len(lines)} cases...")
    
    for i in tqdm(range(len(lines))):
        try:
            case = json.loads(lines[i])
            state_str, method_str = chang_json2strchunk(home_status[case["home_id"]]["home_status"], home_status[case["home_id"]]["method"])
            
            state_str_list = [s for s in state_str.split("<chunk>") if s.strip()]
            method_str_list = [m for m in method_str.split("<chunk>") if m.strip()]
            
            prompt_template = "After thinking step by step, summry this sentence: {input}: " 
            state_inputs = [prompt_template.format(input=s) for s in state_str_list]
            method_inputs = [prompt_template.format(input=m) for m in method_str_list]
            
            if not state_inputs: state_inputs = [""]
            if not method_inputs: method_inputs = [""]

            state_tokens = tokenizer(state_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            method_tokens = tokenizer(method_inputs, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            query_tokens = tokenizer([case["input"]], return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                # 关键：在推理时显式开启 output_hidden_states=True
                # State
                state_outputs = model(**state_tokens, return_dict=True, output_hidden_states=True)
                state_embeddings = state_outputs.hidden_states[-1][:, -1, :]
                
                # Method
                method_outputs = model(**method_tokens, return_dict=True, output_hidden_states=True)
                method_embeddings = method_outputs.hidden_states[-1][:, -1, :]
                
                # Query
                query_outputs = model(**query_tokens, return_dict=True, output_hidden_states=True)
                query_embeddings = query_outputs.hidden_states[-1][:, -1, :][0]

            state_index = calculate_similarity(query_embeddings, state_embeddings, 0.5)
            if len(state_index) == 0:
                state_index = calculate_topk_similarity(query_embeddings, state_embeddings, min(3, len(state_embeddings)))
            
            method_index = calculate_similarity(query_embeddings, method_embeddings, 0.5)
            if len(method_index) == 0:
                method_index = calculate_topk_similarity(query_embeddings, method_embeddings, min(3, len(method_embeddings)))
            
            new_state_str = "".join([state_str_list[idx] for idx in state_index if idx < len(state_str_list)])
            new_method_str = "".join([method_str_list[idx] for idx in method_index if idx < len(method_str_list)])
            
            case_input = "-------------------------------\n" + "Here are the user instructions you need to reply to.\n" + "<User instructions:> \n" + case["input"] + "\n" + "<Machine instructions:>"
            home_status_case = "<home_state>\n  The following provides the status of all devices in the room of the current household, the adjustable attributes of each device, and the threshold values for adjustable attributes:"+ new_state_str + "\n" + "</home_state>\n"
            device_method_case = "<device_method>\n     The following provides the methods to control each device in the current household:"+ new_method_str + "\n" + "</device_method>\n"
            
            full_input = system + home_status_case + device_method_case + examples + case_input
            rag_data.append({"input": full_input, "output": case["output"], "type": case.get("type", "normal")}) # 添加 type 字段
            
        except Exception as e:
            print(f"Error processing case {i}: {e}")
            continue

    output_filename = f"{os.path.basename(model_name).split('/')[-1]}_rag_test_data.json"
    output_path = os.path.join(dataset_dir, output_filename)
    print(f"Saving RAG dataset to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(rag_data, f, indent=4)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RAG dataset using LLM embeddings.")
    parser.add_argument("--model_name", type=str, default="qwen", help="Model name or path")
    parser.add_argument("--cuda_devices", type=str, default="0", help="CUDA devices")
    args = parser.parse_args()
    generate_rag_dataset(args.model_name, args.cuda_devices)
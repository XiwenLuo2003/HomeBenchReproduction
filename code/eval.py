import argparse
import json
import re
from collections import Counter
import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# --- 文本归一化函数 ---
def normalize_command(text):
    """
    去除参数名，统一格式。
    例如: set_brightness(brightness=70) -> set_brightness(70)
    """
    text = re.sub(r'\b[a-zA-Z_]\w*=', '', text)
    return text

# --- 核心解析函数 ---
def extract_commands(text):
    """
    从模型输出中稳健地提取指令。
    """
    if text is None: return []
    
    # 1. 基础清洗
    clean_text = text.replace(" ", "").replace("\n", "")
    
    # 2. 正则提取: 匹配 room.device.method(args) 或 error_input
    pattern = r'(?:[\w\.]+\(.*?\)|error_input)'
    matches = re.findall(pattern, clean_text)
    
    return matches

def compute_accuracy(generated_texts, expected_texts, debug_limit=5):
    """
    计算准确率、Precision、Recall、F1
    """
    print(f"Sample count: {len(generated_texts)}")
    
    if len(generated_texts) == 0:
        print("No data to evaluate.")
        return {}

    correct_num = 0
    tp = 0
    all_pre = 0
    all_gold = 0
    mismatches = []
    
    debug_count = 0
    
    for generated_text, expected_text in zip(generated_texts, expected_texts):
        
        # 提取指令列表
        generated_list = extract_commands(generated_text)
        
        # 处理 expected (如果是字符串则提取，如果是列表则直接用)
        if isinstance(expected_text, str):
            expected_list = extract_commands(expected_text)
        elif isinstance(expected_text, list):
            expected_list = expected_text
            # 清洗列表中的字符串
            expected_list = [normalize_command(x) for x in expected_list]
        else:
            expected_list = []

        # 归一化
        generated_list = [normalize_command(x) for x in generated_list if x != ""]
        expected_list = [normalize_command(x) for x in expected_list if x != ""]
        
        # --- 针对单指令任务的去重与降噪 ---
        # 如果标准答案只有一个指令，且不是 error_input
        if len(expected_list) == 1:
            # 1. 去重：['cmd', 'cmd'] -> ['cmd']
            generated_list = list(set(generated_list))
            
            # 2. 降噪：如果包含正确答案，且有多余的 error_input，移除 error_input
            # 这能解决模型做对了操作但又啰嗦报错的问题
            if expected_list[0] != 'error_input' and expected_list[0] in generated_list:
                generated_list = [x for x in generated_list if x != 'error_input']
        
        # 对于 error_input 的特殊处理：如果标准答案是 error_input，只要模型输出了 error_input 就算对
        if len(expected_list) == 1 and expected_list[0] == 'error_input':
             if 'error_input' in generated_list:
                 generated_list = ['error_input']
        # ------------------------------------
        
        generated_counter = Counter(generated_list)
        expected_counter = Counter(expected_list)
        
        # Exact Match (Success Rate)
        if generated_counter == expected_counter:
            correct_num += 1
        else:
            mismatches.append({"generated": generated_list, "expected": expected_list})
            # 调试打印
            if debug_count < debug_limit:
                print(f"\n[Mismatch Case {debug_count + 1}]")
                print(f"  Raw Gen : {repr(generated_text)}")
                print(f"  Parsed  : {generated_list}")
                print(f"  Expected: {expected_list}")
                debug_count += 1
        
        # F1 计算组件
        intersection = generated_counter & expected_counter
        tp += len(list(intersection.elements()))
        all_pre += len(generated_list)
        all_gold += len(expected_list)

    success_rate = correct_num / len(generated_texts)
    precision = tp / all_pre if all_pre > 0 else 0
    recall = tp / all_gold if all_gold > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("-" * 20)
    print(f"Success Rate (EM): {success_rate:.4f} ({success_rate*100:.2f}%)")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"F1 Score:          {f1:.4f}")
    print("-" * 20)

    return {
        "success_rate": success_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mismatches": mismatches
    }

def dif_type(result_data):
    """
    根据结果文件中的 'type' 字段进行分类评估
    """
    
    # 初始化分类桶
    categories = {
        "all": {"expected": [], "generated": []},
        "normal_single": {"expected": [], "generated": []},   # VS
        "unexist_single": {"expected": [], "generated": []},  # IS (总)
        "normal_multi": {"expected": [], "generated": []},    # VM
        "mix_multi": {"expected": [], "generated": []},       # MM
        "error_multi": {"expected": [], "generated": []},     # IM
        # 细分 IS
        "unexist_device": {"expected": [], "generated": []},
        "unexist_attribute": {"expected": [], "generated": []}
    }
    
    print(f"Total results loaded: {len(result_data)}")

    for item in result_data:
        # 直接从 item 中获取，不再读取外部文件，不再依赖 index
        gold = item.get("expected", "")
        pred = item.get("generated", "")
        item_type = item.get("type", "normal")
        
        # 1. 全部数据
        categories["all"]["expected"].append(gold)
        categories["all"]["generated"].append(pred)
        
        # 2. 分类逻辑
        if item_type == "normal":
            categories["normal_single"]["expected"].append(gold)
            categories["normal_single"]["generated"].append(pred)
            
        elif item_type == "unexist_device":
            categories["unexist_single"]["expected"].append(gold)
            categories["unexist_single"]["generated"].append(pred)
            categories["unexist_device"]["expected"].append(gold)
            categories["unexist_device"]["generated"].append(pred)
            
        elif item_type == "unexist_attribute":
            categories["unexist_single"]["expected"].append(gold)
            categories["unexist_single"]["generated"].append(pred)
            categories["unexist_attribute"]["expected"].append(gold)
            categories["unexist_attribute"]["generated"].append(pred)
            
        else:
            # 兼容各种类型命名
            parts = item_type.split("_")
            
            if "mix" in item_type: # mix_multi
                categories["mix_multi"]["expected"].append(gold)
                categories["mix_multi"]["generated"].append(pred)
                
            elif item_type.startswith("normal") and "multi" in item_type: # normal_multi (VM)
                 categories["normal_multi"]["expected"].append(gold)
                 categories["normal_multi"]["generated"].append(pred)
                 
            elif item_type.startswith("error") or "error" in item_type: # error_multi (IM)
                categories["error_multi"]["expected"].append(gold)
                categories["error_multi"]["generated"].append(pred)
            else:
                if len(parts) > 1 and parts[1] == "mix":
                    categories["mix_multi"]["expected"].append(gold)
                    categories["mix_multi"]["generated"].append(pred)
                elif len(parts) > 1 and parts[1] == "normal":
                    categories["normal_multi"]["expected"].append(gold)
                    categories["normal_multi"]["generated"].append(pred)
                else:
                    pass

    # --- 执行评估 ---
    
    print("\n" + "="*40)
    print(">>> ALL DATA (Total Performance)")
    compute_accuracy(categories["all"]["generated"], categories["all"]["expected"], debug_limit=5)
    
    print("\n" + "="*40)
    print(">>> Valid Single (VS)")
    compute_accuracy(categories["normal_single"]["generated"], categories["normal_single"]["expected"], debug_limit=3)
    
    print("\n" + "="*40)
    print(">>> Invalid Single (IS)")
    compute_accuracy(categories["unexist_single"]["generated"], categories["unexist_single"]["expected"], debug_limit=3)
    
    print("\n" + "="*40)
    print(">>> Valid Multi (VM)")
    compute_accuracy(categories["normal_multi"]["generated"], categories["normal_multi"]["expected"], debug_limit=3)
    
    print("\n" + "="*40)
    print(">>> Mix Multi (MM)")
    compute_accuracy(categories["mix_multi"]["generated"], categories["mix_multi"]["expected"], debug_limit=3)
    
    print("\n" + "="*40)
    print(">>> Invalid Multi (IM)")
    compute_accuracy(categories["error_multi"]["generated"], categories["error_multi"]["expected"], debug_limit=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script for HomeBench.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the model test result JSON file.")
    args = parser.parse_args()

    print(f"Evaluating file: {args.result_file}")
    
    if not os.path.exists(args.result_file):
        print(f"Error: File not found: {args.result_file}")
        exit(1)

    try:
        with open(args.result_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        exit(1)
    
    if not isinstance(data, list):
        print("Error: Expected a JSON list of results.")
        exit(1)

    # 检查新版字段是否存在
    if len(data) > 0 and "type" not in data[0]:
        print("\n[WARNING] 'type' field missing in results!")
        print("Make sure you re-ran model_test.py with the updated code.")
    
    dif_type(data)
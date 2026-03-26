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
    # Remove key=value pairs, e.g., 'brightness=' from 'brightness=60'
    text = re.sub(r'\b[a-zA-Z_]\w*=', '', text)
    
    # Attempt to evaluate simple arithmetic expressions within parentheses
    def evaluate_arithmetic_params(match):
        params_str = match.group(1).strip()
        # Check if the string consists only of numbers, operators, and whitespace
        if re.fullmatch(r'[\d\s\+\-\*\/\.]+', params_str):
            try:
                evaluated_result = eval(params_str)
                return f"({evaluated_result})"
            except (SyntaxError, TypeError, NameError):
                pass # Fallback to original string if evaluation fails
        return f"({params_str})"

    # Apply arithmetic evaluation to content within parentheses
    text = re.sub(r'\((.*?)\)', evaluate_arithmetic_params, text)
    
    return text

# --- 核心解析函数 ---
def extract_commands(text):
    if text is None: return []
    clean_text = text.replace(" ", "").replace("\n", "")
    pattern = r'(?:[\w\.]+\(.*?\)|error_input)'
    matches = re.findall(pattern, clean_text)
    return matches

def compute_accuracy(generated_texts, expected_texts, debug_limit=5):
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
        generated_list = extract_commands(generated_text)
        
        if isinstance(expected_text, str):
            expected_list = extract_commands(expected_text)
        elif isinstance(expected_text, list):
            expected_list = expected_text
            expected_list = [normalize_command(x) for x in expected_list]
        else:
            expected_list = []

        generated_list = [normalize_command(x) for x in generated_list if x != ""]
        expected_list = [normalize_command(x) for x in expected_list if x != ""]
        
        # --- 针对单指令任务的去重与降噪 ---
        if len(expected_list) == 1:
            generated_list = list(set(generated_list))
            if expected_list[0] != 'error_input' and expected_list[0] in generated_list:
                generated_list = [x for x in generated_list if x != 'error_input']
        
        if len(expected_list) == 1 and expected_list[0] == 'error_input':
             if 'error_input' in generated_list:
                 generated_list = ['error_input']
        # ------------------------------------
        
        generated_counter = Counter(generated_list)
        expected_counter = Counter(expected_list)
        
        if generated_counter == expected_counter:
            correct_num += 1
        else:
            mismatches.append({"generated": generated_list, "expected": expected_list})
            if debug_count < debug_limit:
                print(f"\n[Mismatch Case {debug_count + 1}]")
                print(f"  Raw Gen : {repr(generated_text)}")
                print(f"  Parsed  : {generated_list}")
                print(f"  Expected: {expected_list}")
                debug_count += 1
        
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
    # 初始化分类桶
    categories = {
        "all": {"expected": [], "generated": []},
        "normal_single": {"expected": [], "generated": []},
        "unexist_single": {"expected": [], "generated": []},
        "normal_multi": {"expected": [], "generated": []},
        "mix_multi": {"expected": [], "generated": []},
        "error_multi": {"expected": [], "generated": []}, # IM
        "unexist_device": {"expected": [], "generated": []},
        "unexist_attribute": {"expected": [], "generated": []},
        "unknown_type": {"expected": [], "generated": []} # 新增：用于捕获未知类型
    }
    
    print(f"Total results loaded: {len(result_data)}")

    for item in result_data:
        gold = item.get("expected", "")
        pred = item.get("generated", "")
        item_type = item.get("type", "unknown") # 默认值修改为 "unknown"
        
        categories["all"]["expected"].append(gold)
        categories["all"]["generated"].append(pred)
        
        # --- 升级版分类逻辑 ---
        
        # 1. Invalid Multi (IM) - 优先匹配复杂错误类型
        # 捕获所有 unexist_multi, multi2_unexist_device, multi3_unexist_attribute 等变体，以及以 error 开头的类型
        if (item_type == "unexist_multi") or \
           ("multi" in item_type and "unexist" in item_type) or \
           item_type.startswith("error"):
            categories["error_multi"]["expected"].append(gold)
            categories["error_multi"]["generated"].append(pred)
            
        # 2. Mix Multi (MM)
        elif item_type == "mix_multi":
            categories["mix_multi"]["expected"].append(gold)
            categories["mix_multi"]["generated"].append(pred)

        # 3. Valid Multi (VM)
        elif item_type == "normal_multi":
             categories["normal_multi"]["expected"].append(gold)
             categories["normal_multi"]["generated"].append(pred)
            
        # 4. Invalid Single (IS)
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
            
        # 5. Valid Single (VS) - 明确的 normal 类型
        elif item_type == "normal":
            categories["normal_single"]["expected"].append(gold)
            categories["normal_single"]["generated"].append(pred)
            
        else:
            # 兼容性分割判断 (兜底) - 针对可能存在的 multi_mix, multi_normal 等格式
            parts = item_type.split("_")
            if len(parts) > 1:
                if parts[1] == "mix":
                    categories["mix_multi"]["expected"].append(gold)
                    categories["mix_multi"]["generated"].append(pred)
                elif parts[1] == "normal":
                    categories["normal_multi"]["expected"].append(gold)
                    categories["normal_multi"]["generated"].append(pred)
                elif parts[0].startswith("multi") and parts[1] == "unexist":
                    categories["error_multi"]["expected"].append(gold)
                    categories["error_multi"]["generated"].append(pred)
                else:
                    print(f"Warning: Unhandled type '{item_type}'")
                    categories["unknown_type"]["expected"].append(gold) # 归入未知类型
                    categories["unknown_type"]["generated"].append(pred)
            else:
                print(f"Warning: Unhandled type '{item_type}'")
                categories["unknown_type"]["expected"].append(gold) # 归入未知类型
                categories["unknown_type"]["generated"].append(pred)

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

    print("\n" + "="*40)
    print(">>> Unknown Types (for Debugging)") # 新增：打印未知类型
    compute_accuracy(categories["unknown_type"]["generated"], categories["unknown_type"]["expected"], debug_limit=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script for HomeBench.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the model test result JSON file.")
    args = parser.parse_args()

    if not os.path.exists(args.result_file):
        print(f"Error: File not found: {args.result_file}")
        exit(1)

    try:
        with open(args.result_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        exit(1)
    
    if len(data) > 0 and "type" not in data[0]:
        print("\n[WARNING] 'type' field missing in results!")
    
    dif_type(data)
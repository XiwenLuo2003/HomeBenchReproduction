import argparse
import json
import re
from collections import Counter
import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

def normalize_command(text):
    text = re.sub(r'\b[a-zA-Z_]\w*=', '', text)
    return text

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
        
        if len(expected_list) == 1:
            generated_list = list(set(generated_list))
            if expected_list[0] != 'error_input' and expected_list[0] in generated_list:
                generated_list = [x for x in generated_list if x != 'error_input']
        
        if len(expected_list) == 1 and expected_list[0] == 'error_input':
             if 'error_input' in generated_list:
                 generated_list = ['error_input']
        
        generated_counter = Counter(generated_list)
        expected_counter = Counter(expected_list)
        
        if generated_counter == expected_counter:
            correct_num += 1
        else:
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
        "f1": f1
    }

def dif_type(result_data):
    # IC 任务只包含 VS 和 IS (由 "one" 筛选出)
    categories = {
        "all": {"expected": [], "generated": []},
        "normal_single": {"expected": [], "generated": []},   # VS
        "unexist_single": {"expected": [], "generated": []},  # IS
    }
    
    print(f"Total results loaded: {len(result_data)}")

    for item in result_data:
        gold = item.get("expected", "")
        pred = item.get("generated", "")
        item_type = item.get("type", "normal")
        
        categories["all"]["expected"].append(gold)
        categories["all"]["generated"].append(pred)
        
        if item_type == "normal":
            categories["normal_single"]["expected"].append(gold)
            categories["normal_single"]["generated"].append(pred)
        elif item_type in ["unexist_device", "unexist_attribute"]:
            categories["unexist_single"]["expected"].append(gold)
            categories["unexist_single"]["generated"].append(pred)
        # 对于 IC 任务，VM/IM/MM 应该很少见或没有

    print("\n" + "="*40)
    print(">>> IC TASK EVALUATION (Implicit Context)")
    print(">>> ALL DATA (Total Performance)")
    compute_accuracy(categories["all"]["generated"], categories["all"]["expected"], debug_limit=5)
    
    print("\n" + "="*40)
    print(">>> Valid Single (VS) - IC Mode")
    compute_accuracy(categories["normal_single"]["generated"], categories["normal_single"]["expected"], debug_limit=3)
    
    print("\n" + "="*40)
    print(">>> Invalid Single (IS) - IC Mode")
    compute_accuracy(categories["unexist_single"]["generated"], categories["unexist_single"]["expected"], debug_limit=3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Evaluation Script for HomeBench IC Task.")
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
    
    dif_type(data)
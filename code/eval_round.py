import argparse
import json
import os
import re
from collections import Counter
import pandas as pd

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# --- 从 eval.py 复制的核心评估逻辑 ---
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

def extract_commands(text):
    if text is None: return []
    clean_text = text.replace(" ", "").replace("\n", "") # 与 eval.py 保持一致，直接清理文本
    pattern = r'(?:[\w\.]+\(.*?\)|error_input)'
    matches = re.findall(pattern, clean_text)
    return matches

def compute_metrics(generated_texts, expected_texts):
    if len(generated_texts) == 0:
        return {"success_rate": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    correct_num = 0
    tp = 0
    all_pre = 0
    all_gold = 0
    
    for gen_text_raw, exp_text_raw in zip(generated_texts, expected_texts):
        generated_list = extract_commands(gen_text_raw)
        expected_list = extract_commands(exp_text_raw)

        generated_list = [normalize_command(x) for x in generated_list if x != ""]
        expected_list = [normalize_command(x) for x in expected_list if x != ""]
        
        # --- 针对单指令任务的去重与降噪 (与 eval.py 保持一致) ---
        if len(expected_list) == 1:
            generated_list = list(set(generated_list))
            if expected_list[0] != 'error_input' and expected_list[0] in generated_list:
                generated_list = [x for x in generated_list if x != 'error_input']
        
        if len(expected_list) == 1 and expected_list[0] == 'error_input':
             if 'error_input' in generated_list:
                 generated_list = ['error_input']
        # -----------------------------------------------------
        
        gen_counter = Counter(generated_list)
        exp_counter = Counter(expected_list)
        
        if gen_counter == exp_counter:
            correct_num += 1
        
        intersection = gen_counter & exp_counter
        tp += len(list(intersection.elements()))
        all_pre += len(generated_list)
        all_gold += len(expected_list)

    success_rate = correct_num / len(generated_texts) if len(generated_texts) > 0 else 0.0
    precision = tp / all_pre if all_pre > 0 else 0.0
    recall = tp / all_gold if all_gold > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"success_rate": success_rate, "precision": precision, "recall": recall, "f1": f1}

# --- 核心的累积评估函数 ---
def evaluate_cumulative_results(result_filepath):
    if not os.path.exists(result_filepath):
        raise FileNotFoundError(f"Result file not found: {result_filepath}")

    print(f"Loading results from: {result_filepath}")
    with open(result_filepath, "r", encoding="utf-8") as f:
        all_results = json.load(f)
    
    if not all_results:
        print("Result file is empty. No cumulative evaluation performed.")
        return

    cumulative_metrics_list = []
    cumulative_generated = []
    cumulative_expected = []

    print(f"Calculating cumulative metrics for {len(all_results)} rounds...")
    for i, item in enumerate(all_results):
        round_num = i + 1
        cumulative_generated.append(item["generated"])
        cumulative_expected.append(item["expected"])
        
        metrics = compute_metrics(cumulative_generated, cumulative_expected)
        cumulative_metrics_list.append({
            "round": round_num,
            "SR": metrics["success_rate"],
            "P": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"]
        })
        # print(f"Round {round_num}: SR={metrics['success_rate']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    # 保存到 CSV
    output_csv_filename = os.path.basename(result_filepath).replace(".json", ".csv")
    output_csv_path = os.path.join(OUTPUT_DIR, output_csv_filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.DataFrame(cumulative_metrics_list)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Cumulative evaluation results saved to: {output_csv_path}")

# --- 命令行参数解析 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate cumulative SR, P, Recall, F1 for round-based test results.")
    parser.add_argument("--result_file", type=str, required=True, help="Path to the JSON result file generated by round_test.py.")
    args = parser.parse_args()
    
    evaluate_cumulative_results(args.result_file)

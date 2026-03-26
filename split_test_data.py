import os
import json

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # 假设此脚本在项目根目录运行
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

INPUT_FILE = os.path.join(DATASET_DIR, "test_data.jsonl")
OUTPUT_PREFIX = os.path.join(DATASET_DIR, "test_data_part")
NUM_PARTS = 4

def split_jsonl_file(input_filepath, output_prefix, num_parts):
    print(f"Reading data from {input_filepath}...")
    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines in {input_filepath}: {total_lines}")

    if total_lines == 0:
        print("Input file is empty. No parts to create.")
        return

    # 计算每个部分的行数
    lines_per_part = total_lines // num_parts
    remainder = total_lines % num_parts

    start_idx = 0
    for i in range(num_parts):
        # 为前 'remainder' 个文件多分配一行
        end_idx = start_idx + lines_per_part + (1 if i < remainder else 0)
        
        output_filepath = f"{output_prefix}{i+1}.jsonl"
        part_lines = lines[start_idx:end_idx]
        
        print(f"Writing {len(part_lines)} lines to {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            f_out.writelines(part_lines)
        
        start_idx = end_idx
    
    print(f"Successfully split {input_filepath} into {num_parts} parts.")

if __name__ == "__main__":
    # 确保 dataset 目录存在
    os.makedirs(DATASET_DIR, exist_ok=True)
    split_jsonl_file(INPUT_FILE, OUTPUT_PREFIX, NUM_PARTS)

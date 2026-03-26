import os
import json

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # 假设此脚本在项目根目录运行
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

DATASET_FILES = [
    "test_data.jsonl",
    "valid_data.jsonl",
    "train_data_part1.jsonl",
    "train_data_part2.jsonl",
]

# 目标房间 ID
TARGET_HOME_IDS = [59, 13, 92]

# 输出文件名前缀
OUTPUT_FILE_PREFIX = os.path.join(DATASET_DIR, "multi_rounds_of_Home_")

def extract_and_save_data():
    # 为每个目标 home_id 初始化一个列表来存储数据
    extracted_data_by_home_id = {home_id: [] for home_id in TARGET_HOME_IDS}
    total_processed_instructions = 0

    print("Starting extraction of specific single-device instructions...")

    for filename in DATASET_FILES:
        filepath = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Warning: Dataset file not found: {filepath}. Skipping.")
            continue

        print(f"Processing {filename}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # 判断是否为单设备指令
                    is_single_device = ("one" in data.get("type", "") or "one" in data.get("id", ""))
                    home_id = data.get("home_id")
                    
                    if is_single_device and home_id in TARGET_HOME_IDS:
                        extracted_data_by_home_id[home_id].append(data)
                        total_processed_instructions += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line in {filename}: {line.strip()}")
                    continue
    
    print(f"Total single-device instructions matching target Home IDs found: {total_processed_instructions}")

    # 分别保存数据到文件
    for home_id in TARGET_HOME_IDS:
        output_filepath = f"{OUTPUT_FILE_PREFIX}{home_id}.json"
        print(f"Saving {len(extracted_data_by_home_id[home_id])} instructions for Home ID {home_id} to {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f_out:
            json.dump(extracted_data_by_home_id[home_id], f_out, ensure_ascii=False, indent=4)
    
    print("Extraction and saving completed.")

if __name__ == "__main__":
    # 确保 dataset 目录存在
    os.makedirs(DATASET_DIR, exist_ok=True)
    extract_and_save_data()

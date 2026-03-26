import os
import json
from collections import Counter

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

def analyze_single_device_instructions():
    home_id_counter = Counter()
    total_single_device_instructions = 0

    print("Starting analysis of single-device instructions across datasets...")

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
                    # 判断是否为单设备指令：根据 'type' 字段是否包含 'one'
                    # 或者更严格的判断，如 data["type"].startswith("normal") and "one" in data["id"]
                    # 这里使用简单的 'one' 包含判断
                    if "one" in data.get("type", "") or "one" in data.get("id", ""):
                        home_id = data.get("home_id")
                        if home_id is not None:
                            home_id_counter[home_id] += 1
                            total_single_device_instructions += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line in {filename}: {line.strip()}")
                    continue
    
    print(f"Total single-device instructions found: {total_single_device_instructions}")
    
    if not home_id_counter:
        print("No single-device instructions found.")
        return

    print("\nTop 3 home IDs with most single-device instructions:")
    # 按计数从高到低排序
    top_3_home_ids = home_id_counter.most_common(3)
    for home_id, count in top_3_home_ids:
        print(f"Home ID: {home_id}, Count: {count}")

if __name__ == "__main__":
    analyze_single_device_instructions()

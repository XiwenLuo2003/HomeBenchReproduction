import os
import json
from collections import Counter

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

DATASET_FILES = [
    "test_data.jsonl",
    "valid_data.jsonl",
    "train_data_part1.jsonl",
    "train_data_part2.jsonl",
]

def analyze_all_device_instructions():
    home_id_counter = Counter()
    total_instructions = 0

    print("Starting analysis of all instructions across datasets...")

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
                    home_id = data.get("home_id")
                    if home_id is not None:
                        home_id_counter[home_id] += 1
                        total_instructions += 1
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line in {filename}: {line.strip()}")
                    continue
    
    print(f"Total instructions found: {total_instructions}")
    
    if not home_id_counter:
        print("No instructions found.")
        return

    print("\nTop 3 home IDs with most instructions (all types):")
    # 按计数从高到低排序
    top_3_home_ids = home_id_counter.most_common(3)
    for home_id, count in top_3_home_ids:
        print(f"Home ID: {home_id}, Count: {count}")

if __name__ == "__main__":
    analyze_all_device_instructions()
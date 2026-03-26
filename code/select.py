import json
import os

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

INPUT_FILE = os.path.join(PROJECT_ROOT, "dataset", "test_data.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "dataset", "test_data_select.jsonl")

MAX_MULTI_INSTRUCTIONS = 3000

def select_multi_instruction_data():
    selected_data = []
    count = 0

    if not os.path.exists(INPUT_FILE):
        print(f"错误：输入文件不存在于 {INPUT_FILE}")
        return

    print(f"正在从 {INPUT_FILE} 筛选前 {MAX_MULTI_INSTRUCTIONS} 条多指令数据...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        for line in infile:
            try:
                data = json.loads(line)
                # 筛选包含 "multi" 类型的数据
                if "type" in data and "multi" in data["type"]:
                    selected_data.append(data)
                    count += 1
                    if count >= MAX_MULTI_INSTRUCTIONS:
                        break
            except json.JSONDecodeError:
                print(f"警告：跳过无效的 JSON 行: {line.strip()}")
                continue
    
    if not selected_data:
        print("未找到任何多指令数据。")
        return

    print(f"已筛选出 {len(selected_data)} 条多指令数据，正在写入 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for entry in selected_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("数据筛选完成并已保存。")

if __name__ == "__main__":
    select_multi_instruction_data()


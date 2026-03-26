import os
import json
import random
from collections import defaultdict

# --- 路径设置 ---
# SCRIPT_DIR 是此脚本所在的目录 (HomeBenchReproduction)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # 在这种情况下，项目根目录就是脚本的目录
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

DATASET_FILES = [
    "test_data.jsonl",
    "valid_data.jsonl",
    "train_data_part1.jsonl",
    "train_data_part2.jsonl",
]

OUTPUT_FILENAME = os.path.join(DATASET_DIR, "generated_invalid_multi_instructions.jsonl")
NUM_TARGET_INSTRUCTIONS = 2000
MIN_COMBINE = 3
MAX_COMBINE = 6

def generate_invalid_multi_instructions():
    invalid_single_instructions_by_home_id = defaultdict(list)
    
    print("Step 1: Filtering for invalid single-command instructions from all datasets...")

    # 步骤 1: 从所有数据集中筛选无效的单命令指令 (IS)
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
                    
                    # 判断是否为单命令指令 (通过 id 字段判断)
                    is_single_command = "_one_" in data.get("id", "")
                    # 判断是否为无效指令 (type 不为 "normal")
                    is_invalid = data.get("type") != "normal" 

                    if is_single_command and is_invalid:
                        home_id = data.get("home_id")
                        if home_id is not None:
                            invalid_single_instructions_by_home_id[home_id].append(data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line in {filename}: {line.strip()}")
                    continue
    
    print(f"Finished filtering. Found invalid single instructions for {len(invalid_single_instructions_by_home_id)} home IDs.")

    # 准备进行组合
    # 只考虑那些至少有 MAX_COMBINE 条无效单命令指令的 home_id
    # 这样可以确保在随机选择时有足够的指令可用。
    eligible_home_ids = [
        hid for hid, instrs in invalid_single_instructions_by_home_id.items() 
        if len(instrs) >= MAX_COMBINE 
    ]
    
    if not eligible_home_ids:
        print(f"Error: Not enough eligible home IDs with at least {MAX_COMBINE} invalid single instructions to form multi-commands. Found {len(invalid_single_instructions_by_home_id)} home IDs, but none meet the minimum count.")
        return

    generated_multi_instructions = []
    
    print(f"Step 2: Combining invalid single instructions into {NUM_TARGET_INSTRUCTIONS} invalid multi-command instructions...")

    # 步骤 2: 将 IS 指令组合成 IM 指令
    # 创建一个可变的 eligible_home_ids 副本，以防止在循环中修改原始列表
    current_eligible_home_ids = list(eligible_home_ids) 

    while len(generated_multi_instructions) < NUM_TARGET_INSTRUCTIONS and current_eligible_home_ids:
        
        # 从当前符合条件的 home_id 中随机选择一个
        selected_home_id = random.choice(current_eligible_home_ids)
        available_is_instructions = invalid_single_instructions_by_home_id[selected_home_id]
        
        # 确定要组合的指令数量 (3 到 6 条)
        # min() 确保我们不会尝试选择超过可用数量的指令
        num_to_combine = random.randint(MIN_COMBINE, min(MAX_COMBINE, len(available_is_instructions)))
        
        # 从可用指令中随机选择指令，不进行替换 (对于当前组合)
        selected_is_instructions = random.sample(available_is_instructions, num_to_combine)

        # 组合字段
        combined_input_parts = []
        combined_output_parts = []
        combined_type_parts = []
        
        for is_instr in selected_is_instructions:
            combined_input_parts.append(is_instr["input"])
            
            # 对于无效的单命令指令，其 output 预期为 '''error_input'''
            output_content = is_instr["output"].strip()
            # 提取 ''' 和 ''' 之间的内容
            if output_content.startswith("'''") and output_content.endswith("'''"):
                output_content = output_content[3:-3].strip()
            
            # 由于所有都是无效指令，我们期待输出内容是 "error_input"
            combined_output_parts.append(output_content if output_content else "error_input") # 如果输出为空，则默认为 error_input
            
            combined_type_parts.append(is_instr["type"])

        # 构造新字段
        new_input = ", ".join(combined_input_parts)
        if not new_input.endswith('.'): # 如果没有以句号结尾，则添加句号
            new_input += '.'
        
        # output 应为 ''' ... ,''' 格式，与 HomeBench 的多命令输出格式保持一致
        new_output = "'''" + ",".join(combined_output_parts) + ",'''"
        
        # 新的 type 应为构成其的子指令 type 的逗号分隔字符串
        new_type = ",".join(combined_type_parts)
        
        # 新的 ID: HomeID_invalid_multi_序列号
        new_id_seq = len(generated_multi_instructions) + 1
        new_id = f"home{selected_home_id}_invalid_multi_{new_id_seq}"

        generated_multi_instructions.append({
            "id": new_id,
            "input": new_input,
            "output": new_output,
            "home_id": selected_home_id,
            "type": new_type
        })
        
        # 如果某个 home_id 的可用指令数量变得太少，将其从 eligible_home_ids 中移除
        # 这确保我们不会从已经用尽的 home_id 中尝试采样。
        if len(invalid_single_instructions_by_home_id[selected_home_id]) - num_to_combine < MIN_COMBINE:
             if selected_home_id in current_eligible_home_ids:
                 current_eligible_home_ids.remove(selected_home_id)

    print(f"Successfully generated {len(generated_multi_instructions)} invalid multi-command instructions.")

    # 步骤 3: 保存新数据集
    print(f"Step 3: Saving generated instructions to {OUTPUT_FILENAME}")
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f_out:
        for instr in generated_multi_instructions:
            f_out.write(json.dumps(instr, ensure_ascii=False) + '\n')
    
    print("Dataset construction completed.")

if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    generate_invalid_multi_instructions()

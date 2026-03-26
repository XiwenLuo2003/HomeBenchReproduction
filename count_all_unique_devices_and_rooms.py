import os
import json

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # 在这种情况下，项目根目录就是脚本的目录
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

HOME_STATUS_METHOD_FILE = os.path.join(DATASET_DIR, "home_status_method.jsonl")

def count_all_unique_devices_and_rooms():
    unique_devices = set() # 使用集合来存储唯一的设备名称
    unique_rooms = set()   # 使用集合来存储唯一的房间名称
    total_homes_processed = 0

    print(f"Starting to count unique devices and rooms from: {HOME_STATUS_METHOD_FILE}")

    if not os.path.exists(HOME_STATUS_METHOD_FILE):
        print(f"Error: File not found: {HOME_STATUS_METHOD_FILE}")
        return

    with open(HOME_STATUS_METHOD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                total_homes_processed += 1
                
                if "home_status" in data:
                    home_status = data["home_status"]
                    for room_or_device_key, content in home_status.items():
                        if room_or_device_key == "VacuumRobot":
                            # VacuumRobot 可能是顶层设备，不视为房间
                            unique_devices.add("vacuum_robot") # 统一小写，保持一致性
                        elif isinstance(content, dict):
                            # 这是一个房间，添加房间名称，并遍历房间内的设备
                            unique_rooms.add(room_or_device_key)
                            for device_name, device_data in content.items():
                                if device_name != "room_name": # 忽略 room_name 字段
                                    unique_devices.add(device_name) # 添加设备名称
                
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON line: {line.strip()}")
                continue
    
    print(f"Processed {total_homes_processed} home environments.\n") # 修复后的行
    print(f"Total number of unique devices found across all homes: {len(unique_devices)}")
    print(f"List of unique devices: {sorted(list(unique_devices))}\n") # 修复后的行
    print(f"Total number of unique room types found across all homes: {len(unique_rooms)}")
    print(f"List of unique room types: {sorted(list(unique_rooms))}")

if __name__ == "__main__":
    count_all_unique_devices_and_rooms()
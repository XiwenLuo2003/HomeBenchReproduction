import os
import json
import re
from functools import lru_cache

# --- 路径设置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) # 项目根目录是脚本目录的上一级
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

HOME_STATUS_METHOD_FILE = os.path.join(DATASET_DIR, "home_status_method.jsonl")
DEFAULT_HOME_ID = 13

# --- 预定义的房间和设备名称映射 (用户指令 -> 数据集中的名称) ---
# 数据集中的房间名称
CANONICAL_ROOM_NAMES = [
    'balcony', 'bathroom', 'corridor', 'ding_room', 'foyer', 'garage',
    'guest_bedroom', 'kitchen', 'living_room', 'master_bedroom',
    'store_room', 'study_room'
]

# 用户指令中可能出现的房间名及其到规范名称的映射
ROOM_NAME_MAP = {
    "balcony": "balcony", "bathroom": "bathroom", "corridor": "corridor",
    "dinig room": "ding_room", "dining room": "ding_room", "foyer": "foyer",
    "garage": "garage", "guest bedroom": "guest_bedroom", "kitchen": "kitchen",
    "living room": "living_room", "master bedroom": "master_bedroom",
    "store room": "store_room", "study room": "study_room",
    "ding room": "ding_room" # 确保直接匹配 'ding room' 也能到 'ding_room'
}

# 数据集中的设备名称
CANONICAL_DEVICE_NAMES = [
    'air_conditioner', 'air_purifiers', 'aromatherapy', 'blinds', 'curtain',
    'dehumidifiers', 'fan', 'garage_door', 'heating', 'humidifier',
    'light', 'media_player', 'trash', 'vacuum_robot', 'water_heater'
]

# 用户指令中可能出现的设备名及其到规范名称的映射
DEVICE_NAME_MAP = {
    "air conditioner": "air_conditioner", "air purifiers": "air_purifiers",
    "aromatherapy": "aromatherapy", "blinds": "blinds", "curtain": "curtain",
    "dehumidifiers": "dehumidifiers", "fan": "fan", "garage door": "garage_door",
    "heating": "heating", "humidifier": "humidifier", "light": "light",
    "media player": "media_player", "trash": "trash", "vacuum robot": "vacuum_robot",
    "water heater": "water_heater",
    "ac": "air_conditioner", # 常见别名
    "lamp": "light", # 常见别名
    "lights": "light", # 常见别名
    "music player": "media_player", # 常见别名
    "robot": "vacuum_robot", # 单独的robot可能指代扫地机器人
    "vacuum": "vacuum_robot" # 单独的vacuum可能指代扫地机器人
}

# --- 数据加载 (使用 LRU 缓存避免重复加载) ---
@lru_cache(maxsize=1) # 缓存最近一次加载的数据
def _load_home_data():
    home_data_map = {}
    if not os.path.exists(HOME_STATUS_METHOD_FILE):
        print(f"Error: Home data file not found: {HOME_STATUS_METHOD_FILE}")
        return {}

    with open(HOME_STATUS_METHOD_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                home_data_map[data["home_id"]] = {"home_status": data["home_status"], "method": data["method"]}
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON line in {HOME_STATUS_METHOD_FILE}: {line.strip()}")
                continue
    return home_data_map

# --- 辅助函数：标准化名称以便匹配 ---
def _normalize_text(text):
    return text.lower().replace(" ", "_")

# --- 提取房间和设备名 ---
def extract_room_and_device(instruction: str):
    extracted_room = None
    extracted_device = None

    instruction_lower = instruction.lower()

    # 优先匹配完整的房间名 (包括多词房间名)
    for user_room_phrase, canonical_room_name in ROOM_NAME_MAP.items():
        # 使用正则表达式匹配整个单词，避免部分匹配
        if re.search(r'\b' + re.escape(user_room_phrase.lower()) + r'\b', instruction_lower):
            extracted_room = canonical_room_name
            break # 找到第一个就停止
    
    # 优先匹配完整的设备名 (包括多词设备名)
    for user_device_phrase, canonical_device_name in DEVICE_NAME_MAP.items():
        if re.search(r'\b' + re.escape(user_device_phrase.lower()) + r'\b', instruction_lower):
            extracted_device = canonical_device_name
            break # 找到第一个就停止

    # 特殊处理 'vacuum robot' 分开的情况
    if not extracted_device and ("vacuum" in instruction_lower and "robot" in instruction_lower):
        extracted_device = "vacuum_robot"

    return extracted_room, extracted_device

# --- 核心环境感知功能 ---
def sense_environment(user_instruction: str, home_id: int = DEFAULT_HOME_ID):
    home_data_map = _load_home_data()

    if home_id not in home_data_map:
        return {"success": False, "message": f"Home ID {home_id} not found in home data.", "details": {}}
    
    current_home_info = home_data_map[home_id]
    home_status = current_home_info["home_status"]

    matched_room_name, matched_device_name = extract_room_and_device(user_instruction)

    details = {
        "extracted_room_from_instruction": matched_room_name,
        "extracted_device_from_instruction": matched_device_name,
        "room_exists_in_home": False,
        "device_exists_in_room": False,
        "all_device_names_in_extracted_room": [], # 修改：只存储设备名称列表
        "specific_device_full_status_and_attributes": {} # 修改：存储特定设备的完整状态和属性
    }

    if not matched_room_name and not matched_device_name:
        details["message"] = "Could not extract a recognizable room or device from the instruction."
        return {"success": False, "message": "Could not parse instruction.", "details": details}

    # --- Step 1: 检查提取的房间是否存在于当前家庭环境，并获取其下的设备名称列表 ---
    target_room_data = None
    if matched_room_name and matched_room_name in home_status:
        details["room_exists_in_home"] = True
        target_room_data = home_status[matched_room_name]
        
        # 提取此房间下所有设备的名称
        for device_key in target_room_data.keys():
            if device_key != "room_name": # 排除 room_name 字段
                details["all_device_names_in_extracted_room"].append(device_key)
    elif matched_room_name: # 房间名被提取，但未在家庭数据中找到
        details["message"] = f"Room '{matched_room_name}' not found in Home ID {home_id}."
        return {"success": False, "message": "Room not found.", "details": details}

    # --- Step 2: 检查提取的设备是否存在 (在房间内或作为顶层 VacuumRobot) ---
    if matched_device_name:
        # Case A: 设备是 VacuumRobot (它是一个顶层设备，不属于任何房间)
        if matched_device_name == "vacuum_robot" and "VacuumRobot" in home_status:
            # 即使指令中提到了房间，但 VacuumRobot 是顶层设备，所以我们直接从顶层获取其状态
            details["device_exists_in_room"] = True # 视为存在 (在家庭层面)
            details["specific_device_full_status_and_attributes"] = home_status["VacuumRobot"]
            details["message"] = "Successfully sensed environment for VacuumRobot (top-level device)."
            details["success"] = True
            return {"success": True, "message": "Sensing complete.", "details": details}

        # Case B: 设备在已识别的房间内
        elif target_room_data and matched_device_name in target_room_data:
            details["device_exists_in_room"] = True
            details["specific_device_full_status_and_attributes"] = target_room_data[matched_device_name]
            details["message"] = "Successfully sensed environment."
            details["success"] = True
            return {"success": True, "message": "Sensing complete.", "details": details}
        
        # Case C: 设备被提取，但在已识别的房间中未找到
        elif target_room_data and matched_device_name not in target_room_data:
            details["message"] = f"Device '{matched_device_name}' not found in room '{matched_room_name}'."
            details["success"] = False
            return {"success": False, "message": "Device not found in room.", "details": details}
        
        # Case D: 设备被提取，但没有房间上下文，且它不是顶层 VacuumRobot
        else:
            details["message"] = f"Device '{matched_device_name}' extracted, but no valid room context or it's not a top-level VacuumRobot."
            details["success"] = False
            return {"success": False, "message": "Device context unclear.", "details": details}
            
    # --- Step 3: 如果只匹配到房间，但没有特定设备 ---
    elif matched_room_name and details["room_exists_in_home"]: # 只有房间名被匹配，没有设备名
        details["message"] = f"Only room '{matched_room_name}' extracted. No specific device found in instruction."
        details["success"] = True # 视为部分成功，因为房间是存在的
        return {"success": True, "message": "Sensing complete (room only).", "details": details}
    
    # --- Fallback: 理论上不应该到达这里，除非初始提取和检查都失败了 ---
    else:
        details["message"] = "An unexpected state occurred during sensing or no relevant entities were found."
        details["success"] = False
        return {"success": False, "message": "Sensing failed.", "details": details}

# --- 交互模式 (如果直接运行脚本) ---
if __name__ == "__main__":
    print("HomeBench Environment Sensing Tool (Interactive Mode)")
    print(f"Default Home ID: {DEFAULT_HOME_ID}. You can call sense_environment(instruction, home_id) directly in other code.")
    print("Type 'exit' to quit.")
    
    while True:
        user_input = input("\nEnter atomic user instruction: ")
        if user_input.lower() == 'exit':
            break
        
        result = sense_environment(user_input, DEFAULT_HOME_ID)
        print("\n--- Sensing Result ---")
        print(json.dumps(result, indent=4, ensure_ascii=False))
        print("----------------------")
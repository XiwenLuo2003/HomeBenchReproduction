import os
import json
import re
import time
from functools import lru_cache

# ==========================================
# Part 1: Core Tool (环境感知工具)
# ==========================================

class HomeBenchPerceptionTool:
    def __init__(self, dataset_dir=None):
        # 1. 路径配置
        if dataset_dir:
            self.dataset_dir = dataset_dir
        else:
            # 默认假设 dataset 在当前脚本上级目录的 dataset 文件夹中
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            self.dataset_dir = os.path.join(project_root, "dataset")
        
        self.data_file = os.path.join(self.dataset_dir, "home_status_method.jsonl")
        self.default_home_id = 13

        # 2. 知识库：房间名称映射
        self.room_name_map = {
            "balcony": "balcony", "bathroom": "bathroom", "corridor": "corridor",
            "dinig room": "ding_room", "dining room": "ding_room", "foyer": "foyer",
            "garage": "garage", "guest bedroom": "guest_bedroom", "kitchen": "kitchen",
            "living room": "living_room", "master bedroom": "master_bedroom",
            "store room": "store_room", "study room": "study_room",
            "ding room": "ding_room",
            # 增强字典，用于处理用户输入的错误拼写
            "dinigroom": "ding_room", "diningroom": "ding_room",
            "guestbedroom": "guest_bedroom",
            "livingroom": "living_room", "masterbedroom": "master_bedroom",
            "storeroom": "store_room", "studyroom": "study_room",
            "dingroom": "ding_room"
        }

        # 3. 知识库：设备名称映射
        self.device_name_map = {
            "air conditioner": "air_conditioner", "air purifiers": "air_purifiers",
            "aromatherapy": "aromatherapy", "blinds": "blinds", "curtain": "curtain",
            "dehumidifiers": "dehumidifiers", "fan": "fan", "garage door": "garage_door",
            "heating": "heating", "humidifier": "humidifier", "light": "light",
            "media player": "media_player", "trash": "trash", "vacuum robot": "vacuum_robot",
            "water heater": "water_heater",
            "ac": "air_conditioner",
            "lamp": "light",
            "lights": "light",
            "music player": "media_player",
            "robot": "vacuum_robot",
            "vacuum": "vacuum_robot"
        }

    @lru_cache(maxsize=1)
    def _load_home_data(self):
        """加载家庭状态数据 (带缓存)"""
        home_data_map = {}
        if not os.path.exists(self.data_file):
            # 为了防止运行报错，如果没有文件，返回一个空的 Mock 数据结构用于演示
            # print(f"Warning: Home data file not found at {self.data_file}. Running in Mock Mode.")
            return {}

        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    home_data_map[data["home_id"]] = {
                        "home_status": data["home_status"], 
                        "method": data["method"]
                    }
                except json.JSONDecodeError:
                    continue
        return home_data_map

    def extract_entities(self, instruction: str):
        """从自然语言指令中提取标准化的房间名和设备名"""
        extracted_room = None
        extracted_device = None
        instruction_lower = instruction.lower()

        # 匹配房间
        for user_room, canonical_room in self.room_name_map.items():
            if re.search(r'\b' + re.escape(user_room) + r'\b', instruction_lower):
                extracted_room = canonical_room
                break
        
        # 匹配设备
        for user_device, canonical_device in self.device_name_map.items():
            if re.search(r'\b' + re.escape(user_device) + r'\b', instruction_lower):
                extracted_device = canonical_device
                break

        # 特殊处理 vacuum robot 分开的情况
        if not extracted_device and ("vacuum" in instruction_lower and "robot" in instruction_lower):
            extracted_device = "vacuum_robot"

        return extracted_room, extracted_device

    def sense(self, user_instruction: str, home_id: int = None):
        """核心感知逻辑"""
        target_home_id = home_id if home_id is not None else self.default_home_id
        home_data_map = self._load_home_data()

        # 构造基础响应结构
        result = {
            "success": False,
            "home_id": target_home_id,
            "instruction": user_instruction,
            "extracted": {"room": None, "device": None},
            "perception": {
                "room_found": False,
                "device_found": False,
                "available_devices": [], # 该房间内所有可用设备名
                "target_device_state": {} # 目标设备的具体状态
            },
            "message": ""
        }

        # 检查数据是否加载
        if not home_data_map:
            result["message"] = "Database Error: Home data not loaded."
            return result
        
        if target_home_id not in home_data_map:
            result["message"] = f"Home ID {target_home_id} not found."
            return result

        current_home_status = home_data_map[target_home_id]["home_status"]
        
        # 1. 实体提取
        room_name, device_name = self.extract_entities(user_instruction)
        result["extracted"]["room"] = room_name
        result["extracted"]["device"] = device_name

        if not room_name and not device_name:
            result["message"] = "Perception Failed: No room or device detected in instruction."
            return result

        # 2. 房间感知
        target_room_data = None
        if room_name and room_name in current_home_status:
            result["perception"]["room_found"] = True
            target_room_data = current_home_status[room_name]
            # 获取房间内设备列表
            result["perception"]["available_devices"] = [k for k in target_room_data.keys() if k != "room_name"]
        elif room_name:
            result["message"] = f"Room '{room_name}' does not exist in this home."
            return result

        # 3. 设备感知
        if device_name:
            # Case A: 全局设备 (如 VacuumRobot)
            if device_name == "vacuum_robot" and "VacuumRobot" in current_home_status:
                result["perception"]["device_found"] = True
                result["perception"]["target_device_state"] = current_home_status["VacuumRobot"]
                result["success"] = True
                result["message"] = "Sensed global device (VacuumRobot)."
                return result
            
            # Case B: 房间内的设备
            if target_room_data:
                if device_name in target_room_data:
                    result["perception"]["device_found"] = True
                    result["perception"]["target_device_state"] = target_room_data[device_name]
                    result["success"] = True
                    result["message"] = f"Sensed {device_name} in {room_name}."
                    return result
                else:
                    result["message"] = f"Device '{device_name}' not found in '{room_name}'."
                    return result
            
            # Case C: 只有设备名，没房间名 (模糊感知)
            # 在这种情况下，我们无法确定具体的物理设备，除非做全屋搜索（暂时不实现，为了保持严谨）
            result["message"] = f"Device '{device_name}' detected, but room context is missing."
            return result

        # 4. 只有房间没有设备 (部分成功)
        if result["perception"]["room_found"]:
            result["success"] = True
            result["message"] = f"Sensed room {room_name}, but no specific device targeted."
            return result

        return result

# ==========================================
# Part 2: The Agent (智能体封装)
# ==========================================

class AgentResponse:
    def __init__(self, raw_input, perception_result, issues):
        self.raw_input = raw_input
        self.result = perception_result
        self.issues = issues
        self.timestamp = time.time()
    
    def to_json(self):
        return json.dumps({
            "agent": "BOP2-Perception",
            "input": self.raw_input,
            "status": "success" if self.result["success"] else "failed",
            "perception_result": self.result,
            "warnings": self.issues
        }, indent=2, ensure_ascii=False)

class EnvironmentPerceptionAgent:
    """
    BOP2: 环境感知智能体
    职责：根据自然语言指令，查询并返回家庭环境的真实状态 (Ground Truth)。
    """
    def __init__(self):
        self.tool = HomeBenchPerceptionTool()
        # print(">>> [System] Perception Agent Initialized.")

    def _self_reflect(self, result):
        """
        自我反思：检查感知结果是否存在异常
        """
        issues = []
        
        extracted = result["extracted"]
        perception = result["perception"]

        # 1. 检查实体提取是否完整
        if not extracted["room"] and not extracted["device"]:
            issues.append("Zero entities extracted. The instruction might be abstract or contain unknown entities.")
        
        # 2. 检查逻辑冲突 (有房间但找不到房间)
        if extracted["room"] and not perception["room_found"]:
            issues.append(f"Hallucination Risk: User mentioned '{extracted['room']}', but it does not exist in this home.")

        # 3. 检查逻辑冲突 (有设备但找不到设备)
        if extracted["device"] and perception["room_found"] and not perception["device_found"]:
            issues.append(f"Hallucination Risk: User mentioned '{extracted['device']}' in '{extracted['room']}', but it is not there.")

        return issues

    def run(self, user_instruction, home_id=13):
        """
        Agent 主入口
        """
        # 1. 感知 (Sense)
        raw_result = self.tool.sense(user_instruction, home_id)
        
        # 2. 反思 (Reflect)
        validation_issues = self._self_reflect(raw_result)
        
        # 3. 响应 (Respond)
        return AgentResponse(user_instruction, raw_result, validation_issues)

# ==========================================
# Part 3: Interaction Loop
# ==========================================
def main():
    agent = EnvironmentPerceptionAgent()
    
    print("="*60)
    print("👁️ 环境感知智能体 (BOP2: Perception Agent)")
    print("功能：提取指令中的实体，并检索家庭数据库中的真实状态。")
    print("提示：确保 'dataset/home_status_method.jsonl' 存在，否则将无法检索数据。")
    print("="*60)

    while True:
        try:
            user_input = input("\n[User Input (Atomic Instruction)]: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                break
            if not user_input: continue
            
            # 允许用户输入 home_id (格式: id:instruction)
            home_id = 13
            if ":" in user_input and user_input.split(":")[0].isdigit():
                parts = user_input.split(":", 1)
                home_id = int(parts[0])
                user_input = parts[1].strip()

            start_time = time.time()
            response = agent.run(user_input, home_id)
            process_time = (time.time() - start_time) * 1000

            print(f"\n[Agent Output] (Home ID: {home_id}, Time: {process_time:.2f}ms):")
            print(response.to_json())

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
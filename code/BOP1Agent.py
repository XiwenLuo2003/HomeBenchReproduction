import re
import json
import time

# ==========================================
# Part 1: Core Tool (基于 V15 的拆分器)
# ==========================================
class InstructionSplitterV15:
    def __init__(self):
        # 1. 核心动词库
        self.known_verbs = {
            'set', 'turn', 'adjust', 'increase', 'decrease', 'open', 'close', 
            'raise', 'lower', 'change', 'switch', 'move', 'configure', 
            'play', 'stop', 'mute', 'pause', 'skip', 'rewind', 'fast_forward',
            'charge', 'dock', 'start', 'activate', 'put', 'initiate', 
            'instruct', 'clean', 'pack', 'vacuum', 'mop', 'sweep', 'empty', 'dispose', 'allow',
            'reduce', 'enhance', 'boost', 'dim', 'brighten', 'heat', 'cool', 'toggle', 'ensure',
            'maximize', 'minimize', 'reset', 'bump', 'fix', 'remove', 'disable', 'enable',
            'significantly', 'slightly', 'further', 'finally', 'fully', 'additionally', 'also',
            'take', 'max'
        }

        # 2. 介词/冠词/数字库
        self.non_verb_starters = {
            'in', 'on', 'at', 'for', 'to', 'by', 'with', 'from', 'under', 'inside', 'outside',
            'the', 'a', 'an', 'my', 'your', 'this', 'that', 'level', 'mode', 'speed', 'temperature', 'intensity'
        }

        # 3. 分割符
        self.split_pattern = re.compile(r'(?:;|\.|,|\b(?:and|then|finally|additionally|also)\b)', re.IGNORECASE)

        # 4. 正则模版
        self.action_regex = re.compile(r'^(.*?)(\s+(?:to|by)\s+)(.*)$', re.IGNORECASE)
        self.location_regex = re.compile(r'^(.*?)(\s+(?:in|on|at|for)\s+)(.*)$', re.IGNORECASE)
        self.prep_start_regex = re.compile(r'^(?:In|On|At|For|Inside)\s+', re.IGNORECASE)
        self.suffix_loc_regex = re.compile(r'(\s+(?:in|on|at|for)\s+.+)$', re.IGNORECASE)
        self.first_prep_split = re.compile(r'\s+(?:in|on|at|for|to|by|with)\s+', re.IGNORECASE)

    def is_full_command(self, text):
        if not text: return False
        words = text.split()
        if not words: return False
        first_word = words[0].lower()
        if first_word in self.known_verbs: return True
        return self._guess_if_verb(words)

    def _guess_if_verb(self, words):
        first = words[0].lower()
        if first in self.non_verb_starters: return False
        if first[0].isdigit(): return False
        if len(words) > 1:
            second = words[1].lower()
            if second in ['the', 'a', 'an', 'my', 'up', 'down', 'on', 'off', 'out', 'vacuum']:
                return True
        return False

    def is_location_prefix(self, text):
        if not self.prep_start_regex.match(text): return False
        if self.is_full_command(text): return False
        if any(char.isdigit() for char in text): return False
        return True

    def analyze_context(self, text):
        context = {
            'text': text,
            'base_verb': text.split(' ')[0], 
            'action_header': None,  
            'location_header': None,
            'verb_obj_header': None 
        }
        split_parts = self.first_prep_split.split(text, 1)
        if split_parts:
            context['verb_obj_header'] = split_parts[0].strip()
        else:
            context['verb_obj_header'] = text
        action_match = self.action_regex.match(text)
        if action_match:
            context['action_header'] = action_match.group(1) + action_match.group(2)
        loc_match = self.location_regex.match(text)
        if loc_match:
            context['location_header'] = loc_match.group(1) + loc_match.group(2)
        else:
            context['location_header'] = text + " in "
        return context

    def reconstruct_fragment(self, fragment, context):
        if not context: return fragment
        fragment_lower = fragment.lower()
        if fragment_lower.startswith('to ') or fragment_lower.startswith('by '):
            if context['action_header']:
                match = self.action_regex.match(context['text'])
                if match: return f"{match.group(1)} {fragment}"
        if fragment[0].isdigit():
            if context['action_header']: return f"{context['action_header']}{fragment}"
        starts_with_loc = self.prep_start_regex.match(fragment)
        has_param = (' to ' in fragment_lower) or any(c.isdigit() for c in fragment)
        if starts_with_loc and has_param:
            if context['verb_obj_header']:
                return f"{context['verb_obj_header']} {fragment}"
        if (' to ' in fragment_lower or ' by ' in fragment_lower) and not (fragment_lower.startswith('to ') or fragment_lower.startswith('by ')):
            if context['base_verb']: return f"{context['base_verb']} {fragment}"
        if context['location_header']: return f"{context['location_header']}{fragment}"
        if context['action_header']: return f"{context['action_header']}{fragment}"
        return fragment

    def _post_process_backward(self, results):
        if len(results) < 2: return results
        for i in range(len(results) - 2, -1, -1):
            current_cmd = results[i]
            next_cmd = results[i+1]
            curr_verb = current_cmd.split(' ')[0].lower()
            next_verb = next_cmd.split(' ')[0].lower()
            if curr_verb != next_verb: continue
            curr_loc_match = self.suffix_loc_regex.search(current_cmd)
            next_loc_match = self.suffix_loc_regex.search(next_cmd)
            if not curr_loc_match and next_loc_match:
                results[i] = current_cmd + next_loc_match.group(1)
        return results

    def _post_process_forward(self, results):
        last_known_location = None
        new_results = []
        for cmd in results:
            prefix_match = re.match(r'^(In|On|At|For|Inside)\s+(.+?)(?:,|$)', cmd, re.IGNORECASE)
            if prefix_match:
                prep = prefix_match.group(1).lower()
                loc = prefix_match.group(2).strip()
                candidate_loc = f" {prep} {loc}"
                if not re.search(r'\b(?:and|then|finally|additionally|also)\s*$', candidate_loc, re.IGNORECASE):
                    last_known_location = candidate_loc
            loc_match = self.suffix_loc_regex.search(cmd)
            has_valid_suffix = False
            if loc_match:
                candidate_loc = loc_match.group(1)
                if re.search(r'\b(?:and|then|finally|additionally|also)\s*$', candidate_loc, re.IGNORECASE):
                    cmd = cmd[:loc_match.start()]
                    if last_known_location:
                        cmd += last_known_location
                else:
                    last_known_location = candidate_loc
                    has_valid_suffix = True
            if not prefix_match and not has_valid_suffix:
                if last_known_location:
                    cmd += last_known_location
            new_results.append(cmd)
        return new_results

    def process(self, raw_input):
        if not raw_input: return []
        chunks = self.split_pattern.split(raw_input.strip())
        results = []
        last_context = None
        pending_prefix = None 
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk: continue
            while True:
                original_chunk = chunk
                chunk = re.sub(r'^\s*(?:and|then|finally|additionally|also|plus|moreover)[,\s]*', '', chunk, flags=re.IGNORECASE)
                chunk = re.sub(r'^\s*in\s+addition[,\s]*', '', chunk, flags=re.IGNORECASE)
                if chunk == original_chunk: break
            if chunk.endswith('.'): chunk = chunk[:-1]
            chunk = chunk.strip()
            if not chunk or re.match(r'^(?:and|then|finally|additionally|also|in addition)$', chunk, re.IGNORECASE):
                continue
            if self.is_location_prefix(chunk):
                pending_prefix = chunk
                continue
            if pending_prefix:
                chunk = f"{pending_prefix}, {chunk}"
                pending_prefix = None
            if self.is_full_command(chunk):
                final_cmd = chunk
                last_context = self.analyze_context(chunk)
            else:
                final_cmd = self.reconstruct_fragment(chunk, last_context)
                last_context = self.analyze_context(final_cmd)
            results.append(final_cmd)
        results = self._post_process_backward(results)
        results = self._post_process_forward(results)
        return results

# ==========================================
# Part 2: The Agent (智能体封装)
# ==========================================
class AgentResponse:
    def __init__(self, raw_input, actions, issues):
        self.raw_input = raw_input
        self.actions = actions
        self.issues = issues # 潜在问题列表
        self.timestamp = time.time()
    
    def to_json(self):
        return json.dumps({
            "status": "success" if not self.issues else "warning",
            "input": self.raw_input,
            "actions": self.actions,
            "warnings": self.issues
        }, indent=2, ensure_ascii=False)

class InstructionSplittingAgent:
    """
    不依赖 LLM 的纯规则基智能体。
    功能：
    1. 调用 Splitter 工具拆分指令。
    2. 执行“自省”（Heuristic Validation）检查拆分结果的质量。
    3. 输出标准化的结果。
    """
    def __init__(self):
        self.splitter = InstructionSplitterV15()
        print(">>> [System] Rule-Based Agent Initialized. Tool: InstructionSplitter V15")

    def _self_reflect(self, actions):
        """
        [智能体核心]：自我反思/校验逻辑
        虽然没有 LLM，我们可以用规则来判断结果是否“看起来”正确。
        """
        issues = []
        
        for idx, action in enumerate(actions):
            # 检查 1: 是否包含动词?
            if not self.splitter.is_full_command(action):
                issues.append(f"Action {idx+1} ('{action}') lacks a clear verb.")
            
            # 检查 2: 是否包含地点? (HomeBench 中绝大多数指令都有地点)
            # 使用 V15 的正则来简单判断
            has_location = bool(self.splitter.suffix_loc_regex.search(action) or \
                                re.match(r'^(In|On|At|For|Inside)', action, re.IGNORECASE))
            if not has_location:
                # 某些全局指令可能没地点，标记为 Warning 而不是 Error
                # 但在 HomeBench 中，这通常意味着上下文丢失
                issues.append(f"Action {idx+1} ('{action}') might be missing a location context.")

            # 检查 3: 长度异常
            if len(action.split()) < 3:
                issues.append(f"Action {idx+1} ('{action}') is suspiciously short.")

        return issues

    def run(self, user_instruction):
        """
        Agent 的主入口
        """
        # 1. 思考 (Think) - 规则模型不需要显式思考，直接调用工具
        
        # 2. 行动 (Act) - 调用拆分器
        raw_actions = self.splitter.process(user_instruction)
        
        # 3. 反思 (Reflect) - 校验结果
        validation_issues = self._self_reflect(raw_actions)
        
        # 4. 响应 (Respond)
        response = AgentResponse(user_instruction, raw_actions, validation_issues)
        return response

# ==========================================
# Part 3: Interaction Loop
# ==========================================
def main():
    agent = InstructionSplittingAgent()
    
    print("="*60)
    print("🤖 指令拆分智能体 (Rule-Based Agent)")
    print("该智能体不使用 LLM，仅依靠 V15 算法和启发式校验逻辑。")
    print("="*60)

    while True:
        try:
            user_input = input("\n[User Input]: ").strip()
            if user_input.lower() in ['exit', 'quit']:
                print("Agent shutting down.")
                break
            if not user_input: continue
            
            # 处理 JSON 粘贴的情况
            try:
                data = json.loads(user_input)
                if isinstance(data, dict) and "input" in data:
                    user_input = data["input"]
            except: pass

            # 智能体运行
            start_time = time.time()
            result = agent.run(user_input)
            process_time = (time.time() - start_time) * 1000

            # 打印结果
            print(f"\n[Agent Output] ({process_time:.2f}ms):")
            print(result.to_json())

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
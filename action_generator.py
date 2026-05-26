import requests
import json
import re
import time
import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# V2 标签等价对集成
try:
    from confusion_matrix.label_config import LABEL_EQUIVALENCE_PAIRS
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    LABEL_EQUIVALENCE_PAIRS = {}

class ActionGenerator:
    def __init__(self, api_key, base_url="http://172.20.168.207/v1/chat/completions", model="deepseek-v3-0324-ygcx", max_rules=20, max_chars_per_rule=200):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_rules = max_rules
        self.max_chars_per_rule = max_chars_per_rule
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def generate_actions(self, current_prompt, error_analysis, confused_labels=None, k=6):
        """
        生成 k 个候选修改动作
        :param current_prompt: 当前提示词
        :param error_analysis: 错误分析报告 (Dict)
        :param confused_labels: 容易混淆的标签列表
        :param k: 生成候选数量
        :return: List[Dict] -> [{'new_prompt': str, 'confidence': float, 'description': str}]
        """
        
        # 1. 构建 Prompt
        # 统计当前提示词的规则条数
        current_rules = self._count_rules(current_prompt)
        can_add = current_rules < self.max_rules
        
        # 构建等价对信息
        equivalence_info = ""
        if V2_AVAILABLE and confused_labels:
            equivalence_pairs_for_label = []
            for lbl in confused_labels:
                if lbl in LABEL_EQUIVALENCE_PAIRS:
                    pairs = LABEL_EQUIVALENCE_PAIRS[lbl]
                    if isinstance(pairs, list):
                        equivalence_pairs_for_label.extend(pairs)
            if equivalence_pairs_for_label:
                equivalence_info = "\\n【标签等价对警告】\\n下列标签对在语义上等价，修改时特别注意区分（否则会相互冲突）:\\n"
                for pair in set(equivalence_pairs_for_label):
                    equivalence_info += f"  - {pair}\\n"
        
        system_prompt = f"""你是一个专家级的提示词优化工程师。你的任务是根据错误分析报告，对当前的提示词进行修改，以提升目标标签的准确率和召回率。
        
【核心目标 - 铁律】
1. **目标标签准确率和召回率必须双双上升**
2. **混淆标签（其他标签）准确率绝对不能下降** - 这是铁律！
3. 如果无法同时满足以上两点，宁可不修改{equivalence_info}

【重要约束】
1. 仅修改目标标签的规则，绝对不要修改其他标签。
2. 每次修改必须针对具体的错误案例（False Negatives 或 False Positives）。
3. 修改时要非常谨慎，确保不会"抢走"其他标签的样本。
4. 请生成 {k} 个不同的修改方案。
5. 动作类型：{'可以使用 ADD (新增规则) 或 MODIFY (修改现有规则)' if can_add else f'只能使用 MODIFY (修改现有规则)，因为已达到{self.max_rules}条上限'}。
6. 每条规则不超过 {self.max_chars_per_rule} 字。
7. 对于每个方案，你必须给出一个 0.0 到 1.0 的置信度评分 (confidence)，表示你认为该方案能解决问题的把握。
8. **严禁修改当前提示词中已有的规则内容**，只能添加新规则或调整新规则的表述。
9. **规则格式规范**：每条规则必须按照"数字.规则内容；"的格式编写，规则编号必须连续递增，每条换行符间隔。
10. 严格遵守 JSON 格式输出。
"""
        
        user_prompt = f"""
【当前提示词】
{current_prompt}

【错误分析】
目标标签: {error_analysis.get('target_label', 'Unknown')}
假阴性 (漏检): {error_analysis.get('false_negatives', 0)} 例
假阳性 (误检): {error_analysis.get('false_positives', 0)} 例

典型漏检案例 (FN):
{self._format_examples(error_analysis.get('fn_examples', []))}

典型误检案例 (FP):
{self._format_examples(error_analysis.get('fp_examples', []))}

【混淆标签信息】
{f"容易与目标标签混淆的标签: {', '.join(confused_labels)}" if confused_labels else "无混淆标签信息"}
{"注意：优化时必须考虑与这些标签的区分度，避免误伤它们的准确率" if confused_labels else ""}

【当前状态】
- 当前规则条数: {current_rules} / {self.max_rules}
- {'可以新增规则' if can_add else '已达上限，只能修改现有规则'}

【任务】
请生成 {k} 个修改方案。输出格式必须为 JSON 列表：
[
  {{
    "action_type": "ADD" 或 "MODIFY",
    "description": "简短描述修改内容 (例如: 增加关于xxx的关键词)",
    "new_prompt": "修改后的完整提示词内容（必须包含标签名称和所有现有规则加上你的修改）",
    "confidence": 0.85
  }},
  ...
]

【重要提醒】
- new_prompt必须是完整的提示词，从"{error_analysis.get('target_label', '')}:"开始
- 必须包含所有现有规则加上你的新增或修改的规则
- **严禁改动当前提示词中已有规则的内容**，只能在末尾添加新规则
- 不要只返回新增或修改的部分
"""
        
        # 2. 调用 API
        response_content = self._call_deepseek([
            {"role": "system", "content": system_prompt.format(k=k)},
            {"role": "user", "content": user_prompt}
        ])
        
        # 3. 解析结果
        actions = self._parse_response(response_content)
        
        # 4. 后处理：确保new_prompt是完整的（修复DeepSeek不遵守指令的问题）
        target_label = error_analysis.get('target_label', '')
        if target_label and actions:
            for action in actions:
                new_prompt = action.get('new_prompt', '').strip()
                # 如果new_prompt不以标签名称开头，说明只返回了规则部分，需要拼接
                if new_prompt and not new_prompt.startswith(f'{target_label}:'):
                    # 检查是否是纯规则内容（以数字开头，如"5.xxx"）
                    if new_prompt and (new_prompt[0].isdigit() or new_prompt.startswith(';')):
                        # 拼接完整prompt：当前prompt + 分号 + 新规则
                        separator = '' if current_prompt.endswith(';') else ';'
                        action['new_prompt'] = current_prompt + separator + new_prompt
                        logger.warning(f"⚠️ DeepSeek返回不完整prompt，已自动拼接完整: {action['new_prompt'][:100]}...")
                    else:
                        # 其他情况也尝试拼接
                        separator = '' if current_prompt.endswith(';') else ';'
                        action['new_prompt'] = current_prompt + separator + new_prompt
                        logger.warning(f"⚠️ DeepSeek返回不完整prompt，已自动拼接完整: {action['new_prompt'][:100]}...")
                else:
                    # 5. 验证：确保已有规则没有被修改
                    if new_prompt and new_prompt.startswith(f'{target_label}:'):
                        # 检查当前prompt是否被完整保留在new_prompt中
                        # 简单验证：检查当前prompt的前50个字符是否在new_prompt中
                        current_start = current_prompt[:min(50, len(current_prompt))]
                        if current_start not in new_prompt:
                            logger.warning(f"⚠️ 警告：DeepSeek可能修改了已有规则，当前prompt开头未在new_prompt中找到")
                            logger.warning(f"   当前prompt开头: {current_start}")
                            logger.warning(f"   new_prompt开头: {new_prompt[:100]}")
        
        # 6. 归一化置信度 (Softmax)
        if actions:
            confidences = [a['confidence'] for a in actions]
            probs = self._softmax(confidences)
            for i, action in enumerate(actions):
                action['confidence'] = probs[i]
                
        return actions

    def _format_examples(self, examples):
        return "\n".join([f"- {ex}" for ex in examples[:5]])

    def _call_deepseek(self, messages, max_retries=3):
        for attempt in range(max_retries):
            try:
                data = {
                    "messages": messages,
                    "model": self.model,
                    "temperature": 0.8,  # 提高温度增加多样性，适应高迭代策略
                    # 注意：不设置 response_format=json_object，因为 Prompt 要求返回 JSON 数组
                    # json_object 格式会强制模型返回单个对象，与 Prompt 预期的数组格式冲突
                }
                
                response = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content']
                else:
                    logger.error(f"API Error (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text[:300]}")
            except Exception as e:
                logger.error(f"Request Failed (attempt {attempt+1}/{max_retries}): {e}")
            
            if attempt < max_retries - 1:
                wait_secs = 3 * (attempt + 1)
                logger.info(f"等待 {wait_secs}s 后重试...")
                time.sleep(wait_secs)
        
        logger.error("DeepSeek API 调用连续失败，已达最大重试次数")
        return ""

    def _parse_response(self, content):
        try:
            # 尝试直接解析 JSON
            # 有时候模型会包裹在 ```json ... ``` 中
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            data = json.loads(content)
            
            # 处理不同的 JSON 结构
            if isinstance(data, list):
                # 检查元素是否为字典
                if data and isinstance(data[0], dict):
                    return data
                else:
                    logger.warning(f"Parsed list but elements are not dicts: {type(data[0])}")
                    return []
            elif isinstance(data, dict):
                # 情况1：如果有 'actions' 键，返回其值
                if 'actions' in data:
                    actions = data['actions']
                    if isinstance(actions, list):
                        return actions
                    return [actions] if isinstance(actions, dict) else []
                
                # 情况2：如果字典本身就是一个 action（有 action_type、new_prompt 等键）
                if 'new_prompt' in data and ('action_type' in data or 'description' in data):
                    return [data]
                
                # 情况3：如果是 {'data': [...]} 格式
                if 'data' in data and isinstance(data['data'], list):
                    return data['data']
                
                # 情况4：如果整个 dict 是多个 action 的包装
                logger.warning(f"Warning: Dict structure not recognized. Keys: {list(data.keys())}")
                logger.warning(f"Dict content (first 200 chars): {str(data)[:200]}")
                return []
            else:
                logger.warning(f"Unexpected data type: {type(data)}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error: {e}")
            logger.error(f"Raw content (first 500 chars): {content[:500]}")
            # 尝试提取 JSON 部分
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    if isinstance(data, list) and data and isinstance(data[0], dict):
                        return data
                except:
                    pass
            return []
        except Exception as e:
            logger.error(f"Unexpected error in _parse_response: {e}")
            logger.error(f"Content: {str(content)[:200]}")
            return []

    def _count_rules(self, prompt):
        """统计提示词中的规则条数（简单实现：按行或按编号统计）"""
        # 假设规则是按行分隔的，或者有编号如 1. 2. 3.
        lines = [line.strip() for line in prompt.split('\n') if line.strip()]
        # 更精确的统计可以识别编号格式
        numbered_rules = [line for line in lines if re.match(r'^\d+[\.、]', line)]
        return len(numbered_rules) if numbered_rules else len(lines)
    
    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return (e_x / e_x.sum()).tolist()

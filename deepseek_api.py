import time
import requests
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class DeepSeekClient:
    """DeepSeek API客户端"""
    
    def __init__(self, 
                 api_key: str = "sk-gzhFbRu8rikjl4N606EcB46974264f098246D1EeE59eCc20",
                 base_url: str = "http://172.20.168.207/v1/chat/completions",
                 model: str = "deepseek-v3-0324-ygcx"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.request_count = 0
        self.start_time = time.time()
    
    def call_api(self, messages: List[Dict], max_tokens: int = 1000, temperature: float = 0.1):
        """调用DeepSeek API"""
        self.request_count += 1
        
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"]
            else:
                logger.error(f"API调用失败: {response.status_code}, {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"API调用异常: {str(e)}")
            return None
    
    def get_stats(self):
        """获取API调用统计"""
        elapsed_time = time.time() - self.start_time
        return {
            'total_requests': self.request_count,
            'elapsed_time': elapsed_time,
            'requests_per_minute': self.request_count / (elapsed_time / 60) if elapsed_time > 0 else 0
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DeepSeekTest")
    
    client = DeepSeekClient()
    messages = [
        {"role": "user", "content": "请输出《铜雀台赋》全文。"}
    ]
    
    print("正在调用 DeepSeek API...")
    result = client.call_api(messages)
    
    if result:
        print("\n=== API 返回结果 ===")
        print(result)
        print("\n=== 调用统计 ===")
        stats = client.get_stats()
        print(f"总请求次数: {stats['total_requests']}")
        print(f"耗时: {stats['elapsed_time']:.2f}秒")
    else:
        print("API 调用失败")

#  conda activate q3_1 && python3 /ai/hlf/gongdan3_0/mcts_prompt_gen_v4_multi_ltx/deepseek_api.py

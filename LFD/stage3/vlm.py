import requests
import numpy as np
import re


class VLM:
    """视觉语言模型接口"""
    
    def __init__(self, 
                 base_url="https://yinli.one/v1",
                 api_key="sk-bjKmXhPPpWsKR88WQIQJoipchZ92JQrK4p5IL3CelGuMaEgQ",
                 model="gpt-4o-mini",
                 n_experts=7):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.n_experts = n_experts
      
        self.experts = ['stand', 'walk', 'run', 'crawl', 'carry', 'reach_one', 'reach_two']
        if n_experts == 8:
            self.experts = ['stand', 'walk', 'run', 'sit', 'crawl', 'carry', 'reach', 'stair']
    
    def get_weights(self, task_desc, env_info=None, verbose=False):
        """获取专家权重"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"VLM分析过程")
            print(f"{'='*60}")
            print(f"任务描述: {task_desc}")
            if env_info:
                print(f"环境状态: {env_info}")
        
     
        for attempt in range(3):
            try:
                if verbose:
                    print(f"\n调用VLM API...")
                
                response = self._call_api(task_desc, env_info)
                
                if verbose:
                    print(f"VLM响应: {response}")
                
                weights = self._parse(response)
                if weights is not None:
                    if verbose:
                        print(f"\n专家权重:")
                        for i, (name, w) in enumerate(zip(self.experts, weights)):
                            bar = '█' * int(w)
                            print(f"  {name:12s}: {w:4.1f} {bar}")
                        print(f"\n→ 推荐: {self.experts[weights.argmax()]}")
                    return weights
            except Exception as e:
                if verbose:
                    print(f"尝试{attempt+1}失败: {e}")
                pass
        
      
        if verbose:
            print(f"\nAPI失败，使用规则方法...")
        
        weights = self._rule_based(task_desc)
        
        if verbose:
            print(f"\n专家权重:")
            for i, (name, w) in enumerate(zip(self.experts, weights)):
                bar = '█' * int(w)
                print(f"  {name:12s}: {w:4.1f} {bar}")
            print(f"\n→ 推荐: {self.experts[weights.argmax()]}")
        
        return weights
    
    def _call_api(self, task_desc, env_info=None):
        """调用API"""
        if self.n_experts == 7:
            prompt = f"""你是机器人任务分析专家。我有7个运动专家：

1. stand - 站立保持
2. walk - 正常行走  
3. run - 快速跑步
4. crawl - 爬行
5. carry - 搬运物体
6. reach_one - 单手伸手抓取（开门、按钮等精细操作）
7. reach_two - 双手伸手抓取（搬运大物体）

任务："{task_desc}"
{f'当前环境状态：{env_info}' if env_info else ''}

根据任务和当前状态，为每个专家打分(0-10)，表示重要性。

只返回7个数字，逗号分隔：
[stand, walk, run, crawl, carry, reach_one, reach_two]

示例：
"走到门前开门" → [1.0, 7.0, 0.5, 0.1, 0.5, 9.0, 2.0]  (walk走近 + reach_one开门)
"搬运箱子" → [2.0, 5.0, 0.5, 0.1, 8.0, 2.0, 3.0]  (walk+carry为主)"""
        else:
            prompt = f"""你是机器人任务分析专家。我有8个运动专家：

1. stand - 站立保持
2. walk - 正常行走
3. run - 快速跑步
4. sit - 坐下动作
5. crawl - 爬行
6. carry - 搬运物体
7. reach - 伸手抓取
8. stair - 爬楼梯

任务："{task_desc}"

为每个专家打分(0-10)

只返回8个数字，逗号分隔：
[专家1, 专家2, 专家3, 专家4, 专家5, 专家6, 专家7, 专家8]

示例：
"走到门前开门" → [1.0, 8.0, 0.5, 0.2, 0.1, 1.0, 7.5, 0.3]
"坐下休息" → [2.0, 4.0, 0.5, 9.0, 0.2, 0.5, 1.0, 0.3]"""
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "你是机器人专家。"},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 100
            },
            timeout=20
        )
        return resp.json()['choices'][0]['message']['content']
    
    def _parse(self, text):
        """解析权重"""
        numbers = re.findall(r'[\d.]+', text)
        if len(numbers) >= self.n_experts:
            weights = np.array([float(n) for n in numbers[:self.n_experts]], dtype=np.float32)
            return np.clip(weights, 0, 10)
        return None
    
    def _rule_based(self, task_desc):
        """规则方案"""
        weights = np.zeros(self.n_experts, dtype=np.float32)
        desc = task_desc.lower()
        
        if self.n_experts == 7:
          
            keywords = {
                'stand': 0, '站': 0, '立': 0,
                'walk': 1, '走': 1, 'move': 1, '移动': 1,
                'run': 2, '跑': 2, 'fast': 2, '快': 2,
                'crawl': 3, '爬行': 3,
                'carry': 4, '搬': 4, '运': 4,
                'reach': 5, 'grab': 5, '抓': 5, '取': 5, 
                'open': 5, '开': 5, 'door': 5, '门': 5,   
                'hand': 6, '手': 6, 'two': 6,  # reach_two
            }
        else:
           
            keywords = {
                'stand': 0, '站': 0,
                'walk': 1, '走': 1, 'move': 1, '移动': 1,
                'run': 2, '跑': 2, 'fast': 2, '快': 2,
                'sit': 3, '坐': 3,
                'crawl': 4, '爬行': 4,
                'carry': 5, '搬': 5, '运': 5,
                'reach': 6, 'grab': 6, '抓': 6, '取': 6,
                'open': 6, '开': 6, 'door': 6, '门': 6,
                'stair': 7, '楼梯': 7, 'climb': 7, '爬': 7,
            }
        
        for kw, idx in keywords.items():
            if kw in desc:
                weights[idx] += 5.0
        
        if weights.sum() == 0:
            weights[1] = 5.0  
        
        weights += 0.5
        return np.clip(weights, 0, 10)


if __name__ == "__main__":
    vlm = VLM()
    
    tasks = [
        "走到门前开门",
        "坐下休息",
        "快速跑向终点",
    ]
    
    for task in tasks:
        w = vlm.get_weights(task, verbose=True)
        print(f"Top: {vlm.experts[w.argmax()]}\n")

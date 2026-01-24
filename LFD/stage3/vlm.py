import requests
import numpy as np
import re


class VLM:
    """Vision Language Model Interface"""
    
    def __init__(self, 
                 base_url="https://yinli.one/v1",
                 api_key="YOUR_API_KEY",
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
        """Get expert weights"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"VLM Analysis Process")
            print(f"{'='*60}")
            print(f"Task description: {task_desc}")
            if env_info:
                print(f"Environment state: {env_info}")
        
        for attempt in range(3):
            try:
                if verbose:
                    print(f"\nCalling VLM API...")
                
                response = self._call_api(task_desc, env_info)
                
                if verbose:
                    print(f"VLM response: {response}")
                
                weights = self._parse(response)
                if weights is not None:
                    if verbose:
                        print(f"\nExpert weights:")
                        for i, (name, w) in enumerate(zip(self.experts, weights)):
                            bar = '█' * int(w)
                            print(f"  {name:12s}: {w:4.1f} {bar}")
                        print(f"\n→ Recommendation: {self.experts[weights.argmax()]}")
                    return weights
            except Exception as e:
                if verbose:
                    print(f"Attempt {attempt+1} failed: {e}")
                pass
        
        if verbose:
            print(f"\nAPI failed, using rule-based method...")
        
        weights = self._rule_based(task_desc)
        
        if verbose:
            print(f"\nExpert weights:")
            for i, (name, w) in enumerate(zip(self.experts, weights)):
                bar = '█' * int(w)
                print(f"  {name:12s}: {w:4.1f} {bar}")
            print(f"\n→ Recommendation: {self.experts[weights.argmax()]}")
        
        return weights
    
    def _call_api(self, task_desc, env_info=None):
        """Call API"""
        if self.n_experts == 7:
            prompt = f"""You are a robot task analysis expert. I have 7 motion experts:

1. stand - standing maintenance
2. walk - normal walking  
3. run - fast running
4. crawl - crawling
5. carry - carrying objects
6. reach_one - single-hand reaching and grabbing (fine operations like opening doors, pressing buttons, etc.)
7. reach_two - two-hand reaching and grabbing (carrying large objects)

Task: "{task_desc}"
{f'Current environment state: {env_info}' if env_info else ''}

Based on the task and current state, score each expert (0-10) to indicate importance.

Return only 7 numbers, comma-separated:
[stand, walk, run, crawl, carry, reach_one, reach_two]

Example:
"Walk to the door and open it" → [1.0, 7.0, 0.5, 0.1, 0.5, 9.0, 2.0]  (walk to approach + reach_one to open)
"Carry a box" → [2.0, 5.0, 0.5, 0.1, 8.0, 2.0, 3.0]  (walk+carry as main)"""
        else:
            prompt = f"""You are a robot task analysis expert. I have 8 motion experts:

1. stand - standing maintenance
2. walk - normal walking
3. run - fast running
4. sit - sitting action
5. crawl - crawling
6. carry - carrying objects
7. reach - reaching and grabbing
8. stair - climbing stairs

Task: "{task_desc}"

Score each expert (0-10)

Return only 8 numbers, comma-separated:
[expert1, expert2, expert3, expert4, expert5, expert6, expert7, expert8]

Example:
"Walk to the door and open it" → [1.0, 8.0, 0.5, 0.2, 0.1, 1.0, 7.5, 0.3]
"Sit down and rest" → [2.0, 4.0, 0.5, 9.0, 0.2, 0.5, 1.0, 0.3]"""
        
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a robot expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 100
            },
            timeout=20
        )
        return resp.json()['choices'][0]['message']['content']
    
    def _parse(self, text):
        """Parse weights"""
        numbers = re.findall(r'[\d.]+', text)
        if len(numbers) >= self.n_experts:
            weights = np.array([float(n) for n in numbers[:self.n_experts]], dtype=np.float32)
            return np.clip(weights, 0, 10)
        return None
    
    def _rule_based(self, task_desc):
        """Rule-based approach"""
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
        "Walk to the door and open it",
        "Sit down and rest",
        "Run quickly to the finish line",
    ]
    
    for task in tasks:
        w = vlm.get_weights(task, verbose=True)
        print(f"Top: {vlm.experts[w.argmax()]}\n")

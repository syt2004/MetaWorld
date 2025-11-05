import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tdmpc2'))

from tdmpc2.trainer.online_trainer import OnlineTrainer
from LFD.stage2.video_recorder import VideoRecorder
import torch
import torch.nn.functional as F


class ExpertTrainer(OnlineTrainer):
    """基础专家训练器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
  
        video_dir = os.path.join(self.cfg.work_dir, 'videos')
        self.recorder = VideoRecorder(video_dir, fps=15)
        
        print(f"视频目录: {video_dir}")
    
    def eval(self):
        # 评估时也显示引导信息
        if self.guidance_weight > 0 and self.base_experts is not None:
            print(f"\n[评估] 当前使用专家引导（引导权重={self.guidance_weight}）")
      
        ep_rewards, ep_successes = [], []
        
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
            # 显示评估开始时的引导
            if i == 0 and self.guidance_weight > 0:
                _ = self.get_expert_guidance(obs, step_num=0)
         
            if i == 0:
                self.recorder.start()
            
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                obs, reward, done, truncated, info = self.env.step(action)
                done = done or truncated
                ep_reward += reward
                t += 1
                
            
                if i == 0:
                    self.recorder.record(self.env)
            
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get("success", 0))
            
           
            if i == 0:
                self.recorder.stop()
                self.recorder.save(self._step)
                self.recorder.save_comparison(self._step)
        
        return dict(
            episode_reward=np.nanmean(ep_rewards),
            episode_success=np.nanmean(ep_successes),
        )


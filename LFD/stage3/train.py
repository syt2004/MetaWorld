import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tdmpc2'))

import LFD.stage2

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import hydra
from termcolor import colored
from tensordict import TensorDict

from tdmpc2.common.seed import set_seed
from tdmpc2.common.buffer import Buffer
from tdmpc2.agent import TDMPC2
from tdmpc2.common.logger import Logger
from LFD.stage3.experts import Experts
from LFD.stage3.vlm import VLM

torch.backends.cudnn.benchmark = True


class ComplexTaskTrainer:
    """Complex Task Training with VLM Guidance"""
    
    def __init__(self, cfg, env, agent, buffer, logger, base_experts, vlm):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        
        # Base expert guidance
        self.base_experts = base_experts
        self.vlm = vlm
        self.guidance_weight = 0.05
        
        self._step = 0
        self._ep_idx = 0
        
       
        from LFD.stage2.video_recorder import VideoRecorder
        video_dir = os.path.join(cfg.work_dir, 'videos')
        self.recorder = VideoRecorder(video_dir, fps=15)
    
    def get_expert_reference(self, obs):
        """Get reference expert action based on current state"""
        try:
            robot_x = float(obs[0])
            obs_len = obs.shape[0]
            
            
            if obs_len > 51:
                if robot_x < 0.3:
                    expert = 'walk'
                else:
                    expert = 'reach_one'
            else:
                expert = 'walk'
            
         
            if self._step % 1000 == 0:
                print(f"  [Guidance] Step {self._step}: X={robot_x:.2f}m -> Reference {expert}")
            
            # Get expert action
            obs_input = obs[:51] if obs_len > 51 else obs
            if expert == 'reach_one' and obs_len >= 55:
                obs_input = obs[:55]
            
            action = self.base_experts.act(expert, obs_input)
            return torch.from_numpy(action).float().to(self.agent.device)
        
        except:
            return None
    
    def train(self):
        """VLM Guidance"""
        from tensordict import TensorDict
        
        print(f"\nStarting VLM-guided training:")
        print(f"  Base experts: {self.base_experts.names}")
        print(f"  Guidance weight: {self.guidance_weight}")
        print(f"  Total steps: {self.cfg.steps}\n")
        
        train_metrics, done, eval_next = {}, True, True
        _tds = []
        
        while self._step <= self.cfg.steps:
            
            # Regular evaluation
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
            
            # Reset environment
            if done:
                if eval_next:
                    eval_metrics = self.eval()
                    self.logger.log(eval_metrics, "eval")
                    eval_next = False
                
                if self._step > 0:
                    ep_reward = sum([td["reward"] for td in _tds[1:]])
                    train_metrics['episode_reward'] = float(ep_reward)
                    train_metrics['step'] = self._step
                    self.logger.log(train_metrics, "train")
                    self.buffer.add(torch.cat(_tds))
                
                obs = self.env.reset()[0]
                _tds = [self.to_td(obs)]
            
            # Collect data
            if self._step > self.cfg.seed_steps:
                action = self.agent.act(obs, t0=len(_tds) == 1)
                
                # Get expert guidance
                if self.base_experts is not None:
                    self._expert_ref = self.get_expert_reference(obs)
            else:
                action = self.env.rand_act()
            
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            _tds.append(self.to_td(obs, action, reward))
            
            # Update model
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretraining...")
                else:
                    num_updates = 1
                
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)
            
            self._step += 1
    
    def to_td(self, obs, action=None, reward=None):
        
        if isinstance(obs, dict):
            obs = TensorDict(obs, batch_size=(), device="cpu")
        else:
            obs = obs.unsqueeze(0).cpu()
        
        if action is None:
            action = torch.full_like(self.env.rand_act(), float("nan"))
        if reward is None:
            reward = torch.tensor(float("nan"))
        
        return TensorDict(
            dict(obs=obs, action=action.unsqueeze(0), reward=reward.unsqueeze(0)),
            batch_size=(1,),
        )
    
    def eval(self):
        """Evaluate"""
        ep_rewards, ep_successes = [], []
        
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
         
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
            
         
            if i == 0:
                self.recorder.stop()
                self.recorder.save(self._step)
                self.recorder.save_comparison(self._step)
            
            ep_rewards.append(ep_reward)
            ep_successes.append(info.get("success", 0))
        
        return {
            'episode_reward': np.mean(ep_rewards),
            'episode_success': np.mean(ep_successes),
            'step': self._step
        }


@hydra.main(config_name="expert_config", config_path="../stage2", version_base=None)
def train(cfg: dict):
    """VLM Guidance
    
    Usage: python train_complex.py task=h1-door-v0 steps=1000000
    """
    assert cfg.steps > 0
    
    from pathlib import Path
    cfg.work_dir = Path.cwd() / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.multitask = False
    cfg.tasks = [cfg.task]
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    
    set_seed(cfg.seed)
    
    print(colored("VLM-guided Training", "green", attrs=["bold"]))
    print(colored(f"Task: {cfg.task}", "yellow"))
    print(colored(f"Directory: {cfg.work_dir}", "yellow"))
    
    
    print("\nLoading base experts...")
    check_path = os.path.join(PROJECT_ROOT, 'check')
    base_experts = Experts(check_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    vlm = VLM(n_experts=len(base_experts))
    
   
   
    print(colored(f"Loaded {len(base_experts)} base experts", "green"))
    

    import gymnasium as gym
    env = gym.make(cfg.task)
    
    from tdmpc2.envs.wrappers.tensor import TensorWrapper
    env = TensorWrapper(env)
    
    try:
        episode_length = env.get_wrapper_attr('_max_episode_steps')
    except:
        episode_length = 1000  
    
    cfg.obs_shape = {'state': env.observation_space.shape}
    cfg.action_dim = env.action_space.shape[0]
    cfg.episode_length = episode_length
    cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    cfg.multitask = False
    cfg.task_dim = 0

    trainer = ComplexTaskTrainer(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
        base_experts=base_experts,
        vlm=vlm
    )
    
    trainer.train()
    print(colored("\nTraining completed", "green", attrs=["bold"]))


if __name__ == "__main__":
    train()

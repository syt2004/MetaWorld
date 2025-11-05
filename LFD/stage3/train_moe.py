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
from LFD.stage3.moe import Router, MixtureOfExperts

torch.backends.cudnn.benchmark = True


class MoETrainer:
    
    def __init__(self, cfg, env, agent, buffer, logger, moe_policy, vlm=None):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.moe = moe_policy
        self.vlm = vlm
        
        self._step = 0
        self._ep_idx = 0
        
        self.router_optim = torch.optim.Adam(
            self.moe.router.parameters(),
            lr=cfg.get('router_lr', 3e-4)
        )
        
        self.guidance_weight = cfg.get('guidance_weight', 0.3)
        self.vlm_weight = cfg.get('vlm_weight', 1.0)
        self.vlm_decay = cfg.get('vlm_decay', 0.99)
        
        from LFD.stage2.video_recorder import VideoRecorder
        video_dir = os.path.join(cfg.work_dir, 'videos')
        self.recorder = VideoRecorder(video_dir, fps=15)
    
    def train(self):
        print(f"\n{'='*60}")
        print(f"MOE")
        print(f"{'='*60}")
        print(f"任务: {self.cfg.task}")
        print(f"专家: {self.moe.experts.names}")
        print(f"引导权重: {self.guidance_weight:.2f}")
        print(f"VLM weight: {self.vlm_weight:.2f} (decay={self.vlm_decay})")
        print(f"Steps: {self.cfg.steps}\n")
        
        train_metrics, done, eval_next = {}, True, True
        _tds = []
        
        while self._step <= self.cfg.steps:
            
            if self._step % self.cfg.eval_freq == 0:
                eval_next = True
            
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
            
            if self._step > self.cfg.seed_steps:
                agent_action = self.agent.act(obs, t0=len(_tds) == 1)
                
                moe_action, weights = self.moe.act(
                    obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs,
                    return_weights=True
                )
                
                agent_action_cpu = agent_action.cpu() if isinstance(agent_action, torch.Tensor) else agent_action
                action = (1 - self.guidance_weight) * agent_action_cpu.numpy() + \
                         self.guidance_weight * moe_action
                action = torch.from_numpy(action).float()
                
                if self._step % 1000 == 0:
                    idx = weights.argmax()
                    print(f"  [{self._step}] {self.moe.experts.names[idx]} ({weights[idx]:.2f}) vlm={self.vlm_weight:.3f}")
            else:
                action = self.env.rand_act()
            
            obs, reward, done, truncated, info = self.env.step(action)
            done = done or truncated
            _tds.append(self.to_td(obs, action, reward))
            
            if self._step >= self.cfg.seed_steps:
                if self._step == self.cfg.seed_steps:
                    num_updates = self.cfg.seed_steps
                    print("Pretrain agent...")
                else:
                    num_updates = 1
                
                for _ in range(num_updates):
                    _train_metrics = self.agent.update(self.buffer)
                train_metrics.update(_train_metrics)
            
            if self._step > self.cfg.seed_steps and self._step % 10 == 0:
                router_loss = self.update_router()
                train_metrics['router_loss'] = router_loss
            
            self._step += 1
    
    def get_vlm_guidance(self, obs):
        if self.vlm is None:
            return None
        
        task_desc = self.cfg.task.replace('-', ' ')
        env_info = None
        if isinstance(obs, torch.Tensor):
            obs_np = obs.cpu().numpy()
            if obs_np.ndim == 2:
                obs_np = obs_np[0]
            env_info = f"robot_x={obs_np[0]:.2f}"
        
        vlm_weights = self.vlm.get_weights(task_desc, env_info=env_info, verbose=False)
        return torch.from_numpy(vlm_weights).float().to(self.moe.device)
    
    def update_router(self):
        batch = self.buffer.sample()
        obs = batch['obs']
        
        if isinstance(obs, dict):
            obs = obs['state']
        
        self.moe.train_mode()
        router_w = self.moe.router(obs)
        
        if self.vlm is not None and self.vlm_weight > 0.01:
            vlm_target = self.get_vlm_guidance(obs[0])
            vlm_target = vlm_target.unsqueeze(0).expand(obs.shape[0], -1)
            vlm_target = torch.softmax(vlm_target, dim=-1)
            
            kl_loss = torch.nn.functional.kl_div(
                torch.log(router_w + 1e-8),
                vlm_target,
                reduction='batchmean'
            )
            
            entropy = -(router_w * torch.log(router_w + 1e-8)).sum(dim=-1).mean()
            loss = self.vlm_weight * kl_loss - 0.1 * entropy
            
            self.vlm_weight *= self.vlm_decay
        else:
            entropy = -(router_w * torch.log(router_w + 1e-8)).sum(dim=-1).mean()
            loss = -entropy
        
        self.router_optim.zero_grad()
        loss.backward()
        self.router_optim.step()
        self.moe.eval_mode()
        
        return loss.item()
    
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
        ep_rewards, ep_successes = [], []
        
        for i in range(self.cfg.eval_episodes):
            obs, done, ep_reward, t = self.env.reset()[0], False, 0, 0
            
            if i == 0:
                self.recorder.start()
                print(f"\n[Eval {self._step}]")
            
            while not done:
                action = self.agent.act(obs, t0=t == 0, eval_mode=True)
                
                if self.guidance_weight > 0:
                    moe_action = self.moe.act(
                        obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
                    )
                    action_cpu = action.cpu() if isinstance(action, torch.Tensor) else action
                    action = (1 - self.guidance_weight) * action_cpu.numpy() + \
                             self.guidance_weight * moe_action
                    action = torch.from_numpy(action).float()
                
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
    """MOE 训练入口
    
    用法: python train_moe.py task=h1-door-v0 steps=1000000
    """
    assert cfg.steps > 0
    
    from pathlib import Path
    cfg.work_dir = Path.cwd() / "logs" / cfg.task / str(cfg.seed) / "moe"
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.multitask = False
    cfg.tasks = [cfg.task]
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    
    set_seed(cfg.seed)
    
    print(colored("MOE 训练", "green", attrs=["bold"]))
    print(colored(f"任务: {cfg.task}", "yellow"))
    print(colored(f"目录: {cfg.work_dir}", "yellow"))
    
    # 加载基础专家
    print("\n加载基础专家...")
    check_path = os.path.join(PROJECT_ROOT, 'check')
    base_experts = Experts(check_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(colored(f"已加载 {len(base_experts)} 个专家", "green"))
    
    # 加载 VLM
    from LFD.stage3.vlm import VLM
    vlm = VLM(n_experts=len(base_experts))
    print(colored(f"已加载 VLM (作为 Router 的教师)", "green"))
    
    # 创建环境
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
    
    # 创建路由器
    obs_dim = env.observation_space.shape[0]
    router = Router(
        obs_dim=obs_dim,
        n_experts=len(base_experts),
        hidden_dim=256
    )
    
    # 创建 MOE
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moe_policy = MixtureOfExperts(base_experts, router, device=device)
    
    # 创建训练器
    trainer = MoETrainer(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
        moe_policy=moe_policy,
        vlm=vlm
    )
    
    trainer.train()
    
    # 保存路由器
    save_path = cfg.work_dir / "router.pt"
    torch.save({
        'router': router.state_dict(),
        'experts': base_experts.names,
        'obs_dim': obs_dim,
        'cfg': cfg
    }, save_path)
    
    print(colored(f"\n训练完成", "green", attrs=["bold"]))
    print(colored(f"路由器已保存: {save_path}", "yellow"))


if __name__ == "__main__":
    train()


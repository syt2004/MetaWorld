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
import argparse
from termcolor import colored

from LFD.stage3.experts import Experts
from LFD.stage3.moe import Router, MixtureOfExperts


def eval_moe(router_path, task, n_episodes=10, hard_routing=False):
    
    print(f"Loading router: {router_path}")
    ckpt = torch.load(router_path)
    
    obs_dim = ckpt['obs_dim']
    expert_names = ckpt['experts']
    n_experts = len(expert_names)
    
    router = Router(obs_dim, n_experts, hidden_dim=256)
    router.load_state_dict(ckpt['router'])
    
    check_path = os.path.join(PROJECT_ROOT, 'check')
    base_experts = Experts(check_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    moe = MixtureOfExperts(base_experts, router, device=device)
    
    import gymnasium as gym
    env = gym.make(task)
    
    print(f"\n{'='*60}")
    print(f"Evaluating MoE")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Experts: {expert_names}")
    print(f"Hard routing: {hard_routing}")
    print(f"Episodes: {n_episodes}\n")
    
    rewards = []
    successes = []
    
    for ep in range(n_episodes):
        obs, done = env.reset()[0], False
        ep_reward = 0
        step = 0
        
        expert_usage = {name: 0 for name in expert_names}
        
        while not done:
            action, weights = moe.act(obs, hard_routing=hard_routing, return_weights=True)
            
            idx = weights.argmax()
            expert_usage[expert_names[idx]] += 1
            
            obs, reward, done, truncated, info = env.step(action)
            done = done or truncated
            ep_reward += reward
            step += 1
        
        rewards.append(ep_reward)
        successes.append(info.get("success", 0))
        
        print(f"Ep {ep+1}/{n_episodes}:")
        print(f"  reward={ep_reward:.2f} success={info.get('success', 0)} steps={step}")
        print(f"  experts: {expert_usage}")
    
    print(f"\n{'='*60}")
    print(f"Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Avg success: {np.mean(successes):.2f}")
    print(f"{'='*60}\n")
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--router', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--hard', action='store_true')
    
    args = parser.parse_args()
    eval_moe(args.router, args.task, args.episodes, args.hard)


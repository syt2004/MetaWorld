import torch
import torch.nn as nn
import numpy as np


class Router(nn.Module):
    
    def __init__(self, obs_dim, n_experts, hidden_dim=256):
        super().__init__()
        self.n_experts = n_experts
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, n_experts)
        
        self.temp = 1.0
    
    def forward(self, obs, hard=False):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        
        if hard:
            return torch.nn.functional.gumbel_softmax(logits, tau=self.temp, hard=True)
        return torch.softmax(logits, dim=-1)


class MixtureOfExperts:
    
    def __init__(self, base_experts, router, device='cuda'):
        self.experts = base_experts
        self.router = router
        self.device = device
        
        self.router.to(device)
        self.router.eval()
    
    def act(self, obs, hard=False, return_weights=False):
        obs_t = self._to_tensor(obs)
        batched = obs_t.dim() == 2
        
        if not batched:
            obs_t = obs_t.unsqueeze(0)
        
        with torch.no_grad():
            w = self.router(obs_t, hard=hard)
            actions = self._gather_actions(obs_t)
            mixed = self._mix(actions, w)
        
        out = mixed.squeeze(0).cpu().numpy() if not batched else mixed.cpu().numpy()
        
        if return_weights:
            w_out = w.squeeze(0).cpu().numpy() if not batched else w.cpu().numpy()
            return out, w_out
        return out
    
    def _to_tensor(self, obs):
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float().to(self.device)
        return obs.to(self.device)
    
    def _gather_actions(self, obs_t):
        bs = obs_t.shape[0]
        
        configs = {
            'stand': (51, 19), 'walk': (51, 19), 'run': (51, 19),
            'crawl': (151, 61), 'carry': (51, 19),
            'reach_one': (55, 19), 'reach_two': (61, 19),
        }
        
        target_dim = 19
        all_actions = []
        
        for name in self.experts.names:
            obs_dim, act_dim = configs.get(name, (51, 19))
            batch_acts = []
            
            for i in range(bs):
                obs = obs_t[i].cpu().numpy()
                
                # match expected obs dim
                if len(obs) < obs_dim:
                    padded = np.zeros(obs_dim, dtype=np.float32)
                    padded[:len(obs)] = obs
                    obs = padded
                elif len(obs) > obs_dim:
                    obs = obs[:obs_dim]
                
                try:
                    act = self.experts.act(name, obs)
                    if isinstance(act, np.ndarray):
                        act = torch.from_numpy(act).float().to(self.device)
                    
                    # match target action dim
                    if act.shape[0] < target_dim:
                        padded = torch.zeros(target_dim, device=self.device)
                        padded[:act.shape[0]] = act
                        act = padded
                    elif act.shape[0] > target_dim:
                        act = act[:target_dim]
                    
                    batch_acts.append(act)
                except:
                    batch_acts.append(torch.zeros(target_dim, device=self.device))
            
            all_actions.append(torch.stack(batch_acts, dim=0))
        
        return torch.stack(all_actions, dim=1)
    
    def _mix(self, actions, weights):
        w = weights.unsqueeze(-1)
        return (actions * w).sum(dim=1)
    
    def train_mode(self):
        self.router.train()
    
    def eval_mode(self):
        self.router.eval()


import os
import sys
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../tdmpc2'))
from tdmpc2.agent import TDMPC2


class SimpleMLP(nn.Module):
    
    def __init__(self, state_dict):
        super().__init__()
    
        input_dim = state_dict['dense1.weight'].shape[1]
        hidden_dim = state_dict['dense1.weight'].shape[0]
        output_dim = state_dict['dense3.weight'].shape[0]
        
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, output_dim)
        
        
        self.load_state_dict(state_dict)
        self.eval()
    
    def forward(self, x):
        x = torch.relu(self.dense1(x))
        x = torch.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class Experts:
    """专家管理器"""
    
    def __init__(self, ckpt_dir, device='cuda'):
        self.device = device
        self.agents = {}
        self.names = []
        self._load(ckpt_dir)
    
    def _load(self, base_dir):
        
      
        is_check_dir = os.path.basename(base_dir) == 'check' or 'check' in base_dir
        
        if is_check_dir:
            print(f"从check目录加载专家...")
            
            tasks = [
                ('stand', 'stand'),
                ('walk', 'walk'),
                ('run', 'run'),
                ('crawl', 'crawl'),
                ('carry', 'carry'),
                ('reach_one', 'reach_one_hand'),   
                ('reach_two', 'reach_two_hand'),
            ]
            
            print(f"  基础路径: {os.path.abspath(base_dir)}")
            
            for name, folder in tasks:
                folder_path = os.path.join(base_dir, folder)
                print(f"  尝试: {name} <- {folder_path}")
                
                if not os.path.exists(folder_path):
                    print(f"    [SKIP] 不存在")
                    continue
                
                pt_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
                if not pt_files:
                    print(f"    [SKIP] 无.pt文件")
                    continue
                
                print(f"    [FOUND] {pt_files[0]}")
                
                ckpt = os.path.join(folder_path, pt_files[0])
                
                try:
                    data = torch.load(ckpt, map_location=self.device)
                    
                    
                    if 'dense1.weight' in data or (isinstance(data, dict) and 'dense1.weight' in data.get('model', {})):
                    
                        state_dict = data if 'dense1.weight' in data else data['model']
                        model = SimpleMLP(state_dict).to(self.device)
                        
                    
                        class SimpleAgent:
                            def __init__(self, model, device):
                                self.model = model
                                self.device = device
                            def act(self, obs, t0=True, eval_mode=True):
                                with torch.no_grad():
                                    if isinstance(obs, np.ndarray):
                                        obs = torch.from_numpy(obs).float().to(self.device)
                                    return self.model(obs).cpu().numpy()
                        
                        agent = SimpleAgent(model, self.device)
                    else:
                        # TD-MPC2模型
                        cfg = self._create_default_cfg(task_name=name)
                        agent = TDMPC2(cfg)
                        agent.model.load_state_dict(data['model'])
                    
                    self.agents[name] = agent
                    self.names.append(name)
                    print(f"[OK] {name}")
                except Exception as e:
                    print(f"[FAIL] {name}: {e}")
        else:
            
            tasks = [
                ('stand', 'h1-stand-tracking-v0'),
                ('walk', 'h1-walk-tracking-v0'),
                ('run', 'h1-run-tracking-v0'),
                ('sit', 'h1-sit-tracking-v0'),
                ('crawl', 'h1-crawl-tracking-v0'),
                ('carry', 'h1-carry-tracking-v0'),
                ('reach', 'h1-reach-tracking-v0'),
                ('stair', 'h1-stair-tracking-v0'),
            ]
            
            for name, task in tasks:
                ckpt = os.path.join(base_dir, task, '1/expert/models/model.pt')
                if not os.path.exists(ckpt):
                    continue
                
                try:
                    data = torch.load(ckpt, map_location=self.device)
                    agent = TDMPC2(data['cfg'])
                    agent.model.load_state_dict(data['model'])
                
                    
                    self.agents[name] = agent
                    self.names.append(name)
                    print(f"[OK] {name}")
                except Exception as e:
                    print(f"[FAIL] {name}: {e}")
        
        print(f"\n已加载 {len(self.agents)} 个专家\n")
    
    def _create_default_cfg(self, task_name='walk'):
       
        from types import SimpleNamespace
        
        cfg = SimpleNamespace()
        
      
        obs_dims = {
            'stand': 51, 'walk': 51, 'run': 51,
            'crawl': 151,  
            'carry': 51, 
            'reach_one': 51, 'reach_two': 51,
        }
        action_dims = {
            'stand': 19, 'walk': 19, 'run': 19,
            'crawl': 61,  
            'carry': 19, 
            'reach_one': 19, 'reach_two': 19,
        }
        obs_dim = obs_dims.get(task_name, 51)
        action_dim = action_dims.get(task_name, 19)
        
        cfg.obs_shape = {'state': (obs_dim,)}
        cfg.action_dim = action_dim
        cfg.episode_length = 1000
        cfg.obs = 'state'  
        cfg.action_dims = {0: action_dim} 
        cfg.obs_shapes = {0: {'state': (obs_dim,)}} 
        cfg.episode_lengths = {0: 1000}
        
        # 架构参数
        cfg.latent_dim = 512
        cfg.mlp_dim = 512
        cfg.num_enc_layers = 2
        cfg.enc_dim = 256
        cfg.num_bins = 101
        cfg.vmin = -10
        cfg.vmax = 10
        cfg.task_dim = 0
        cfg.multitask = False
        cfg.simnorm_dim = 8
        cfg.num_channels = 32
        cfg.num_q = 5
        cfg.dropout = 0.01
        cfg.log_std_min = -10
        cfg.log_std_max = 2
        
        # 训练参数
        cfg.lr = 3e-4
        cfg.enc_lr_scale = 0.3
        cfg.grad_clip_norm = 20
        cfg.tau = 0.01
        cfg.reward_coef = 0.1
        cfg.value_coef = 0.1
        cfg.consistency_coef = 20
        cfg.rho = 0.5
        cfg.discount_denom = 5
        cfg.discount_min = 0.95
        cfg.discount_max = 0.995
        cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
        
        # MPC参数
        cfg.mpc = True
        cfg.iterations = 6
        cfg.num_samples = 512
        cfg.num_elites = 64
        cfg.num_pi_trajs = 24
        cfg.horizon = 3
        cfg.min_std = 0.05
        cfg.max_std = 2
        cfg.temperature = 0.5
        cfg.entropy_coef = 1e-4
        
        return cfg
    
    def act(self, name, obs):
        """获取动作"""
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().to(self.device)
        
        with torch.no_grad():
            action = self.agents[name].act(obs, t0=True, eval_mode=True)
        
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action
    
    def act_by_id(self, idx, obs):
        """根据ID获取动作"""
        return self.act(self.names[idx], obs)
    
    def __len__(self):
        return len(self.agents)


if __name__ == "__main__":
    experts = Experts('LFD/stage2/logs')
    
    if len(experts) > 0:
        obs = np.random.randn(52)
        action = experts.act('walk', obs)
        print(f"Action shape: {action.shape}")

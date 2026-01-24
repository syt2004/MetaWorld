import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'tdmpc2'))

# Register tasks
import LFD.stage2

# if sys.platform == "linux":
#     os.environ["MUJOCO_GL"] = "egl"
# elif sys.platform == "win32":
#     os.environ["MUJOCO_GL"] = "glfw"

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings
warnings.filterwarnings("ignore")

import torch
import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.common.buffer import Buffer
from tdmpc2.envs import make_env
from tdmpc2.agent import TDMPC2
from tdmpc2.common.logger import Logger


from LFD.stage2.trainer import ExpertTrainer

torch.backends.cudnn.benchmark = True


@hydra.main(config_name="expert_config", config_path=".", version_base=None)
def train(cfg: dict):
    """Train base experts (tracking tasks)
    
    Usage: python train_expert.py task=h1-walk-tracking-v0 steps=1000000
    """
    assert cfg.steps > 0
    

    from pathlib import Path
    cfg.work_dir = Path.cwd() / "logs" / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.multitask = False
    cfg.tasks = [cfg.task]
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)
    
    set_seed(cfg.seed)
    
    print(colored("Training base experts", "green", attrs=["bold"]))
    print(colored(f"Task: {cfg.task}", "yellow"))
    print(colored(f"Directory: {cfg.work_dir}", "yellow"))
    
 
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
    
    # Create trainer
    trainer = ExpertTrainer(
        cfg=cfg,
        env=env,
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    
    trainer.train()
    print(colored("\nTraining completed", "green", attrs=["bold"]))
    print(colored(f"Videos: {cfg.work_dir}/videos/", "yellow"))


if __name__ == "__main__":
    train()


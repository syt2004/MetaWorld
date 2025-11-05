import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class StandTracking(BaseTracking):
    """站立跟踪任务"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_stand.npz'
    w_tracking = 0.8  # stand更注重姿态跟踪
    w_task = 0.2
    
    frame_skip = 10
    max_episode_steps = 500  # stand任务更短
    
    def __init__(self, robot, env, **kwargs):
        super().__init__(robot, env, **kwargs)
    
    @property
    def observation_space(self):
        return Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.robot.dof * 2 - 1),),
            dtype=np.float64,
        )
    
    def get_obs(self):
        """观察空间"""
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))
    
    def get_task_reward(self):
        """保持直立和稳定"""
        # 保持直立
        r_upright = self.robot.torso_upright()
        
        # 头部高度
        head_h = self.robot.head_height()
        r_height = np.exp(-3.0 * abs(head_h - 1.65))
        
        # 稳定性（速度小）
        vel = np.linalg.norm(self.robot.body_velocity())
        r_stable = np.exp(-vel)
        
        total = 0.4 * r_upright + 0.3 * r_height + 0.3 * r_stable
        
        info = {
            'r_upright': r_upright,
            'r_height': r_height,
            'r_stable': r_stable,
        }
        
        return total, info
    
    def get_terminated(self):
        """终止条件"""
        # 摔倒
        if self.robot.head_height() < 0.5:
            return True, {'fall': True}
        
        # 翻倒
        if abs(self.robot.torso_upright()) < 0.5:
            return True, {'flip': True}
        
        return False, {}


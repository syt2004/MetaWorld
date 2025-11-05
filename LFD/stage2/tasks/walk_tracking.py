import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class WalkTracking(BaseTracking):
    """行走跟踪任务"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_walk.npz'
    w_tracking = 0.6   # 降低tracking，增加task（强调方向）
    w_task = 0.4       # 提高task权重，强调向前
    
    frame_skip = 10
    max_episode_steps = 1000
    
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
        
        # 前进速度
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x / 1.5, 0, 2.0)  
        
        # 保持直立
        r_upright = self.robot.torso_upright()
        
        # 保持直线
        vel_y = abs(self.robot.body_velocity()[1])
        r_straight = np.exp(-3.0 * vel_y)
        
        
        total = 0.7 * r_forward + 0.2 * r_upright + 0.1 * r_straight
        
        info = {
            'r_forward': r_forward,
            'r_upright': r_upright,
            'r_straight': r_straight,
        }
        
        return total, info
    
    def get_terminated(self):
        """终止条件"""
        # 摔倒
        if self.robot.head_height() < 0.5:
            return True, {'fall': True}
        
        # 翻倒
        if abs(self.robot.torso_upright()) < 0.3:
            return True, {'flip': True}
        
        return False, {}


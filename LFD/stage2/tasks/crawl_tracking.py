import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class CrawlTracking(BaseTracking):
    """爬行跟踪任务"""
    
    qpos0_robot = {
        "h1": "0 0 0.5 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_crawl.npz'
    w_tracking = 0.8  # 爬行更注重姿态跟踪
    w_task = 0.2
    
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
        """任务特定奖励：低姿态 + 前进"""
        # 前进速度（爬行速度较慢）
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x, 0, 1.0) / 1.0
        
        # 保持低姿态
        head_h = self.robot.head_height()
        r_low = np.exp(-5.0 * max(0, head_h - 0.6))  # 头部应该低于0.6m
        
        # 躯干倾斜（爬行时躯干应该接近水平）
        torso_upright = self.robot.torso_upright()
        r_horizontal = np.exp(-3.0 * abs(torso_upright - 0.3))  # 接近水平
        
        # 保持直线
        vel_y = abs(self.robot.body_velocity()[1])
        r_straight = np.exp(-3.0 * vel_y)
        
        total = 0.3 * r_forward + 0.3 * r_low + 0.2 * r_horizontal + 0.2 * r_straight
        
        info = {
            'r_forward': r_forward,
            'r_low': r_low,
            'r_horizontal': r_horizontal,
            'r_straight': r_straight,
        }
        
        return total, info
    
    def get_terminated(self):
        """终止条件"""
        # 翻倒（完全侧翻或倒立）
        if abs(self.robot.torso_upright()) > 0.9:
            return True, {'flip': True}
        
        # 离开地面太高（不应该站起来）
        if self.robot.head_height() > 1.0:
            return True, {'stand_up': True}
        
        return False, {}


import numpy as np
from LFD.stage2.base_tracking import BaseTracking


class SitTracking(BaseTracking):
    """Sit tracking task"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_sit.npz'
    w_tracking = 0.7
    w_task = 0.3
    
    frame_skip = 10
    max_episode_steps = 1000
    
    def __init__(self, robot, env, **kwargs):
        super().__init__(robot, env, **kwargs)
    
    @property
    def observation_space(self):
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))
    
    def get_obs(self):
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))
    
    def get_task_reward(self):
        # 保持低高度
        r_low = np.exp(-3.0 * max(0, self.robot.head_height() - 0.8))
        
        # 保持稳定
        r_stable = self.robot.torso_upright()
        
        total = 0.6 * r_low + 0.4 * r_stable
        
        info = {
            'r_low': r_low,
            'r_stable': r_stable,
        }
        
        return total, info
    
    def get_terminated(self):
        if self.robot.head_height() < 0.3:
            return True, {}
        return False, {}


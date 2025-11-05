import numpy as np
from LFD.stage2.base_tracking import BaseTracking


class StairTracking(BaseTracking):
    """Stair tracking task"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_stair.npz'
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
        # 向前移动
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x / 1.0, 0, 2.0)
        
        # 向上移动
        vel_z = self.robot.body_velocity()[2]
        r_upward = np.clip(vel_z / 0.5, 0, 2.0)
        
        # 保持直立
        r_upright = self.robot.torso_upright()
        
        total = 0.5 * r_forward + 0.3 * r_upward + 0.2 * r_upright
        
        info = {
            'r_forward': r_forward,
            'r_upward': r_upward,
            'r_upright': r_upright,
        }
        
        return total, info
    
    def get_terminated(self):
        if self.robot.head_height() < 0.5:
            return True, {}
        return False, {}


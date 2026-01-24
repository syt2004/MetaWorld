import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class WalkTracking(BaseTracking):
    """Walking tracking task"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_walk.npz'
    w_tracking = 0.6   # Reduce tracking, increase task (emphasize direction)
    w_task = 0.4       # Increase task weight, emphasize forward movement
    
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
        """Observation space"""
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))
    
    def get_task_reward(self):
        
        # Forward speed
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x / 1.5, 0, 2.0)  
        
        # Stay upright
        r_upright = self.robot.torso_upright()
        
        # Stay straight
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
        """Termination conditions"""
        # Fall down
        if self.robot.head_height() < 0.5:
            return True, {'fall': True}
        
        # Flip over
        if abs(self.robot.torso_upright()) < 0.3:
            return True, {'flip': True}
        
        return False, {}


import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class RunTracking(BaseTracking):
    """Running tracking task"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_run.npz'
    w_tracking = 0.7   # Reduce tracking weight, give more space to task
    w_task = 0.3       # Increase task weight, emphasize forward movement
    
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
        """Task reward"""
        # Forward speed reward
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x / 1.5, 0, 2.0)  # Encourage reaching 1.5m/s, maximum 2x reward
        
        # Stay upright
        r_upright = self.robot.torso_upright()
        
        # Don't fall down
        head_h = self.robot.head_height()
        r_height = float(head_h > 0.5)
        
        # Forward movement
        total = 0.7 * r_forward + 0.2 * r_upright + 0.1 * r_height
        
        info = {
            'r_forward': r_forward,
            'r_upright': r_upright,
            'r_height': r_height,
        }
        
        return total, info
    
    def get_terminated(self):
        """Termination conditions"""
        # Fall down
        if self.robot.head_height() < 0.3:  
            return True, {'fall': True}
        # Flip over
        if abs(self.robot.torso_upright()) < 0.1:  
            return True, {'flip': True}
        return False, {}


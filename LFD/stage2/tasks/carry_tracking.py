import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class CarryTracking(BaseTracking):
    """Carrying tracking task"""
    
    qpos0_robot = {
        "h1": "0 0 0.98 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }
    
    traj_file = 'h1_carry.npz'
    w_tracking = 0.8
    w_task = 0.2
    
    frame_skip = 10
    max_episode_steps = 500
    
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
        """Task reward: carrying posture"""
        # Upper body upright
        r_upright = self.robot.torso_upright()
        
        # Hands position (hands should be in front when carrying)
        left_hand_h = self.robot.left_hand_position()[2]
        right_hand_h = self.robot.right_hand_position()[2]
        hands_h = (left_hand_h + right_hand_h) / 2
        r_hands = np.exp(-2.0 * abs(hands_h - 1.0))
        
        # Slow movement
        vel_x = abs(self.robot.body_velocity()[0])
        r_slow = np.exp(-abs(vel_x - 0.5))
        
        total = 0.5 * r_upright + 0.3 * r_hands + 0.2 * r_slow
        
        info = {
            'r_upright': r_upright,
            'r_hands': r_hands,
            'r_slow': r_slow,
        }
        
        return total, info
    
    def get_terminated(self):
        """Termination conditions"""
        # Fall down
        if self.robot.head_height() < 0.5:
            return True, {'fall': True}
        # Flip over
        if abs(self.robot.torso_upright()) < 0.4:
            return True, {'flip': True}
        return False, {}


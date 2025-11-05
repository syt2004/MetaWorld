import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class CarryTracking(BaseTracking):
    """搬运跟踪任务"""
    
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
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))
    
    def get_task_reward(self):
        """任务奖励：携带姿态"""
        # 上身直立
        r_upright = self.robot.torso_upright()
        
        # 双手位置（携带物体时手应该在前方）
        left_hand_h = self.robot.left_hand_position()[2]
        right_hand_h = self.robot.right_hand_position()[2]
        hands_h = (left_hand_h + right_hand_h) / 2
        r_hands = np.exp(-2.0 * abs(hands_h - 1.0))
        
        # 缓慢移动
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
        if self.robot.head_height() < 0.5:
            return True, {'fall': True}
        if abs(self.robot.torso_upright()) < 0.4:
            return True, {'flip': True}
        return False, {}


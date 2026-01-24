import numpy as np
from gymnasium.spaces import Box

from LFD.stage2.base_tracking import BaseTracking


class CrawlTracking(BaseTracking):
    """Crawling tracking task"""

    qpos0_robot = {
        "h1": "0 0 0.5 1 0 0 0 0 0 -0.4 0.8 -0.4 0 0 -0.4 0.8 -0.4 0 0 0 0 0 0 0 0 0",
    }

    traj_file = 'h1_crawl.npz'
    w_tracking = 0.8  # Crawling focuses more on pose tracking
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
        """Observation space"""
        position = self._env.data.qpos.flat.copy()[:self.robot.dof]
        velocity = self._env.data.qvel.flat.copy()[:self.robot.dof - 1]
        return np.concatenate((position, velocity))

    def get_task_reward(self):
        """Task-specific reward: low posture + forward movement"""
        # Forward speed (crawling speed is slower)
        vel_x = self.robot.body_velocity()[0]
        r_forward = np.clip(vel_x, 0, 1.0) / 1.0

        # Maintain low posture
        head_h = self.robot.head_height()
        r_low = np.exp(-5.0 * max(0, head_h - 0.6))  # Head should be below 0.6m

        # Torso tilt (torso should be close to horizontal when crawling)
        torso_upright = self.robot.torso_upright()
        r_horizontal = np.exp(-3.0 * abs(torso_upright - 0.3))  # Close to horizontal

        # Stay straight
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
        """Termination conditions"""
        # Flip over (complete side flip or handstand)
        if abs(self.robot.torso_upright()) > 0.9:
            return True, {'flip': True}

        # Too high off the ground (should not stand up)
        if self.robot.head_height() > 1.0:
            return True, {'stand_up': True}

        return False, {}


import os
import numpy as np
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from humanoid_bench.tasks import Task
from LFD.stage2 import rewards


class BaseTracking(Task):
    """Base class for tracking tasks
    
    Uses AMASS reference trajectories to guide robot learning movements
    """
    
  
    traj_file = None
    w_tracking = 0.7
    w_task = 0.3
    
    def __init__(self, robot, env, **kwargs):
        super().__init__(robot, env, **kwargs)
        
        if env is None:
            return
        
        # Load reference trajectory
        self.ref_traj = self._load_traj()
        self.traj_len = len(self.ref_traj['dof'])
        
        # Current time step
        self.t = 0
        self.prev_action = None
        
        print(f"Loading reference trajectory: {self.traj_file}")
        print(f"  Frames: {self.traj_len}, Duration: {self.traj_len/50:.1f}s")
    
    def _load_traj(self):
        """Load reference trajectory"""
        traj_path = os.path.join(
            PROJECT_ROOT, 
            'LFD/data', 
            self.traj_file
        )
        
        # if not os.path.exists(traj_path):
        #     raise FileNotFoundError(f"Trajectory not found: {traj_path}")
        
        return np.load(traj_path)
    
    def get_ref_frame(self):
        """Get reference frame at current time"""
        # Loop through trajectory
        idx = self.t % self.traj_len
        
        return {
            'dof': self.ref_traj['dof'][idx],
            'root_pos': self.ref_traj['root_pos'][idx],
        }
    
    def compute_tracking_reward(self):
        """Compute tracking reward"""
        ref = self.get_ref_frame()
        
        # Current state
        q_cur = self.robot.joint_angles()
        q_ref = ref['dof']
        
        # Velocity 
        vel_cur = self.robot.joint_velocities()
        vel_ref = self._compute_ref_velocity()
        
      
        r_track, info = rewards.tracking_reward_simple(
            q_cur, q_ref, 
            vel_cur, vel_ref
        )
        
        return r_track, info
    
    def _compute_ref_velocity(self):
        """Compute reference velocity from trajectory"""
        if self.t == 0:
            # First frame, use zero velocity
            return np.zeros_like(self.robot.joint_velocities())
        
        # Get current and previous frame joint angles
        idx_curr = self.t % self.traj_len
        idx_prev = (self.t - 1) % self.traj_len
        
       
        if idx_prev >= idx_curr and self.t > 0:
            # Crossed loop boundary, use zero velocity
            return np.zeros_like(self.robot.joint_velocities())
        
        q_curr = self.ref_traj['dof'][idx_curr]
        q_prev = self.ref_traj['dof'][idx_prev]
        
        # Compute velocity (angle difference / time difference)
        dt = 1.0 / 50.0  # 50Hz sampling rate
        vel_ref = (q_curr - q_prev) / dt
        
        return vel_ref
    
    def get_task_reward(self):
       
        return 0.0, {}
    
    def get_reward(self):
        """Total reward = tracking reward + task reward"""
        # Tracking reward
        r_track, track_info = self.compute_tracking_reward()
        
        # Task reward
        r_task, task_info = self.get_task_reward()
        
        # Combine
        total = self.w_tracking * r_track + self.w_task * r_task
        
        info = {
            'r_tracking': r_track,
            'r_task': r_task,
            **track_info,
            **task_info,
        }
        
        return total, info
    
    def step(self, action):
        """Execute action"""
        # Record action (for smoothness)
        self.prev_action = action.copy()
        
        # Execute
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Update time
        self.t += 1
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        """Reset"""
        self.t = 0
        self.prev_action = None
        return super().reset_model()


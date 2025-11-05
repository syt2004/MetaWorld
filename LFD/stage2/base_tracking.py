import os
import numpy as np
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from humanoid_bench.tasks import Task
from LFD.stage2 import rewards


class BaseTracking(Task):
    """跟踪任务基类
    
    使用AMASS参考轨迹引导机器人学习运动
    """
    
  
    traj_file = None
    w_tracking = 0.7
    w_task = 0.3
    
    def __init__(self, robot, env, **kwargs):
        super().__init__(robot, env, **kwargs)
        
        if env is None:
            return
        
        # 加载参考轨迹
        self.ref_traj = self._load_traj()
        self.traj_len = len(self.ref_traj['dof'])
        
        # 当前时间步
        self.t = 0
        self.prev_action = None
        
        print(f"加载参考轨迹: {self.traj_file}")
        print(f"  帧数: {self.traj_len}, 时长: {self.traj_len/50:.1f}s")
    
    def _load_traj(self):
        """加载参考轨迹"""
        traj_path = os.path.join(
            PROJECT_ROOT, 
            'LFD/data', 
            self.traj_file
        )
        
        # if not os.path.exists(traj_path):
        #     raise FileNotFoundError(f"找不到轨迹: {traj_path}")
        
        return np.load(traj_path)
    
    def get_ref_frame(self):
        """获取当前时刻的参考帧"""
        # 循环使用轨迹
        idx = self.t % self.traj_len
        
        return {
            'dof': self.ref_traj['dof'][idx],
            'root_pos': self.ref_traj['root_pos'][idx],
        }
    
    def compute_tracking_reward(self):
        """计算跟踪奖励"""
        ref = self.get_ref_frame()
        
        # 当前状态
        q_cur = self.robot.joint_angles()
        q_ref = ref['dof']
        
        # 速度 
        vel_cur = self.robot.joint_velocities()
        vel_ref = self._compute_ref_velocity()
        
      
        r_track, info = rewards.tracking_reward_simple(
            q_cur, q_ref, 
            vel_cur, vel_ref
        )
        
        return r_track, info
    
    def _compute_ref_velocity(self):
        """从参考轨迹计算速度参考"""
        if self.t == 0:
            # 第一帧，使用零速度
            return np.zeros_like(self.robot.joint_velocities())
        
        # 获取当前和前一帧的关节角度
        idx_curr = self.t % self.traj_len
        idx_prev = (self.t - 1) % self.traj_len
        
       
        if idx_prev >= idx_curr and self.t > 0:
            # 跨越循环边界，使用前一次的速度或零速度
            return np.zeros_like(self.robot.joint_velocities())
        
        q_curr = self.ref_traj['dof'][idx_curr]
        q_prev = self.ref_traj['dof'][idx_prev]
        
        # 计算速度 (角度差 / 时间差)
        dt = 1.0 / 50.0  # 50Hz采样率
        vel_ref = (q_curr - q_prev) / dt
        
        return vel_ref
    
    def get_task_reward(self):
       
        return 0.0, {}
    
    def get_reward(self):
        """总奖励 = 跟踪奖励 + 任务奖励"""
        # 跟踪奖励
        r_track, track_info = self.compute_tracking_reward()
        
        # 任务奖励
        r_task, task_info = self.get_task_reward()
        
        # 组合
        total = self.w_tracking * r_track + self.w_task * r_task
        
        info = {
            'r_tracking': r_track,
            'r_task': r_task,
            **track_info,
            **task_info,
        }
        
        return total, info
    
    def step(self, action):
        """执行动作"""
        # 记录动作（用于平滑度）
        self.prev_action = action.copy()
        
        # 执行
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 更新时间
        self.t += 1
        
        return obs, reward, terminated, truncated, info
    
    def reset_model(self):
        """重置"""
        self.t = 0
        self.prev_action = None
        return super().reset_model()


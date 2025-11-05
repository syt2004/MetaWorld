import numpy as np


def pose_reward(q_cur, q_ref, scale=2.0):
    """关节角度匹配奖励
    
    使用指数函数，误差越小奖励越高
    scale越大，对误差越敏感
    """
    error = np.linalg.norm(q_cur - q_ref)
    return np.exp(-scale * error)


def ee_reward(pos_cur, pos_ref, scale=5.0):
    """末端位置匹配奖励"""
    error = np.linalg.norm(pos_cur - pos_ref)
    return np.exp(-scale * error ** 2)


def velocity_reward(vel_cur, vel_ref, scale=0.1):
    """速度匹配奖励"""
    error = np.linalg.norm(vel_cur - vel_ref)
    return np.exp(-scale * error)


def com_reward(com_cur, com_ref, scale=3.0):
    """质心位置匹配奖励"""
    # 只比较水平位置
    error_xy = np.linalg.norm(com_cur[:2] - com_ref[:2])
    # 高度容忍度更大
    error_z = abs(com_cur[2] - com_ref[2])
    error = error_xy + 0.3 * error_z
    return np.exp(-scale * error)


def tracking_reward(q_cur, q_ref, ee_cur, ee_ref, vel_cur, vel_ref, 
                   w_pose=0.4, w_ee=0.3, w_vel=0.3):
    """组合跟踪奖励
    
    参数:
        w_pose: 姿态权重
        w_ee: 末端位置权重
        w_vel: 速度权重
    """
    r_pose = pose_reward(q_cur, q_ref)
    r_ee = ee_reward(ee_cur, ee_ref)
    r_vel = velocity_reward(vel_cur, vel_ref)
    
    total = w_pose * r_pose + w_ee * r_ee + w_vel * r_vel
    
    return total, {
        'r_pose': r_pose,
        'r_ee': r_ee,
        'r_vel': r_vel,
        'r_total': total,
    }


def tracking_reward_simple(q_cur, q_ref, vel_cur, vel_ref, 
                          w_pose=0.7, w_vel=0.3):
    """跟踪奖励 
    
    参数:
        w_pose: 姿态权重（主要）
        w_vel: 速度权重（辅助）
    
  
    姿态是核心，权重70%
    速度作为辅助，权重30%
   
    """
    # 姿态奖励
    r_pose = pose_reward(q_cur, q_ref, scale=1.0)  # 
    
    # 速度奖励 
    r_vel = velocity_reward(vel_cur, vel_ref, scale=0.03)  
    
    total = w_pose * r_pose + w_vel * r_vel
    
    return total, {
        'r_pose': r_pose,
        'r_vel': r_vel,
        'r_total': total,
    }


def smooth_reward(action, prev_action, scale=0.001):
    """动作平滑度奖励"""
    if prev_action is None:
        return 0.0
    diff = np.linalg.norm(action - prev_action)
    return -scale * diff


def control_cost(action, scale=0.0001):
    """控制代价"""
    return -scale * np.linalg.norm(action) ** 2


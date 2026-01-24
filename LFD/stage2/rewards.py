import numpy as np


def pose_reward(q_cur, q_ref, scale=2.0):
    """Joint angle matching reward
    
    Uses exponential function, smaller error gives higher reward
    Larger scale makes it more sensitive to error
    """
    error = np.linalg.norm(q_cur - q_ref)
    return np.exp(-scale * error)


def ee_reward(pos_cur, pos_ref, scale=5.0):
    """End effector position matching reward"""
    error = np.linalg.norm(pos_cur - pos_ref)
    return np.exp(-scale * error ** 2)


def velocity_reward(vel_cur, vel_ref, scale=0.1):
    """Velocity matching reward"""
    error = np.linalg.norm(vel_cur - vel_ref)
    return np.exp(-scale * error)


def com_reward(com_cur, com_ref, scale=3.0):
    """Center of mass position matching reward"""
    # Only compare horizontal positions
    error_xy = np.linalg.norm(com_cur[:2] - com_ref[:2])
    # More tolerance for height
    error_z = abs(com_cur[2] - com_ref[2])
    error = error_xy + 0.3 * error_z
    return np.exp(-scale * error)


def tracking_reward(q_cur, q_ref, ee_cur, ee_ref, vel_cur, vel_ref, 
                   w_pose=0.4, w_ee=0.3, w_vel=0.3):
    """Composite tracking reward
    
    Parameters:
        w_pose: Pose weight
        w_ee: End effector position weight
        w_vel: Velocity weight
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
    """Tracking reward 
    
    Parameters:
        w_pose: Pose weight (primary)
        w_vel: Velocity weight (secondary)
    
  
    Pose is core, 70% weight
    Velocity as auxiliary, 30% weight
   
    """
    # Pose reward
    r_pose = pose_reward(q_cur, q_ref, scale=1.0)  # 
    
    # Velocity reward 
    r_vel = velocity_reward(vel_cur, vel_ref, scale=0.03)  
    
    total = w_pose * r_pose + w_vel * r_vel
    
    return total, {
        'r_pose': r_pose,
        'r_vel': r_vel,
        'r_total': total,
    }


def smooth_reward(action, prev_action, scale=0.001):
    """Action smoothness reward"""
    if prev_action is None:
        return 0.0
    diff = np.linalg.norm(action - prev_action)
    return -scale * diff


def control_cost(action, scale=0.0001):
    """Control cost"""
    return -scale * np.linalg.norm(action) ** 2


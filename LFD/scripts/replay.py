
'''回放专家动作 '''

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer





SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


TASK_MAP = {
    'stand': 'stand',
    'walk': 'walk',
    'run': 'run',
    'carry': 'walk',  # carry用walk场景
    'sit': 'sit_simple',
    'stair': 'stair',
}


def load_env_model(task='walk'):
    
  
    env_task = TASK_MAP.get(task, task)
    
    
    model_path = os.path.join(
        PROJECT_ROOT,
        f"humanoid_bench/assets/envs/h1_pos_{env_task}.xml"
    )
    
    # if not os.path.exists(model_path):
    #     print(f"错误: 找不到场景文件 {model_path}")
    #     sys.exit(1)
    
    print(f"加载场景: {os.path.basename(model_path)}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    return model, data


def load_traj(npz_path):
    """加载轨迹"""
    # if not os.path.exists(npz_path):
    #     print(f"错误: 找不到文件 {npz_path}")
    #     sys.exit(1)
    
    traj = np.load(npz_path)
    print(f"加载轨迹: {npz_path}")
    print(f"  帧数: {len(traj['root_pos'])}")
    print(f"  时长: {len(traj['root_pos'])/50:.2f}s")
    
    return traj


def play(model, data, traj, loop=True):
    """回放轨迹"""
    root_pos = traj['root_pos']
    dof = traj['dof']
    T = len(root_pos)
    
    print("\n开始回放...")
  
    
    frame = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # H1的qpos结构
            # 前7个: free joint [x, y, z, qw, qx, qy, qz]
            # 后19个: 关节角度
            
            # 设置状态
            data.qpos[0:3] = root_pos[frame]
            data.qpos[3:7] = [1, 0, 0, 0]
            data.qpos[7:26] = dof[frame]
            
        
            mujoco.mj_forward(model, data)
            
      
            viewer.sync()
            
       
            frame = (frame + 1) % T
            if frame == 0 and not loop:
                break
            
            
            import time
            time.sleep(0.02)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python visualize.py <npz_file> [task]")
        print("示例: python visualize.py LFD/data/h1_walk.npz")
        print("      python save_video.py LFD/data/h1_run.npz run")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    

    filename = os.path.basename(npz_path)
    task = filename.replace('h1_', '').replace('.npz', '')
    

    if len(sys.argv) > 2:
        task = sys.argv[2]
    
    print(f"任务场景: {task}")
    
    #
    model, data = load_env_model(task)
    traj = load_traj(npz_path)
    
    #回放
    play(model, data, traj, loop=True)


if __name__ == '__main__':
    main()

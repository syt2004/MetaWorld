import os
import sys
import numpy as np
import mujoco
import imageio


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


TASK_MAP = {
    'stand': 'stand',
    'walk': 'walk',
    'run': 'run',
    'carry': 'walk',
    'sit': 'sit_simple',
    'stair': 'stair',
}


def load_model(task):
 
    env_task = TASK_MAP.get(task, task)
    model_path = os.path.join(
        PROJECT_ROOT,
        f"humanoid_bench/assets/envs/h1_pos_{env_task}.xml"
    )
    
    if not os.path.exists(model_path):
        print(f"erro  模型 {model_path}")
        sys.exit(1)
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    
    renderer = mujoco.Renderer(model, height=480, width=640)
    
    return model, data, renderer


def render_video(model, data, renderer, traj, output_path, max_frames=None):
  
    root_pos = traj['root_pos']
    dof = traj['dof']
    T = len(root_pos)
    
    if max_frames:
        T = min(T, max_frames)
  
    frames = []
    
    for frame in range(T):
        
        data.qpos[0:3] = root_pos[frame]
        data.qpos[3:7] = [1, 0, 0, 0]
        data.qpos[7:26] = dof[frame]
        
        
        mujoco.mj_forward(model, data)

        
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels)
        
        
      
 
    print(f"\n保存视频 {output_path}")
    imageio.mimsave(output_path, frames, fps=50)
 


def main():
    
    if len(sys.argv) < 2:
        print("用法: python save_video.py <npz_file> [output_mp4] [max_frames]")
        print("示例: python save_video.py LFD/data/h1_walk.npz")
        print("      python save_video.py LFD/data/h1_walk.npz walk.mp4 500")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    
    
    filename = os.path.basename(npz_path)
    task = filename.replace('h1_', '').replace('.npz', '')
    
 
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    else:
        output_path = f"LFD/videos/{task}.mp4"

    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None
    
    print(f"任务: {task}")
    
    
    traj = np.load(npz_path)
    print(f"加载轨迹: {npz_path}")
    print(f"  帧数: {len(traj['root_pos'])}")
    
    model, data, renderer = load_model(task)
    
   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    
    render_video(model, data, renderer, traj, output_path, max_frames)


if __name__ == '__main__':
    main()


import os
import numpy as np
import imageio
from PIL import Image


class VideoRecorder:
   
    
    def __init__(self, save_dir, fps=15):
        self.save_dir = save_dir
        self.fps = fps
        self.frames = []
        self.is_recording = False
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start(self):
       
        self.frames = []
        self.is_recording = True
    
    def record(self, env):
       
        if not self.is_recording:
            return
        
     
        frame = env.render()
        if frame is not None:
            self.frames.append(frame)
    
    def stop(self):
      
        self.is_recording = False
    
    def save(self, step, prefix='video'):
       
      
        
        
        mp4_path = os.path.join(self.save_dir, f'{prefix}_{step}.mp4')
        imageio.mimsave(mp4_path, self.frames, fps=self.fps)
        print(f"  保存视频: {mp4_path} ({len(self.frames)}帧)")
        
      
        gif_frames = self.frames[:min(100, len(self.frames))]
        gif_path = os.path.join(self.save_dir, f'{prefix}_{step}.gif')
        
        # 缩小尺寸以减小gif大小
        resized = []
        for frame in gif_frames:
            img = Image.fromarray(frame)
            img = img.resize((320, 240), Image.Resampling.LANCZOS)
            resized.append(np.array(img))
        
        imageio.mimsave(gif_path, resized, fps=self.fps, loop=0)
        print(f"  保存GIF: {gif_path} ({len(resized)}帧)")
        
      
        self.frames = []
    
    def save_comparison(self, step):
    
        if len(self.frames) == 0:
            return
        
    
        indices = [0, len(self.frames)//2, -1]
        comparison = np.hstack([self.frames[i] for i in indices])
        
        img_path = os.path.join(self.save_dir, f'frames_{step}.png')
        imageio.imwrite(img_path, comparison)
    


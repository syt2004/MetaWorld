import os
import sys


SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))

from utils.loader import AMassLoader



DATA_ROOT = os.path.join(PROJECT_ROOT, "data/expert_data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../data")



TASKS = {
    'walk': {
        'path': os.path.join(DATA_ROOT, 'h1_walk/amass_all_cleaned.pkl'),
        'fps': 30, 
        'filter': ['walk', 'walking'],
    },
    'run': {
        'path': os.path.join(DATA_ROOT, 'h1_new_run/amass_all.pkl'),
        'fps': 30,
        'filter': None,  # 新数据已经是run动作，不需要过滤
    },
    'stand': {
        'path': os.path.join(DATA_ROOT, 'h1_new_stand/amass_all.pkl'),
        'fps': 30,
        'filter': None,  # 新数据已经是stand动作，不需要过滤
    },
    'sit': {
        'path': os.path.join(DATA_ROOT, 'h1_sit/amass_all.pkl'),
        'fps': 30,
        'filter': None,  # 使用h1_sit目录的数据
    },
    'crawl': {
        'path': os.path.join(DATA_ROOT, 'h1_new_crawl/amass_all.pkl'),
        'fps': 30,
        'filter': None,  # 爬行动作
    },
    'carry': {
        'path': os.path.join(DATA_ROOT, 'h1_carry/amass_all_cleaned.pkl'),
        'fps': 30,
        'filter': None,
    },
    'stair': {
        'path': os.path.join(DATA_ROOT, 'h1_stairs/amass_all_cleaned.pkl'),
        'fps': 30,
        'filter': None,
    },
    # reach 复用 data/reach_one_hand/
}


def process_all():

    loader = AMassLoader(fps=50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for task_name, config in TASKS.items():
        pkl_path = config['path']
        
      
        
        print(f"\n处理 {task_name}...")
        print(f"  输入: {pkl_path}")
        
        try:
            
            filter_kw = config.get('filter', None)
            keypoints = loader.process(pkl_path, src_fps=config['fps'], 
                                      filter_keywords=filter_kw)
            
            
            out_path = os.path.join(OUTPUT_DIR, f'h1_{task_name}.npz')
            loader.save(keypoints, out_path)
            
            
            T = len(keypoints['root_pos'])
            duration = T / 50.0
            print(f"  帧数: {T}, 时长: {duration:.2f}s")
            
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()


def process_single(task_name):
 
    if task_name not in TASKS:
       
        return
    
    loader = AMassLoader(fps=50)
    config = TASKS[task_name]
    pkl_path = config['path']
    filter_kw = config.get('filter', None)
    
    print(f"处理 {task_name}...")
    # if filter_kw:
    #     print(f"  过滤关键词: {filter_kw}")
    
    keypoints = loader.process(pkl_path, src_fps=config['fps'], 
                              filter_keywords=filter_kw)
    
    out_path = os.path.join(OUTPUT_DIR, f'h1_{task_name}.npz')
    loader.save(keypoints, out_path)
    



if __name__ == '__main__':
    if len(sys.argv) > 1:
       
        task = sys.argv[1]
        process_single(task)
    else:
      
        process_all()


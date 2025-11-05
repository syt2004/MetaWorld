import sys
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from gymnasium.envs import register
from LFD.stage2.tasks import (
    WalkTracking, StandTracking, RunTracking, SitTracking,
    CrawlTracking, CarryTracking, StairTracking
)


TRACKING_TASKS = {
    'stand': {
        'task_class': StandTracking,
        'model': 'h1_pos_stand.xml',
        'max_steps': 500,
    },
    'walk': {
        'task_class': WalkTracking,
        'model': 'h1_pos_walk.xml',
        'max_steps': 1000,
    },
    'run': {
        'task_class': RunTracking,
        'model': 'h1_pos_run.xml',
        'max_steps': 1000,
    },
    'sit': {
        'task_class': SitTracking,
        'model': 'h1_pos_sit_simple.xml',
        'max_steps': 1000,
    },
    'crawl': {
        'task_class': CrawlTracking,
        'model': 'h1_pos_walk.xml',  
        'max_steps': 1000,
    },
    'carry': {
        'task_class': CarryTracking,
        'model': 'h1_pos_walk.xml',  
        'max_steps': 500,
    },
    'stair': {
        'task_class': StairTracking,
        'model': 'h1_pos_stair.xml',
        'max_steps': 1000,
    },
}


for task_name, config in TRACKING_TASKS.items():
    register(
        id=f'h1-{task_name}-tracking-v0',
        entry_point='humanoid_bench.env:HumanoidEnv',
        max_episode_steps=config['max_steps'],
        kwargs={
            'robot': 'h1',
            'control': 'pos',
            'task': config['task_class'],
            'model_path': os.path.join(
                PROJECT_ROOT, 
                f"humanoid_bench/assets/envs/{config['model']}"
            ),
        }
    )


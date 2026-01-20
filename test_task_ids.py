#!/usr/bin/env python
"""Quick test to verify task_ids are loaded correctly."""
import sys
sys.path.insert(0, '.')
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from lumos.datasets.vision_wm_disk_dataset import VisionWMDiskDataset

PROJECT_ROOT = Path('.')
training_dir = Path('/home/akopyane/Desktop/rl/new_data/training')

observation_space = OmegaConf.load(PROJECT_ROOT / 'config/datamodule/observation_space/rgb_static_gripper_rel_act.yaml')
proprio_state = OmegaConf.load(PROJECT_ROOT / 'config/datamodule/proprioception_dims/robot_full.yaml')

dataset = VisionWMDiskDataset(
    datasets_dir=training_dir,
    obs_space=observation_space,
    proprio_state=proprio_state,
    key='vis',
    lang_folder='',
    num_workers=0,
    transforms={},
    batch_size=1,
    min_window_size=8,
    max_window_size=8,
    pad=False,
    for_wm=True,
    reset_prob=0.0,
    save_format='npz',
    use_cached_data=False,
    skip_empty_task_ids=False,
)

print(f'Dataset length: {len(dataset)}')
sample = dataset[0]
print(f'Sample keys: {list(sample.keys())}')
if 'state_info' in sample:
    print(f'state_info keys: {list(sample["state_info"].keys())}')
    if 'current_task_ids' in sample['state_info']:
        task_ids = sample['state_info']['current_task_ids']
        print(f'current_task_ids shape: {task_ids.shape}')
        print(f'current_task_ids values: {task_ids.flatten().tolist()}')
        print(f'unique values: {task_ids.unique().tolist()}')

        # Check if any are -1
        if -1 in task_ids.flatten().tolist():
            print('WARNING: Found -1 values (missing task labels)!')
        else:
            print('SUCCESS: All task_ids are valid (no -1 values)')
    else:
        print('ERROR: current_task_ids NOT in state_info!')

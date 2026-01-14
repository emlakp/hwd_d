#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASETS_DIR = Path("/home/akopyane/Desktop/rl/new_data/training")

import torch
from lumos.datasets.vision_wm_disk_dataset import VisionWMDiskDataset
from lumos.utils.nn_utils import transpose_collate_wm
from torch.utils.data import DataLoader

print(f"Loading dataset from {DATASETS_DIR}")
dataset = VisionWMDiskDataset(
    datasets_dir=DATASETS_DIR,
    obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "state_obs": ["robot_obs"], "actions": ["rel_actions"]},
    proprio_state={"robot_obs": 15},
    key="lang",
    lang_folder="lang_annotations",
    num_workers=4,
    transforms={},
    batch_size=4,
    min_window_size=50,
    max_window_size=50,
    pad=True,
    for_wm=True,
    reset_prob=0.05,
    save_format="npz",
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,  # Single process to see debug output
    collate_fn=transpose_collate_wm,
)

print("Starting to iterate...")
try:
    for i, batch in enumerate(loader):
        if i == 0:
            print(f"\nBatch {i} shapes:")
            if "state_info" in batch and "current_task_ids" in batch["state_info"]:
                print(f"  current_task_ids: {batch['state_info']['current_task_ids'].shape}")
        if i >= 2:
            print(f"Stopping after {i+1} batches")
            break
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

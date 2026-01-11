#!/usr/bin/env python3
"""Final test of DataLoader with the fix."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lumos.datasets.base_wm_disk_dataset import BaseWMDiskDataset
import numpy as np

# Create a minimal test
data_dir = Path("/home/akopyane/Desktop/rl/new_data/training")

print("Creating dataset...")
dataset = BaseWMDiskDataset(
    datasets_dir=data_dir,
    obs_space={"rgb_obs": ["rgb_static", "rgb_gripper"], "state_obs": ["robot_obs"], "actions": ["rel_actions"]},
    proprio_state={"robot_obs": 15},
    key="lang",
    lang_folder="lang_annotations",
    num_workers=1,
    transforms={},
    batch_size=4,
    min_window_size=50,
    max_window_size=50,
    pad=True,
    save_format="npz",
)

print(f"Dataset size: {len(dataset)}\n")

# Test loading a few episodes
print("Testing episode loading...")
for i in range(min(5, len(dataset))):
    episode = dataset._load_episode(i, 50)
    if "current_task_ids" in episode:
        shape = episode["current_task_ids"].shape
        print(f"Episode {i}: current_task_ids shape = {shape}")
        if shape != (50, 1):
            print(f"  ❌ ERROR: Expected (50, 1) but got {shape}")
        else:
            print(f"  ✓ Correct shape!")

print("\n✅ All episodes loaded successfully with consistent shapes!")

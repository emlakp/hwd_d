#!/usr/bin/env python3
"""
Simple direct check of what happens when we stack current_task_ids.
"""

import numpy as np
from pathlib import Path

data_dir = Path("/home/akopyane/Desktop/rl/new_data/training")

# Load episode boundaries
ep_start_end_ids = np.load(data_dir / "ep_start_end_ids.npy")
print(f"Testing first episode: {ep_start_end_ids[0]}")

start_idx, end_idx = ep_start_end_ids[0]

# Load first 50 frames of episode 0
frames_to_load = list(range(start_idx, start_idx + 50))

print(f"\nLoading frames {frames_to_load[0]} to {frames_to_load[-1]}...")

task_ids_list = []
for frame_idx in frames_to_load:
    npz_file = data_dir / f"episode_{frame_idx:07d}.npz"
    data = np.load(npz_file, allow_pickle=True)

    if "current_task_ids" in data:
        task_ids = data["current_task_ids"]
        task_ids_list.append(task_ids)
        if frame_idx < start_idx + 3:  # Print first 3
            print(f"Frame {frame_idx}: shape = {task_ids.shape}, value = {task_ids}")

# Now stack them like the dataset does
stacked = np.stack(task_ids_list)
print(f"\nAfter stacking {len(task_ids_list)} frames:")
print(f"  Shape: {stacked.shape}")
print(f"  Dtype: {stacked.dtype}")
print(f"  First 3 values:\n{stacked[:3]}")

# Check what ndim is
print(f"\n  stacked.ndim = {stacked.ndim}")

# Now simulate what get_state_info_dict does
task_ids = stacked
print(f"\nSimulating get_state_info_dict logic:")
print(f"  task_ids.ndim = {task_ids.ndim}")
if task_ids.ndim == 1:
    print(f"  → Would reshape from {task_ids.shape} to (-1, 1)")
    task_ids = task_ids.reshape(-1, 1)
    print(f"  → New shape: {task_ids.shape}")
else:
    print(f"  → No reshape needed, ndim != 1")

print(f"\nFinal shape going to collate: {task_ids.shape}")

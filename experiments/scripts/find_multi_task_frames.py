#!/usr/bin/env python3
"""Find frames that have multiple task IDs."""

import numpy as np
from pathlib import Path
import random

data_dir = Path("/home/akopyane/Desktop/rl/new_data/training")

# Load episode boundaries
ep_start_end_ids = np.load(data_dir / "ep_start_end_ids.npy")
print(f"Loaded {len(ep_start_end_ids)} episodes")

# Sample more frames - 5000 to increase chances of finding the issue
all_valid_indices = []
for start_idx, end_idx in ep_start_end_ids:
    all_valid_indices.extend(range(start_idx, end_idx + 1))

print(f"Total valid frames: {len(all_valid_indices)}")

n_samples = min(5000, len(all_valid_indices))
sampled_indices = random.sample(all_valid_indices, n_samples)

print(f"Checking {n_samples} random frames for non-(1,) shapes...\n")

multi_task_frames = []

for idx in sampled_indices:
    npz_file = data_dir / f"episode_{idx:07d}.npz"

    if not npz_file.exists():
        continue

    try:
        data = np.load(npz_file, allow_pickle=True)
        if "current_task_ids" in data:
            task_ids = data["current_task_ids"]
            if task_ids.shape != (1,):
                multi_task_frames.append((idx, task_ids.shape, task_ids))
                if len(multi_task_frames) <= 20:
                    print(f"  Frame {idx}: shape {task_ids.shape}, values: {task_ids}")
    except Exception as e:
        print(f"  Error loading {npz_file}: {e}")

print(f"\n{'='*70}")
print(f"RESULT: Found {len(multi_task_frames)} frames with non-(1,) shapes")
print(f"{'='*70}")

if multi_task_frames:
    print("\nSample of problematic frames:")
    for idx, shape, task_ids in multi_task_frames[:30]:
        print(f"  episode_{idx:07d}.npz: shape {shape}, values: {task_ids}")

    # Analyze which episodes they're from
    print(f"\n{'='*70}")
    print("Which episodes are affected?")
    print(f"{'='*70}")
    episode_map = {}
    for start_idx, end_idx in ep_start_end_ids:
        for frame_idx, shape, _ in multi_task_frames:
            if start_idx <= frame_idx <= end_idx:
                ep_num = list(ep_start_end_ids).index([start_idx, end_idx])
                if ep_num not in episode_map:
                    episode_map[ep_num] = []
                episode_map[ep_num].append((frame_idx, shape))

    for ep_num in sorted(episode_map.keys()):
        print(f"\nEpisode {ep_num} ({ep_start_end_ids[ep_num][0]}-{ep_start_end_ids[ep_num][1]}):")
        frames = episode_map[ep_num]
        print(f"  {len(frames)} affected frames")
        if len(frames) <= 10:
            for frame_idx, shape in frames:
                print(f"    Frame {frame_idx}: shape {shape}")
else:
    print("✓ All frames have shape (1,)")

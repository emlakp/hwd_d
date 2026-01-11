#!/usr/bin/env python3
"""Find the maximum number of task IDs in any frame."""

import numpy as np
from pathlib import Path
import random

data_dir = Path("/home/akopyane/Desktop/rl/new_data/training")

# Sample many frames to find the max
ep_start_end_ids = np.load(data_dir / "ep_start_end_ids.npy")

all_valid_indices = []
for start_idx, end_idx in ep_start_end_ids:
    all_valid_indices.extend(range(start_idx, end_idx + 1))

print(f"Sampling 10000 frames to find max task_ids length...\n")

n_samples = min(10000, len(all_valid_indices))
sampled_indices = random.sample(all_valid_indices, n_samples)

max_length = 0
length_distribution = {}
examples_by_length = {}

for idx in sampled_indices:
    npz_file = data_dir / f"episode_{idx:07d}.npz"

    if not npz_file.exists():
        continue

    try:
        data = np.load(npz_file, allow_pickle=True)
        if "current_task_ids" in data:
            task_ids = data["current_task_ids"]
            length = len(task_ids)

            if length > max_length:
                max_length = length

            if length not in length_distribution:
                length_distribution[length] = 0
                examples_by_length[length] = []
            length_distribution[length] += 1

            if len(examples_by_length[length]) < 5:
                examples_by_length[length].append((idx, task_ids))
    except Exception as e:
        pass

print(f"{'='*70}")
print(f"RESULTS")
print(f"{'='*70}")
print(f"Maximum task_ids length found: {max_length}")
print(f"\nDistribution:")
for length in sorted(length_distribution.keys()):
    count = length_distribution[length]
    pct = 100 * count / n_samples
    print(f"  Length {length}: {count} frames ({pct:.2f}%)")

    if length > 0 and len(examples_by_length[length]) > 0:
        print(f"    Examples:")
        for frame_idx, task_ids in examples_by_length[length][:3]:
            print(f"      Frame {frame_idx}: {task_ids}")

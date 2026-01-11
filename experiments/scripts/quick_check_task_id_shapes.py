#!/usr/bin/env python3
"""
Quick check: sample random files and check task_id shapes.
Since we have 500k+ files, we'll sample randomly to find problematic ones faster.
"""

import numpy as np
from pathlib import Path
import random

data_dir = Path("/home/akopyane/Desktop/rl/new_data/training")

# Load episode boundaries to sample from valid episodes
ep_start_end_ids = np.load(data_dir / "ep_start_end_ids.npy")
print(f"Loaded {len(ep_start_end_ids)} episodes")
print(f"Episode boundaries after trimming:\n{ep_start_end_ids[:5]}")

# Sample 1000 random valid frame indices
all_valid_indices = []
for start_idx, end_idx in ep_start_end_ids:
    all_valid_indices.extend(range(start_idx, end_idx + 1))

print(f"\nTotal valid frames: {len(all_valid_indices)}")

# Sample random indices
n_samples = min(1000, len(all_valid_indices))
sampled_indices = random.sample(all_valid_indices, n_samples)

print(f"Checking {n_samples} random frames...\n")

shapes_found = {}
problematic_files = []

for idx in sampled_indices:
    npz_file = data_dir / f"episode_{idx:07d}.npz"

    if not npz_file.exists():
        continue

    try:
        data = np.load(npz_file, allow_pickle=True)
        if "current_task_ids" in data:
            task_ids = data["current_task_ids"]
            shape = task_ids.shape

            # Count shapes
            if shape not in shapes_found:
                shapes_found[shape] = []
            shapes_found[shape].append(idx)

            # Flag problematic ones
            if len(task_ids) != 1:
                problematic_files.append((idx, shape, task_ids))
                if len(problematic_files) <= 10:  # Show first 10
                    print(f"  Frame {idx}: shape {shape}, values: {task_ids}")
    except Exception as e:
        print(f"  Error loading {npz_file}: {e}")

print(f"\n{'='*70}")
print("SHAPE DISTRIBUTION")
print(f"{'='*70}")
for shape, indices in sorted(shapes_found.items()):
    print(f"Shape {shape}: {len(indices)} files ({100*len(indices)/n_samples:.1f}%)")
    if len(indices) <= 5:
        print(f"  Indices: {indices}")

print(f"\n{'='*70}")
print(f"PROBLEMATIC FILES (non-(1,) shape): {len(problematic_files)}")
print(f"{'='*70}")
if problematic_files:
    print("These files have unexpected task_id counts:")
    for idx, shape, task_ids in problematic_files[:20]:  # Show first 20
        print(f"  episode_{idx:07d}.npz: shape {shape}, values: {task_ids}")
else:
    print("✓ All sampled files have shape (1,)")

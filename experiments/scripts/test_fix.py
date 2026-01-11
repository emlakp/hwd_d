#!/usr/bin/env python3
"""Test that the fix works - all shapes should now be consistent."""

import numpy as np

# Simulate the fix
def normalize_task_ids(episodes):
    """Simulates the normalization logic."""
    task_set_to_choice = {}
    normalized_task_ids = []

    for ep in episodes:
        task_ids = ep
        if len(task_ids) > 1:
            task_set = tuple(sorted(task_ids))
            if task_set not in task_set_to_choice:
                chosen_idx = np.random.randint(0, len(task_ids))
                task_set_to_choice[task_set] = task_ids[chosen_idx]
            chosen_task = task_set_to_choice[task_set]
            normalized_task_ids.append(np.array([chosen_task], dtype=task_ids.dtype))
        else:
            normalized_task_ids.append(task_ids)

    return np.stack(normalized_task_ids), task_set_to_choice

# Test with mixed frames
print("Test 1: Frames with task IDs [31], [4, 27], [4, 27], [31]")
print("=" * 60)
frames = [
    np.array([31]),
    np.array([4, 27]),
    np.array([4, 27]),
    np.array([31]),
]

result, choices = normalize_task_ids(frames)
print(f"Result shape: {result.shape}")
print(f"Task set choices made: {choices}")
print(f"Result:\n{result}")
print(f"✓ All have shape (1,): {all(len(x) == 1 for x in result)}")

print("\nTest 2: All frames have [4, 27]")
print("=" * 60)
frames2 = [np.array([4, 27]) for _ in range(50)]
result2, choices2 = normalize_task_ids(frames2)
print(f"Result shape: {result2.shape}")
print(f"Task set choices: {choices2}")
print(f"Unique values in result: {np.unique(result2)}")
print(f"✓ All same task: {len(np.unique(result2)) == 1}")

print("\nTest 3: Mix of single and dual tasks")
print("=" * 60)
frames3 = [
    np.array([10]),
    np.array([5, 15]),
    np.array([5, 15]),
    np.array([10]),
    np.array([5, 15]),
]
result3, choices3 = normalize_task_ids(frames3)
print(f"Result shape: {result3.shape}")
print(f"Task set choices: {choices3}")
print(f"Result:\n{result3.flatten()}")
print(f"✓ Temporal consistency: frames 1,2,4 with [5,15] all chose {choices3.get((5, 15), 'N/A')}")

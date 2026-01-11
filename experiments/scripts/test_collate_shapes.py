#!/usr/bin/env python3
"""Test what happens when we collate batches with different task_id shapes."""

import torch
from torch.utils.data._utils.collate import default_collate

# Simulate 4 samples in a batch
# Each sample is a dictionary with state_info containing current_task_ids

print("Test 1: All samples have shape [50, 1]")
print("=" * 60)
batch1 = [
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
]
try:
    collated1 = default_collate(batch1)
    print(f"✓ Success! Shape: {collated1['state_info']['current_task_ids'].shape}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTest 2: Mix of [50, 1] and [50, 2]")
print("=" * 60)
batch2 = [
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 2)}},  # Different!
    {"state_info": {"current_task_ids": torch.randn(50, 1)}},
]
try:
    collated2 = default_collate(batch2)
    print(f"✓ Success! Shape: {collated2['state_info']['current_task_ids'].shape}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTest 3: What if we have [50, 1, 1]?")
print("=" * 60)
batch3 = [
    {"state_info": {"current_task_ids": torch.randn(50, 1, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1, 1)}},
    {"state_info": {"current_task_ids": torch.randn(50, 1, 1)}},
]
try:
    collated3 = default_collate(batch3)
    print(f"✓ Success! Shape: {collated3['state_info']['current_task_ids'].shape}")
    print(f"  Note: This becomes [4, 50, 1, 1]")
except Exception as e:
    print(f"✗ Error: {e}")

print("\nTest 4: What about [50]?")
print("=" * 60)
batch4 = [
    {"state_info": {"current_task_ids": torch.randn(50)}},
    {"state_info": {"current_task_ids": torch.randn(50)}},
    {"state_info": {"current_task_ids": torch.randn(50)}},
    {"state_info": {"current_task_ids": torch.randn(50)}},
]
try:
    collated4 = default_collate(batch4)
    print(f"✓ Success! Shape: {collated4['state_info']['current_task_ids'].shape}")
    print(f"  Note: [50] becomes [4, 50] after collate")
except Exception as e:
    print(f"✗ Error: {e}")

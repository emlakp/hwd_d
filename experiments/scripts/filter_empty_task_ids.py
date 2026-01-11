#!/usr/bin/env python3
"""
One-time script to identify and filter out episodes with empty current_task_ids.
Creates a file listing which episodes to skip.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def filter_episodes_with_empty_task_ids(data_dir: Path):
    """
    Scan all .npz files and identify which ones have empty current_task_ids.
    Save the list of file indices to skip.
    """
    print(f"Scanning {data_dir}...")

    npz_files = sorted(data_dir.glob("**/*.npz"))
    print(f"Found {len(npz_files)} files")

    empty_file_indices = []

    for npz_file in tqdm(npz_files, desc="Checking files"):
        # Extract episode number from filename
        import re
        match = re.search(r"episode_(\d+)\.npz", str(npz_file))
        if not match:
            continue

        file_idx = int(match.group(1))

        try:
            data = np.load(npz_file, allow_pickle=True)
            if "current_task_ids" in data:
                task_ids = data["current_task_ids"]
                if len(task_ids) == 0:
                    empty_file_indices.append(file_idx)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            continue

    # Save the list of indices to skip
    output_file = data_dir / "empty_task_ids_indices.npy"
    np.save(output_file, np.array(empty_file_indices, dtype=np.int32))

    print(f"\n=== Summary ===")
    print(f"Total files: {len(npz_files)}")
    print(f"Files with empty current_task_ids: {len(empty_file_indices)} ({100*len(empty_file_indices)/len(npz_files):.2f}%)")
    print(f"Saved indices to skip: {output_file}")

    return empty_file_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter episodes with empty current_task_ids")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        exit(1)

    filter_episodes_with_empty_task_ids(data_dir)

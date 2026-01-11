#!/usr/bin/env python3
"""
Script to trim episode boundaries by removing empty current_task_ids frames at the end.
Creates a corrected ep_start_end_ids.npy file.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def trim_episode_boundaries(data_dir: Path):
    """
    Load ep_start_end_ids.npy and trim episodes to exclude empty current_task_ids at the end.
    Save the corrected boundaries back to disk.
    """
    print(f"Processing {data_dir}...")

    # Load original episode boundaries
    ep_file = data_dir / "ep_start_end_ids.npy"
    if not ep_file.exists():
        print(f"Error: {ep_file} not found")
        return

    # Backup original file
    backup_file = data_dir / "ep_start_end_ids_original.npy"
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        np.save(backup_file, np.load(ep_file))
    else:
        print(f"Backup already exists: {backup_file}")

    ep_start_end_ids = np.load(ep_file)
    print(f"Loaded {len(ep_start_end_ids)} episodes\n")

    corrected_episodes = []
    total_trimmed_frames = 0
    episodes_trimmed = 0

    for start_idx, end_idx in tqdm(ep_start_end_ids, desc="Trimming episodes"):
        # Find first empty frame from the end by going backwards
        first_empty_idx = None

        for frame_idx in range(end_idx, start_idx - 1, -1):  # Go backwards from end
            npz_file = data_dir / f"episode_{frame_idx:07d}.npz"

            if not npz_file.exists():
                continue

            try:
                data = np.load(npz_file, allow_pickle=True)
                if "current_task_ids" in data:
                    task_ids = data["current_task_ids"]
                    if len(task_ids) == 0:
                        first_empty_idx = frame_idx
                    else:
                        # Found a non-empty frame, stop searching
                        break
            except Exception as e:
                print(f"Error loading {npz_file}: {e}")
                break

        # Adjust end_idx if we found empty frames
        if first_empty_idx is not None and first_empty_idx > start_idx:
            new_end_idx = first_empty_idx - 1
            trimmed = end_idx - new_end_idx

            if trimmed > 0:
                total_trimmed_frames += trimmed
                episodes_trimmed += 1
                print(f"  Episode {start_idx}-{end_idx}: trimmed {trimmed} frames → new end: {new_end_idx}")
                corrected_episodes.append([start_idx, new_end_idx])
            else:
                corrected_episodes.append([start_idx, end_idx])
        else:
            # No empty frames found, keep original
            corrected_episodes.append([start_idx, end_idx])

    # Save corrected boundaries
    corrected_episodes = np.array(corrected_episodes, dtype=np.int32)
    np.save(ep_file, corrected_episodes)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total episodes: {len(ep_start_end_ids)}")
    print(f"Episodes trimmed: {episodes_trimmed}")
    print(f"Total frames trimmed: {total_trimmed_frames}")
    print(f"Average frames trimmed per episode: {total_trimmed_frames/episodes_trimmed:.1f}")
    print(f"\nSaved corrected boundaries to: {ep_file}")
    print(f"Original backed up to: {backup_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim episode boundaries by removing empty current_task_ids")
    parser.add_argument("data_dir", type=str, help="Path to dataset directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        exit(1)

    trim_episode_boundaries(data_dir)

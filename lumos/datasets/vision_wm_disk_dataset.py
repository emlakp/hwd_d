import logging
import pickle
from typing import Any

import numpy as np
from tqdm import tqdm

from lumos.datasets.base_wm_disk_dataset import BaseWMDiskDataset, load_npz

logger = logging.getLogger(__name__)


class VisionWMDiskDataset(BaseWMDiskDataset):
    """
    Dataset that loads episodes as individual files from disk.

    Args:
        skip_frames: Skip this amount of windows for language dataset.
        save_format: File format in datasets_dir (pkl or npz).
        pretrain: Set to True when pretraining.
    """

    def __init__(
        self,
        *args: Any,
        reset_prob: float = 0.05,
        skip_frames: int = 1,
        save_format: str = "npz",
        pretrain: bool = False,
        use_cached_data: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            *args,
            reset_prob=reset_prob,
            skip_frames=skip_frames,
            save_format=save_format,
            pretrain=pretrain,
            use_cached_data=False,
            **kwargs,
        )
        # Preloading is different for LIBERO compared to CALVIN
        self.use_cached_data = use_cached_data
        if self.use_cached_data:
            self.preloaded_data = {}  # Initialize as a dictionary
            self.preload_dataset(self.abs_datasets_dir / "cached_data.pkl")

    def preload_dataset(self, cached_data_path):
        """Preloads the entire dataset into memory."""
        if cached_data_path.is_file():
            logger.info("Loading preloaded data from cache...")
            with open(str(cached_data_path), "rb") as f:
                self.preloaded_data = pickle.load(f)
        else:
            data_dir_list = sorted([item for item in self.abs_datasets_dir.iterdir()])
            for file_path in tqdm(data_dir_list, desc="Preloading dataset"):
                if "npz" not in file_path.suffix:
                    continue

                # Skip non-episode files
                if "episode_" not in file_path.name:
                    continue

                key = self.extract_episode_number(file_path)
                np_obj = load_npz(file_path)
                data = {k: np_obj[k] for k in np_obj.keys()}

                value = {
                    "rel_actions": np.squeeze(data["rel_actions"]),
                    "robot_obs": np.squeeze(data["robot_obs"]),
                    "rgb_static": np.squeeze(data["rgb_static"]),
                    "rgb_gripper": np.squeeze(data["rgb_gripper"]),
                }
                # Add current_task_ids if available
                # Don't squeeze it - keep the shape (1,) to maintain consistency
                if "current_task_ids" in data:
                    task_ids = data["current_task_ids"]
                    # Remove only the first dimension added by stacking
                    if task_ids.ndim > 1:
                        task_ids = task_ids[0]
                    value["current_task_ids"] = task_ids
                self.preloaded_data[key] = value

            with open(str(cached_data_path), "wb") as f:
                pickle.dump(self.preloaded_data, f)
        logger.info("Preloaded the dataset into cache.")

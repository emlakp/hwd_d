import argparse
import logging
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningDataModule, Trainer, seed_everything
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from lumos.datasets.vision_wm_disk_dataset import VisionWMDiskDataset
from lumos.datasets.utils.episode_utils import load_dataset_statistics
from lumos.utils.info_utils import (
    get_last_checkpoint,
    print_system_env_info,
    setup_callbacks,
    setup_logger,
)
from lumos.utils.nn_utils import transpose_collate_wm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


class DebugDataModule(LightningDataModule):
    """LightningDataModule wrapping debug loaders because CalvinDataLoader is missing."""

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader, train_tfms: Dict, val_tfms: Dict):
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.train_transforms = dict(train_tfms)
        self.val_transforms = dict(val_tfms)
        self.train_transforms["out_rgb"] = self._wrap_out_rgb_transform(self.train_transforms.get("out_rgb"))
        self.val_transforms["out_rgb"] = self._wrap_out_rgb_transform(self.val_transforms.get("out_rgb"))

    def train_dataloader(self) -> DataLoader:
        return self._train_loader

    def val_dataloader(self) -> DataLoader:
        return self._val_loader

    @staticmethod
    def _wrap_out_rgb_transform(transform):
        if transform is None:
            return None

        def wrapped(tensor):
            if hasattr(tensor, "detach"):
                data = tensor.detach().cpu().numpy()
            else:
                data = np.asarray(tensor)
            return transform(data)

        return wrapped


def load_config() -> DictConfig:
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "config")):
        cfg = compose(config_name="train_wm_lstm")

    OmegaConf.set_struct(cfg, False)
    cfg.root = str(PROJECT_ROOT)
    if not cfg.exp_dir:
        cfg.exp_dir = str(PROJECT_ROOT / "logs" / "lstm_train" / datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Set logger name and id for wandb
    if cfg.logger and "name" in cfg.logger:
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        if cfg.comment != "":
            cfg.logger.name = f"{cfg.comment}_{date_time}"
        else:
            cfg.logger.name = date_time
    if cfg.logger and "id" in cfg.logger:
        cfg.logger.id = cfg.logger.name.replace("/", "_")

    if isinstance(cfg.logger, dict) and not cfg.logger:
        cfg.logger = None
    if "callbacks" in cfg:
        if "checkpoint" in cfg.callbacks:
            checkpoint_dir = Path(cfg.exp_dir) / "checkpoints"
            OmegaConf.set_struct(cfg.callbacks.checkpoint, False)
            cfg.callbacks.checkpoint.dirpath = str(checkpoint_dir)
            OmegaConf.set_struct(cfg.callbacks.checkpoint, True)
        OmegaConf.set_struct(cfg.callbacks, True)

    # Resolve config AFTER setting logger name
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)
    return cfg


def build_transforms(cfg: DictConfig) -> Tuple[Dict, Dict]:
    """Build train and validation transforms (64×64 resize)."""
    train_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "training"
    val_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "validation"

    transforms_cfg = load_dataset_statistics(train_dir, val_dir, cfg.datamodule.transforms)

    def compose_branch(branch: str) -> Dict:
        return {
            cam: torchvision.transforms.Compose([hydra.utils.instantiate(t) for t in transforms_cfg[branch][cam]])
            for cam in transforms_cfg[branch]
        }

    train_tfms = compose_branch("train")
    val_tfms = compose_branch("val")

    resize64 = transforms.Lambda(lambda x: F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False))

    def inject_resize(pipe: transforms.Compose) -> transforms.Compose:
        steps = list(pipe.transforms)
        return transforms.Compose([steps[0], resize64, *steps[1:]])

    for cam in ["rgb_static", "rgb_gripper"]:
        train_tfms[cam] = inject_resize(train_tfms[cam])
        val_tfms[cam] = inject_resize(val_tfms[cam])

    return train_tfms, val_tfms


def create_dataloader(cfg: DictConfig, train_tfms: Dict) -> Tuple[VisionWMDiskDataset, DataLoader]:
    train_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "training"
    dataset_cfg = cfg.datamodule.datasets.vision_dataset

    batch_size = cfg.datamodule.batch_size
    # batch_size = 64
    available = multiprocessing.cpu_count()
    OmegaConf.set_struct(dataset_cfg, False)
    # Reduce workers to avoid OOM - use at most 4 workers
    dataset_cfg.num_workers = min(4, max(1, available // 4))
    OmegaConf.set_struct(dataset_cfg, True)

    train_ds = VisionWMDiskDataset(
        datasets_dir=train_dir,
        obs_space=cfg.datamodule.observation_space,
        proprio_state=cfg.datamodule.proprioception_dims,
        key=dataset_cfg.key,
        lang_folder=dataset_cfg.lang_folder,
        num_workers=dataset_cfg.num_workers,
        transforms=train_tfms,
        batch_size=batch_size,
        min_window_size=cfg.datamodule.seq_len,
        max_window_size=cfg.datamodule.seq_len,
        pad=dataset_cfg.pad,
        for_wm=dataset_cfg.for_wm,
        reset_prob=cfg.datamodule.reset_prob,
        save_format=dataset_cfg.save_format,
        use_cached_data=dataset_cfg.use_cached_data,
    )

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=dataset_cfg.num_workers,
        collate_fn=transpose_collate_wm,
        persistent_workers=False,  # Disable to save memory
    )

    return train_ds, loader


def instantiate_model(cfg: DictConfig):
    wm_cfg = cfg.world_model
    OmegaConf.set_struct(wm_cfg, False)
    try:
        model = hydra.utils.instantiate(wm_cfg)
    finally:
        OmegaConf.set_struct(wm_cfg, True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.to(device)
    return model


def configure_paths(cfg: DictConfig) -> Path:
    exp_dir = Path(cfg.exp_dir)
    checkpoint_dir = Path(cfg.callbacks.checkpoint.dirpath)
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


def resolve_checkpoint(cfg: DictConfig, checkpoint_dir: Path) -> Optional[str]:
    explicit = OmegaConf.select(cfg, "ckpt_path", default=None)
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            rank_zero_info(f"Resuming from explicit checkpoint: {path}")
            return path.as_posix()
        rank_zero_info(f"Requested checkpoint not found: {path}")

    if OmegaConf.select(cfg, "resume", default=False):
        chk = get_last_checkpoint(checkpoint_dir)
        if chk:
            rank_zero_info(f"Resuming from checkpoint: {chk}")
            return chk.as_posix()
        rank_zero_info("Resume requested but no checkpoint found. Starting fresh.")

    return None


def main(resume: bool = False, checkpoint_path: Optional[str] = None) -> None:
    cfg = load_config()
    OmegaConf.set_struct(cfg, False)
    cfg.resume = resume
    cfg.ckpt_path = checkpoint_path
    OmegaConf.set_struct(cfg, True)

    seed_everything(cfg.seed, workers=True)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("medium")

    logger = setup_logger(cfg)
    callbacks = setup_callbacks(cfg.callbacks)
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)

    rank_zero_info(f"Training debug LSTM run with config:\n{OmegaConf.to_yaml(cfg)}")
    rank_zero_info(print_system_env_info())

    train_tfms, val_tfms = build_transforms(cfg)
    train_ds, train_loader = create_dataloader(cfg, train_tfms)
    _ = train_ds[0]  # Touch dataset once to surface loading issues early.

    datamodule = DebugDataModule(train_loader, train_loader, train_tfms, val_tfms)
    model = instantiate_model(cfg)

    if not isinstance(trainer_kwargs, dict):
        raise TypeError("cfg.trainer must translate to a mapping of Trainer arguments.")

    trainer = Trainer(logger=logger, callbacks=callbacks, **trainer_kwargs)
    checkpoint_dir = configure_paths(cfg)
    ckpt_path = resolve_checkpoint(cfg, checkpoint_dir)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug LSTM training script.")
    parser.add_argument("--resume", action="store_true", help="Resume training from the most recent checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint file to resume from.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(resume=args.resume, checkpoint_path=args.checkpoint)

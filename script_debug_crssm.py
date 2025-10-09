#!/usr/bin/env python3

import argparse
import multiprocessing
import sys
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision

from lumos.datasets.vision_wm_disk_dataset import VisionWMDiskDataset
from lumos.datasets.utils.episode_utils import load_dataset_statistics
from lumos.utils.nn_utils import transpose_collate_wm


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config():
    with initialize_config_dir(version_base="1.3", config_dir=str(PROJECT_ROOT / "config")):
        cfg = compose(config_name="train_wm_debug_crssm")

    OmegaConf.set_struct(cfg, False)
    cfg.root = str(PROJECT_ROOT)
    if not cfg.exp_dir:
        cfg.exp_dir = str(PROJECT_ROOT / "logs" / "notebook" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    if isinstance(cfg.logger, dict) and not cfg.logger:
        cfg.logger = None
    if "callbacks" in cfg:
        if "checkpoint" in cfg.callbacks:
            checkpoint_dir = Path(cfg.exp_dir) / "checkpoints"
            OmegaConf.set_struct(cfg.callbacks.checkpoint, False)
            cfg.callbacks.checkpoint.dirpath = str(checkpoint_dir)
            OmegaConf.set_struct(cfg.callbacks.checkpoint, True)
        OmegaConf.set_struct(cfg.callbacks, True)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, True)

    return cfg


def resolve_checkpoint_path(cfg, explicit_path=None):
    """Resolve checkpoint path when resuming training."""
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.is_file():
            return str(path)
        print(f"Requested checkpoint not found: {path}")
        return None

    checkpoint_dir = Path(cfg.exp_dir) / "checkpoints"
    checkpoint_cfg = getattr(cfg, "callbacks", None)
    if checkpoint_cfg is not None and getattr(checkpoint_cfg, "checkpoint", None):
        cfg_dir = getattr(checkpoint_cfg.checkpoint, "dirpath", None)
        if cfg_dir:
            checkpoint_dir = Path(cfg_dir)

    last_ckpt = checkpoint_dir / "last.ckpt"
    if last_ckpt.is_file():
        return str(last_ckpt)

    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return str(checkpoints[0])
    return None


def build_transforms(cfg):
    """Build train and validation transforms with 64x64 resize injected."""
    train_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "training"
    val_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "validation"

    transforms_cfg = load_dataset_statistics(train_dir, val_dir, cfg.datamodule.transforms)

    def build_tfms(branch):
        return {
            cam: torchvision.transforms.Compose([
                hydra.utils.instantiate(t) for t in transforms_cfg[branch][cam]
            ])
            for cam in transforms_cfg[branch]
        }

    train_tfms = build_tfms("train")
    val_tfms = build_tfms("val")

    resize64 = transforms.Lambda(
        lambda x: F.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)
    )

    def inject_resize(pipe):
        steps = list(pipe.transforms)
        return transforms.Compose([steps[0], resize64, *steps[1:]])

    for cam in ["rgb_static", "rgb_gripper"]:
        train_tfms[cam] = inject_resize(train_tfms[cam])
        val_tfms[cam] = inject_resize(val_tfms[cam])

    return train_tfms, val_tfms


def create_dataloader(cfg, train_tfms):
    train_dir = PROJECT_ROOT / cfg.datamodule.root_data_dir / "training"
    dataset_cfg = cfg.datamodule.datasets.vision_dataset

    batch_size = cfg.datamodule.batch_size

    available = multiprocessing.cpu_count()
    dataset_cfg.num_workers = max(1, available - 1)

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
    )

    return train_ds, loader


class DebugDataModule(LightningDataModule):
    """Minimal LightningDataModule to avoid ugly hacks and share transforms with the trainer"""

    def __init__(self, train_loader, val_loader, train_transforms, val_transforms):
        super().__init__()
        self._train_loader = train_loader
        self._val_loader = val_loader
        self.train_transforms = dict(train_transforms)
        self.val_transforms = dict(val_transforms)
        self.train_transforms["out_rgb"] = self._wrap_out_rgb_transform(self.train_transforms.get("out_rgb"))
        self.val_transforms["out_rgb"] = self._wrap_out_rgb_transform(self.val_transforms.get("out_rgb"))

    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
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


def create_model(cfg):
    """Instantiate ContextRSSM world model."""
    wm_cfg = cfg.world_model
    OmegaConf.set_struct(wm_cfg, False)

    model = hydra.utils.instantiate(wm_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.eval()

    return model


def create_trainer(cfg):
    """Create PyTorch Lightning trainer with CSV logging."""
    csv_logger = CSVLogger(save_dir=str(PROJECT_ROOT / "logs"), name="overfit_debug")
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    callbacks = []
    callbacks_cfg = getattr(cfg, "callbacks", None)
    if callbacks_cfg is not None:
        for cb in callbacks_cfg.values():
            callbacks.append(hydra.utils.instantiate(cb))
    if (
        isinstance(trainer_kwargs, dict)
        and trainer_kwargs.get("accelerator") == "gpu"
        and not torch.cuda.is_available()
    ):
        trainer_kwargs["accelerator"] = "cpu"
        trainer_kwargs["devices"] = 1
    trainer = Trainer(logger=csv_logger, callbacks=callbacks, **trainer_kwargs)
    return trainer



def main(resume=False, checkpoint_path=None):

    cfg = load_config()
    print(OmegaConf.to_yaml(cfg.datamodule))

    train_tfms, val_tfms = build_transforms(cfg)

    train_ds, loader = create_dataloader(cfg, train_tfms)

    sample = train_ds[0]


    model = create_model(cfg)

    datamodule = DebugDataModule(loader, loader, train_tfms, val_tfms)
    trainer = create_trainer(cfg)

    ckpt_path = None
    if resume:
        ckpt_path = resolve_checkpoint_path(cfg, checkpoint_path)
        if ckpt_path:
            print(f"\nResuming from checkpoint: {ckpt_path}")
        else:
            print("\nResume requested but no checkpoint found. Starting fresh.")

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug ContextRSSM training script.")
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

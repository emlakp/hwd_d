"""Quick sanity script for Context RSSM on HybridWMDiskDataset.

This avoids the old CalvinDataModule plumbing by instantiating
`HybridWMDiskDataset` directly, wrapping it in a `DataLoader`, and
running a single forward + loss pass through `DreamerV2ContextRSSM`.

Usage
-----
source .venv/bin/activate
python scripts/debug_hybrid_wm.py \
    --data-root dataset/calvin/30_64_rgbsg \
    --batch-size 4 \
    --seq-len 50
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import hydra
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from lumos.datasets.hybrid_wm_disk_dataset import HybridWMDiskDataset
from lumos.utils.nn_utils import transpose_collate_hybrid_wm


def build_model_cfg(robot_dim: int, action_dim: int) -> OmegaConf:
    """Return a minimal config for DreamerV2ContextRSSM."""

    cnn_depth = 48
    kernels = [4, 4, 4, 4]
    strides = [2, 2, 2, 2]
    paddings = [1, 1, 1, 1]

    decoder_base = {
        "_target_": "lumos.world_models.decoders.cnn_decoder.CnnDecoder",
        "cnn_depth": cnn_depth,
        "kernels": [5, 5, 6, 6],
        "strides": [2, 2, 2, 2],
        "paddings": [2, 2, 2, 2],
        "out_channels": [128, 64, 32, 3],
        "layer_norm": True,
        "activation": "elu",
        "mlp_layers": 0,
        "use_gripper_camera": True,
        "in_dim": 0,  # will be overwritten in `_prepare_decoder_cfgs`
    }

    cfg = {
        "_target_": "lumos.world_models.dreamer_v2_contextrssm.DreamerV2ContextRSSM",
        "_recursive_": False,
        "encoder": {
            "_target_": "lumos.world_models.encoders.cnn_encoder.CnnEncoder",
            "cnn_depth": cnn_depth,
            "kernels": kernels,
            "strides": strides,
            "paddings": paddings,
            "activation": "elu",
            "use_gripper_camera": True,
        },
        "decoder": {
            "precise": decoder_base,
            "coarse": {**decoder_base, "use_gripper_camera": False},
        },
        "crssm": {
            "_target_": "lumos.world_models.contextrssm.core.ContextRSSMCore",
            "_recursive_": False,
            "cell": {
                "_target_": "lumos.world_models.contextrssm.cell.ContextRSSMCell",
                "embed_dim": 0,  # overwritten during model init
                "action_dim": action_dim,
                "deter_dim": 512,
                "stoch_dim": 32,
                "stoch_rank": 32,
                "context_dim": 64,
                "hidden_dim": 512,
                "ensemble": 3,
                "layer_norm": True,
                "context_sample_noise": 0.05,
            },
        },
        "amp": {
            "autocast": {"_target_": "contextlib.nullcontext"},
            "scaler": {"_target_": "torch.cuda.amp.GradScaler", "enabled": False},
        },
        "optimizer": {"_target_": "torch.optim.Adam", "_partial_": True, "lr": 1e-3},
        "loss": {
            "kl_balance": 0.5,
            "kl_weight": 1.0,
            "context_kl_weight": 1.0,
            "image_weight": 1.0,
            "grad_clip": 100.0,
        },
        "train_batch_size": 0,  # overwritten after instantiation
        "val_batch_size": 0,
        "with_proprio": True,
        "use_gripper_camera": True,
        "robot_dim": robot_dim,
        "name": "debug_dreamer_v2_contextrssm",
    }

    return OmegaConf.create(cfg)


def load_observation_cfgs(root: Path) -> Tuple[OmegaConf, OmegaConf]:
    obs_cfg = OmegaConf.load(root / "config/datamodule/observation_space/rgb_static_gripper_rel_act.yaml")
    proprio_cfg = OmegaConf.load(root / "config/datamodule/proprioception_dims/robot_full.yaml")
    return obs_cfg, proprio_cfg


def make_dataloader(args: argparse.Namespace, obs_cfg: OmegaConf, proprio_cfg: OmegaConf) -> DataLoader:
    datasets_dir = Path(args.data_root)
    train_dir = datasets_dir / "training"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Expected training data under {train_dir}")

    dataset = HybridWMDiskDataset(
        datasets_dir=train_dir,
        obs_space=obs_cfg,
        proprio_state=proprio_cfg,
        key="hybrid",
        lang_folder="",
        num_workers=0,
        transforms={},
        batch_size=args.batch_size,
        min_window_size=args.seq_len,
        max_window_size=args.seq_len,
        pad=False,
        for_wm=True,
        reset_prob=args.reset_prob,
        save_format="npz",
        use_cached_data=args.cache_dataset,
    )

    def collate_wrapper(batch):
        return {"hybrid": transpose_collate_hybrid_wm(batch)}

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=collate_wrapper,
    )


def tensor_tree_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            out[key] = tensor_tree_to_device(value, device)
        elif torch.is_tensor(value):
            tensor = value
            if tensor.dtype == torch.uint8:
                tensor = tensor.float() / 255.0
            else:
                tensor = tensor.float() if tensor.is_floating_point() else tensor.to(device)
            out[key] = tensor.to(device)
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid WM debug runner")
    parser.add_argument("--data-root", type=Path, default=Path("dataset/calvin/30_64_rgbsg"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=50)
    parser.add_argument("--reset-prob", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cache-dataset", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    obs_cfg, proprio_cfg = load_observation_cfgs(project_root)

    dataloader = make_dataloader(args, obs_cfg, proprio_cfg)

    batch_cpu = next(iter(dataloader))
    hybrid_cpu = batch_cpu["hybrid"]
    robot_dim = hybrid_cpu["state_obs"].shape[-1]
    action_dim = hybrid_cpu["actions"]["pre_actions"].shape[-1]

    model_cfg = build_model_cfg(robot_dim=robot_dim, action_dim=action_dim)
    model_cfg.train_batch_size = args.batch_size
    model_cfg.val_batch_size = args.batch_size

    model = hydra.utils.instantiate(model_cfg)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    hybrid = tensor_tree_to_device(hybrid_cpu, device)

    init_state = tuple(t.to(device) for t in model.crssm_core.init_state(args.batch_size))

    outs = model(
        hybrid["rgb_obs"]["rgb_static"],
        hybrid["state_obs"],
        hybrid["actions"]["pre_actions"],
        hybrid["reset"],
        init_state,
        rgb_g=hybrid["rgb_obs"].get("rgb_gripper"),
    )

    metrics = model.loss(hybrid, outs)

    print("=== Forward summary ===")
    for key, tensor in outs.items():
        if torch.is_tensor(tensor):
            print(f"{key:>16}: {tuple(tensor.shape)}")
    print("\n=== Loss summary ===")
    for key, value in metrics.items():
        print(f"{key:>20}: {value.item():.4f}")


if __name__ == "__main__":
    main()

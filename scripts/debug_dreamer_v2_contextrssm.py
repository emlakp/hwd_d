"""Minimal forward pass script for DreamerV2ContextRSSM.""" 

import sys
from pathlib import Path

import torch
import hydra
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_cfg(image_size: int):
    cnn_depth = 8
    kernels = [4, 4, 4, 4]
    strides = [2, 2, 2, 2]
    paddings = [1, 1, 1, 1]

    final_spatial = image_size // (2 ** len(kernels))
    final_channels = cnn_depth * (2 ** (len(kernels) - 1))
    embed_dim = final_channels * final_spatial * final_spatial

    decoder_base = {
        "_target_": "lumos.world_models.decoders.cnn_decoder.CnnDecoder",
        "cnn_depth": cnn_depth,
        "kernels": kernels,
        "strides": strides,
        "paddings": paddings,
        "out_channels": [64, 32, 16, 3],
        "layer_norm": True,
        "activation": "elu",
        "mlp_layers": 0,
        "use_gripper_camera": False,
        "in_dim": 0,
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
            "use_gripper_camera": False,
        },
        "decoder": {
            "precise": decoder_base,
            "coarse": decoder_base.copy(),
        },
        "crssm": {
            "_target_": "lumos.world_models.contextrssm.core.ContextRSSMCore",
            "_recursive_": False,
            "cell": {
                "_target_": "lumos.world_models.contextrssm.cell.ContextRSSMCell",
                "embed_dim": embed_dim,
                "action_dim": 4,
                "deter_dim": 128,
                "stoch_dim": 16,
                "stoch_rank": 16,
                "context_dim": 64,
                "hidden_dim": 128,
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
        "train_batch_size": 2,
        "val_batch_size": 2,
        "with_proprio": False,
        "use_gripper_camera": False,
        "robot_dim": 0,
        "name": "debug_dreamer_v2_contextrssm",
    }

    return OmegaConf.create(cfg)


def main() -> None:
    device = torch.device("cpu")
    image_size = 32
    cfg = build_cfg(image_size)
    model = hydra.utils.instantiate(cfg).to(device)
    model.eval()

    T, B = 5, 2
    H = W = image_size
    action_dim = cfg.crssm.cell.action_dim

    rgb_static = torch.randn(T, B, 3, H, W, device=device)
    proprio = torch.zeros(T, B, 1, device=device)
    actions = torch.randn(T, B, action_dim, device=device)
    resets = torch.zeros(T, B, dtype=torch.bool, device=device)
    init_state = tuple(t.to(device) for t in model.crssm_core.init_state(B))

    with torch.no_grad():
        outs = model(rgb_static, proprio, actions, resets, init_state)

    print("=== Forward outputs ===")
    for key, value in outs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:>20}: {tuple(value.shape)}")
    if "out_states" in outs:
        for i, state in enumerate(outs["out_states"]):
            print(f"out_states[{i}]: {tuple(state.shape)}")

    losses = model.loss({"rgb_obs": {"rgb_static": rgb_static}}, outs)
    print("\n=== Loss summary ===")
    for key, value in losses.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:>24}: {value.item():.6f}")


if __name__ == "__main__":
    main()

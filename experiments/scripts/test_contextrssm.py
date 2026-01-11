"""Quick sanity rollout for ContextRSSMCore.

Run with the project venv activated:
    python scripts/test_contextrssm.py
"""

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lumos.world_models.contextrssm.core import ContextRSSMCore


def build_cfg(args: argparse.Namespace) -> OmegaConf:
    cfg_dict = {
        "_target_": "lumos.world_models.contextrssm.cell.ContextRSSMCell",
        "embed_dim": args.embed_dim,
        "action_dim": args.action_dim,
        "deter_dim": args.deter_dim,
        "stoch_dim": args.stoch_dim,
        "stoch_rank": args.stoch_rank,
        "context_dim": args.context_dim,
        "hidden_dim": args.hidden_dim,
        "ensemble": args.ensemble,
        "layer_norm": True,
        "context_sample_noise": args.context_noise,
    }
    return OmegaConf.create(cfg_dict)


def rollout(core: ContextRSSMCore, horizon: int, batch: int, device: torch.device) -> None:
    state = core.init_state(batch)
    state = tuple(s.to(device) for s in state)

    embeds = torch.randn(horizon, batch, core.cell.embed_dim, device=device)
    actions = torch.randn(horizon, batch, core.cell.action_dim, device=device)
    resets = torch.zeros(horizon, batch, dtype=torch.bool, device=device)

    outputs, next_state = core.forward(embeds, actions, resets, state)

    def shape_dict(tensors: Sequence[str]) -> str:
        return ", ".join(f"{key}:{outputs[key].shape}" for key in tensors)

    print("=== ContextRSSM rollout ===")
    print(f"device      : {device}")
    print(f"horizon/batch: {horizon}/{batch}")
    print(shape_dict(["priors", "posts", "ctxt_priors", "ctxt_posts"]))
    print(shape_dict(["deter", "stoch", "context", "gates", "features"]))
    print(
        "next_state :",
        ", ".join(f"{s.shape}" for s in next_state),
    )
    print("gate mean   :", outputs["gates"].mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity rollout for ContextRSSMCore")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--deter-dim", type=int, default=64)
    parser.add_argument("--stoch-dim", type=int, default=8)
    parser.add_argument("--stoch-rank", type=int, default=16)
    parser.add_argument("--context-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--ensemble", type=int, default=3)
    parser.add_argument("--context-noise", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    cfg = build_cfg(args)
    core = ContextRSSMCore(cfg).to(device)
    rollout(core, args.horizon, args.batch, device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run an old DreamerV2-LSTM checkpoint on a few samples and visualize task predictions."""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lumos.datasets.vision_wm_disk_dataset import VisionWMDiskDataset
from lumos.utils.nn_utils import transpose_collate_wm
from lumos.utils.transforms import NormalizeVector, ScaleImageTensor, UnNormalizeImageTensor
from lumos.world_models.dreamer_v2_lstm import DreamerV2LSTM


def parse_args() -> argparse.Namespace:
    default_checkpoint = PROJECT_ROOT / "logs" / "lstm_train" / "20260112_181214" / "checkpoints" / "last.ckpt"
    default_training_dir = Path("/home/akopyane/Desktop/rl/new_data/training")
    default_output = PROJECT_ROOT / "outputs" / "task_predictions" / "oldest_lstm_checkpoint.png"

    parser = argparse.ArgumentParser(description="Visualize DreamerV2-LSTM task predictions.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Checkpoint to load.")
    parser.add_argument("--training-dir", type=Path, default=default_training_dir, help="Training dataset directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Where to save the visualization image (PNG)."
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of windows per batch.")
    parser.add_argument("--seq-len", type=int, default=8, help="Sequence length to sample for each window.")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of frames to display.")
    parser.add_argument(
        "--example-index",
        type=int,
        default=0,
        help="Batch index (0-based) to visualize; useful if sampling more than one window.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for inference. Default CPU keeps things lightweight.",
    )
    return parser.parse_args()


def load_task_vocab() -> Dict[int, str]:
    """Load ordered task list from CALVIN task spec so we can map IDs to names."""
    task_config = PROJECT_ROOT / "config/callbacks/rollout/tasks/new_playtable_tasks.yaml"
    data = yaml.safe_load(task_config.read_text())
    tasks = list(data["tasks"].keys())
    return {idx: name for idx, name in enumerate(tasks)}


def build_transforms(resize_to: int = 64):
    """Create simple 64x64 resize + normalization pipeline."""
    scale = ScaleImageTensor()
    resize = transforms.Lambda(lambda x: F.interpolate(x, size=(resize_to, resize_to), mode="bilinear", align_corners=False))
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    branch = transforms.Compose([scale, resize, normalize])
    robot_norm = NormalizeVector()
    out_rgb = UnNormalizeImageTensor(mean=[0.5], std=[0.5])

    transforms_dict = {
        "rgb_static": branch,
        "rgb_gripper": branch,
        "robot_obs": robot_norm,
        "out_rgb": out_rgb,
    }
    return transforms_dict


def create_dataset(
    training_dir: Path,
    seq_len: int,
    batch_size: int,
    transforms_dict,
) -> VisionWMDiskDataset:
    observation_space = OmegaConf.load(PROJECT_ROOT / "config/datamodule/observation_space/rgb_static_gripper_rel_act.yaml")
    proprio_state = OmegaConf.load(PROJECT_ROOT / "config/datamodule/proprioception_dims/robot_full.yaml")

    dataset = VisionWMDiskDataset(
        datasets_dir=training_dir,
        obs_space=observation_space,
        proprio_state=proprio_state,
        key="vis",
        lang_folder="",
        num_workers=0,
        transforms=transforms_dict,
        batch_size=batch_size,
        min_window_size=seq_len,
        max_window_size=seq_len,
        pad=False,
        for_wm=True,
        reset_prob=0.0,
        save_format="npz",
        use_cached_data=False,
        skip_empty_task_ids=False,
    )
    return dataset


def load_model(checkpoint_path: Path, device: torch.device) -> DreamerV2LSTM:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        model = DreamerV2LSTM.load_from_checkpoint(checkpoint_path, strict=True, map_location=device)
    except RuntimeError as err:
        if "Missing key(s)" in str(err):
            print("Strict load failed (likely due to new parameters). Retrying with strict=False...")
            model = DreamerV2LSTM.load_from_checkpoint(checkpoint_path, strict=False, map_location=device)
        else:
            raise
    model.to(device)
    model.eval()
    return model


def run_inference(model: DreamerV2LSTM, batch, device: torch.device):
    rgb_static = batch["rgb_obs"]["rgb_static"].to(device)
    rgb_gripper = batch["rgb_obs"]["rgb_gripper"].to(device)
    robot_obs = batch["robot_obs"].to(device)
    pre_actions = batch["actions"]["pre_actions"].to(device)
    resets = batch["reset"].to(device)

    init_state = tuple(state.to(device) for state in model.lstm_core.init_state(rgb_static.shape[1]))
    with torch.no_grad():
        outputs = model(
            rgb_static,
            robot_obs,
            pre_actions,
            resets,
            init_state,
            rgb_g=rgb_gripper,
        )
        logits = outputs["context_logits"].detach().cpu()
    preds = logits.argmax(dim=-1)
    return preds, outputs


def select_timesteps(
    gt_seq: torch.Tensor,
    pred_seq: torch.Tensor,
    valid_mask: torch.Tensor,
    num_examples: int,
) -> List[int]:
    matches = (gt_seq == pred_seq) & valid_mask
    candidate_idxs = matches.nonzero(as_tuple=False).flatten().tolist()
    valid_idxs = valid_mask.nonzero(as_tuple=False).flatten().tolist()

    chosen: List[int] = []
    for idx in candidate_idxs:
        if idx not in chosen:
            chosen.append(idx)
        if len(chosen) == num_examples:
            break

    if len(chosen) < num_examples:
        for idx in valid_idxs:
            if idx not in chosen:
                chosen.append(idx)
            if len(chosen) == num_examples:
                break

    if not chosen:
        chosen = list(range(min(num_examples, gt_seq.shape[0])))
    return chosen[:num_examples]


def tensor_to_uint8(img_tensor: torch.Tensor, transform) -> np.ndarray:
    array = img_tensor.detach().cpu().numpy()
    if transform is not None:
        array = transform(array)
    else:
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = np.clip(array * 255.0, 0, 255)
        array = array.astype("uint8")
    return array


def format_task_label(task_id: int, id_to_name: Dict[int, str]) -> str:
    name = id_to_name.get(int(task_id))
    if name:
        return f"{name} (id={task_id})"
    return f"id={task_id}"


def save_visualization(
    images: Sequence[torch.Tensor],
    gt_ids: Sequence[int],
    pred_ids: Sequence[int],
    timesteps: Sequence[int],
    frame_ids: Sequence[int],
    out_transform,
    output_path: Path,
    id_to_name: Dict[int, str],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    panels: List[Image.Image] = []

    for img_tensor, gt, pred, step, frame in zip(images, gt_ids, pred_ids, timesteps, frame_ids):
        arr = tensor_to_uint8(img_tensor, out_transform)
        img = Image.fromarray(arr)

        label_height = 52
        panel = Image.new("RGB", (img.width, img.height + label_height), color=(20, 20, 20))
        panel.paste(img, (0, 0))

        label_lines = [
            f"t={step} frame={frame}",
            f"GT: {format_task_label(gt, id_to_name)}",
            f"Pred: {format_task_label(pred, id_to_name)}",
        ]

        draw = ImageDraw.Draw(panel)
        text_y = img.height + 6
        for line in label_lines:
            draw.text((8, text_y), line, font=font, fill=(230, 230, 230))
            text_y += font.size + 2

        panels.append(panel)

    total_width = sum(panel.width for panel in panels) + 8 * (len(panels) - 1)
    max_height = max(panel.height for panel in panels)
    canvas = Image.new("RGB", (total_width, max_height), color=(30, 30, 30))
    offset = 0
    for panel in panels:
        canvas.paste(panel, (offset, 0))
        offset += panel.width + 8

    canvas.save(output_path)


def main():
    args = parse_args()
    device = torch.device(args.device)
    training_dir = args.training_dir.expanduser().resolve()
    if not training_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {training_dir}")
    id_to_name = load_task_vocab()

    transforms_dict = build_transforms()
    dataset = create_dataset(training_dir, args.seq_len, args.batch_size, transforms_dict)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=transpose_collate_wm,
    )
    batch = next(iter(loader))
    frame_ids = batch["frame"]

    model = load_model(args.checkpoint, device)
    preds, _ = run_inference(model, batch, device)

    gt_ids = batch["state_info"]["current_task_ids"].squeeze(-1)
    valid_mask = gt_ids >= 0

    example_idx = min(args.example_index, preds.shape[1] - 1)
    gt_seq = gt_ids[:, example_idx]
    pred_seq = preds[:, example_idx]
    valid_seq = valid_mask[:, example_idx]
    frame_seq = frame_ids[:, example_idx]

    chosen_steps = select_timesteps(gt_seq, pred_seq, valid_seq, args.num_examples)
    out_transform = transforms_dict.get("out_rgb")

    images = [batch["rgb_obs"]["rgb_static"][step, example_idx] for step in chosen_steps]
    gt_values = [int(gt_seq[step].item()) for step in chosen_steps]
    pred_values = [int(pred_seq[step].item()) for step in chosen_steps]
    frame_values = [int(frame_seq[step].item()) for step in chosen_steps]

    save_visualization(
        images=images,
        gt_ids=gt_values,
        pred_ids=pred_values,
        timesteps=chosen_steps,
        frame_ids=frame_values,
        out_transform=out_transform,
        output_path=args.output,
        id_to_name=id_to_name,
    )

    print("Saved visualization to:", args.output)
    print("Sampled timesteps (t, frame, gt, pred, match):")
    for step, frame, gt, pred in zip(chosen_steps, frame_values, gt_values, pred_values):
        match = gt == pred
        print(f"  t={step:02d} frame={frame:06d} gt={gt} pred={pred} match={match}")


if __name__ == "__main__":
    main()

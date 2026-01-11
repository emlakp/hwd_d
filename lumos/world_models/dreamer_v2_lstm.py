import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import MissingMandatoryValue
from pytorch_lightning.utilities import rank_zero_only
import torch
from torch import Tensor
import torch.distributions as D
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from PIL import Image
import wandb

from lumos.utils.nn_utils import init_weights, st_clamp
from lumos.world_models.world_model import WorldModel
from lumos.world_models.contextrssm.lstm_core import ContextLSTMCore

logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


class DreamerV2LSTM(WorldModel):
    """
    The lightning module used for training DreamerV2 with LSTM-based context.
    This variant replaces GateL0RD with a simple LSTM for context dynamics.
    """

    def __init__(
        self,
        encoder: DictConfig,
        decoder: DictConfig,
        lstm: DictConfig,
        amp: DictConfig,
        optimizer: DictConfig,
        loss: DictConfig,
        train_batch_size: int,
        val_batch_size: int,
        with_proprio: bool,
        use_gripper_camera: bool,
        robot_dim: int,
        name: str,
    ):
        super(DreamerV2LSTM, self).__init__(name=name)
        self.use_gripper_camera = use_gripper_camera

        self.encoder = hydra.utils.instantiate(encoder)

        self.deter_dim = lstm.cell.deter_dim
        self.stoch_dim_total = lstm.cell.stoch_dim * (lstm.cell.stoch_rank or 1)
        self.context_dim = lstm.cell.context_dim
        precise_in_dim = self.deter_dim + self.stoch_dim_total + self.context_dim
        coarse_in_dim = self.stoch_dim_total + self.context_dim

        self.with_proprio = with_proprio
        try:
            current_embed_dim = lstm.cell.embed_dim
        except MissingMandatoryValue:
            current_embed_dim = None
        lstm.cell.embed_dim = self._infer_embed_dim(current_embed_dim, robot_dim)
        self.lstm_core = hydra.utils.instantiate(lstm)
        if not isinstance(self.lstm_core, ContextLSTMCore):
            raise TypeError(
                "DreamerV2LSTM expects lstm to instantiate ContextLSTMCore, "
                f"got {type(self.lstm_core)}"
            )

        precise_cfg, coarse_cfg = self._prepare_decoder_cfgs(decoder, precise_in_dim, coarse_in_dim)
        self.precise_decoder = hydra.utils.instantiate(precise_cfg)
        self.coarse_decoder = hydra.utils.instantiate(coarse_cfg)
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)
        self.optimizer = optimizer
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.kl_balance = loss.kl_balance
        self.kl_weight = loss.kl_weight
        self.context_kl_weight = getattr(loss, "context_kl_weight", 1.0)
        self.image_weight = loss.image_weight
        self.grad_clip = loss.grad_clip
        # Note: gate-related losses will be zero since LSTM returns dummy gates
        self.gate_activity_weight = getattr(loss, "gate_activity_weight", 0.0)
        self.ctxt_sparsity_weight = getattr(loss, "ctxt_sparsity_weight", 0.0)
        self.task_weight = getattr(loss, "task_weight", 1.0)
        self.num_tasks = getattr(loss, "num_tasks", self.context_dim)
        # Map bounded context features to unbounded task logits for supervision
        self.context_task_head = nn.Linear(self.context_dim, self.num_tasks)

        self.image_log_interval = getattr(loss, "image_log_interval", 1000)
        self.grad_log_interval = getattr(loss, "grad_log_interval", 200)
        self.metric_log_interval = getattr(loss, "metric_log_interval", None)
        self.metric_flush_size = getattr(loss, "metric_flush_size", 500)

        # Context tracking options
        self.track_context_changes = getattr(loss, "track_context_changes", False)
        self.track_context_variance = getattr(loss, "track_context_variance", False)
        self.track_context_update_freq = getattr(loss, "track_context_update_freq", False)
        self.log_context_heatmaps = getattr(loss, "log_context_heatmaps", False)
        self.context_change_threshold = getattr(loss, "context_change_threshold", 0.1)

        self.automatic_optimization = False

        self.batch_metrics = [
            "loss_total",
            "loss_reconstr",
            "loss_precise",
            "loss_precise-static",
            "loss_precise-gripper",
            "loss_coarse",
            "loss_coarse-static",
            "loss_coarse-gripper",
            "loss_kl-c",
            "loss_kl-c-post",
            "loss_kl-c-prior",
            "loss_kl-p",
            "loss_kl-p-post",
            "loss_kl-p-prior",
            "loss_gate-activity",
            "loss_ctxt-sparsity",
            "gate_activity",
            "gate_sparsity",
            "gate_time-steps",
            "task_accuracy",
        ]

        for m in self.modules():
            init_weights(m)
        self.num_val_batches = 0
        self.train_step_history: List[Dict[str, float]] = []
        self.save_hyperparameters()

    def _prepare_decoder_cfgs(
        self,
        decoder_cfg: DictConfig,
        precise_in_dim: int,
        coarse_in_dim: int,
    ) -> Tuple[DictConfig, DictConfig]:
        """Create separate decoder configs for precise and coarse heads."""
        decoder_dict = OmegaConf.to_container(decoder_cfg, resolve=False)
        if not isinstance(decoder_dict, dict):
            raise TypeError("decoder config must be a mapping")

        if "precise" in decoder_dict or "coarse" in decoder_dict:
            if "precise" not in decoder_dict or "coarse" not in decoder_dict:
                raise ValueError("decoder config must define both 'precise' and 'coarse' sections")
            precise_dict = deepcopy(decoder_dict["precise"])
            coarse_dict = deepcopy(decoder_dict["coarse"])
        else:
            precise_dict = deepcopy(decoder_dict)
            coarse_dict = deepcopy(decoder_dict)

        precise_dict["in_dim"] = precise_in_dim
        coarse_dict["in_dim"] = coarse_in_dim
        precise_dict["use_gripper_camera"] = self.use_gripper_camera
        coarse_dict["use_gripper_camera"] = self.use_gripper_camera

        precise_cfg = OmegaConf.create(precise_dict)
        coarse_cfg = OmegaConf.create(coarse_dict)
        return precise_cfg, coarse_cfg

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=3e-4)
        return {"optimizer": optimizer}


    def forward(
        self,
        rgb_s: Tensor,
        s_obs: Tensor,
        act: Tensor,
        reset: Tensor,
        in_state: Tuple[Tensor, Tensor, Tensor, Tensor],  # Now includes cell_state
        rgb_g: Tensor = None,
    ) -> Dict[str, Tensor]:
        embed = self._encode_observations(rgb_s, rgb_g, s_obs)

        outputs, out_states = self.lstm_core.forward(embed, act, reset, in_state)

        precise_out = self.precise_decoder(outputs["features"])
        dcd_img_s, dcd_img_g, dcd_s_obs = self._unpack_precise_decoder(precise_out)

        coarse_input = torch.cat((outputs["stoch"], outputs["context"]), dim=-1)
        coarse_out = self.coarse_decoder(coarse_input)
        coarse_rgb, coarse_rgb_g = self._unpack_coarse_decoder(coarse_out)

        outputs["dcd_img_s"] = dcd_img_s
        if dcd_img_g is not None:
            outputs["dcd_img_g"] = dcd_img_g
        if dcd_s_obs is not None:
            outputs["dcd_s_obs"] = dcd_s_obs
        outputs["coarse_rgb"] = coarse_rgb
        if coarse_rgb_g is not None:
            outputs["coarse_rgb_gripper"] = coarse_rgb_g
        outputs["out_states"] = out_states
        outputs["context_logits"] = self.context_task_head(outputs["context"])

        return outputs

    @staticmethod
    def _unpack_precise_decoder(decoded: Any) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        if isinstance(decoded, (tuple, list)):
            if len(decoded) == 3:
                return decoded[0], decoded[1], decoded[2]
            if len(decoded) == 2:
                return decoded[0], decoded[1], None
            if len(decoded) == 1:
                return decoded[0], None, None
        return decoded, None, None

    @staticmethod
    def _unpack_coarse_decoder(decoded: Any) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(decoded, (tuple, list)):
            if len(decoded) >= 2:
                return decoded[0], decoded[1]
            if len(decoded) == 1:
                return decoded[0], None
        return decoded, None

    def _infer_embed_dim(self, default_embed_dim: Optional[int], robot_dim: int) -> int:
        """Derive the encoder output dimensionality expected by ContextLSTM."""
        if hasattr(self.encoder, "cnn_out_dim"):
            img_dim = self.encoder.cnn_out_dim
        elif hasattr(self.encoder, "out_dim"):
            img_dim = self.encoder.out_dim
        elif hasattr(self.encoder, "cnn_depth") and hasattr(self.encoder, "kernels"):
            img_dim = self.encoder.cnn_depth * (2 ** (len(self.encoder.kernels) + 1))
        else:
            if default_embed_dim is None:
                raise ValueError("Unable to infer encoder output dimension for ContextLSTM")
            img_dim = default_embed_dim

        embed_dim = img_dim
        if self.with_proprio:
            if hasattr(self.encoder, "state_out_dim"):
                embed_dim += self.encoder.state_out_dim
            else:
                embed_dim += robot_dim

        return embed_dim

    def _encode_observations(self, rgb_s: Tensor, rgb_g: Optional[Tensor], s_obs: Tensor) -> Tensor:
        """Run observations through the encoder, handling proprio optionality."""
        if self.with_proprio:
            try:
                return self.encoder(rgb_s, rgb_g, s_obs)
            except TypeError:
                vision_embed = self.encoder(rgb_s, rgb_g)
                return torch.cat((vision_embed, s_obs), dim=-1)
        return self.encoder(rgb_s, rgb_g)

    def on_train_epoch_start(self) -> None:
        super(DreamerV2LSTM, self).on_train_epoch_start()
        self.in_state = self.lstm_core.init_state(self.train_batch_size)
        self.train_step_history = []

    def on_validation_epoch_start(self) -> None:
        super(DreamerV2LSTM, self).on_validation_epoch_start()
        self.val_state = self.lstm_core.init_state(self.val_batch_size)
        self.val_running_metrics: Dict[str, Tensor] = {}
        self.num_val_batches = 0
        self.val_gt_img_s: Optional[Tensor] = None
        self.val_precise_img_s: Optional[Tensor] = None
        self.val_coarse_img_s: Optional[Tensor] = None
        self.val_gt_img_g: Optional[Tensor] = None
        self.val_precise_img_g: Optional[Tensor] = None
        self.val_coarse_img_g: Optional[Tensor] = None
        # For tracking context changes
        self.context_change_images: List[Dict[str, Tensor]] = []

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor], context_task_loss = True) -> Dict[str, Tensor]:
        # Reconstruction losses split by camera and decoder head
        rgb_static = batch["rgb_obs"]["rgb_static"]
        precise_static = outs["dcd_img_s"]
        coarse_static = outs["coarse_rgb"]

        loss_precise_static = 0.5 * torch.square(precise_static - rgb_static).sum(dim=[-1, -2, -3])
        loss_coarse_static = 0.5 * torch.square(coarse_static - rgb_static).sum(dim=[-1, -2, -3])

        loss_precise_static_mean = loss_precise_static.mean()
        loss_coarse_static_mean = loss_coarse_static.mean()

        rgb_gripper = batch["rgb_obs"]["rgb_gripper"]
        precise_gripper = outs["dcd_img_g"]
        coarse_gripper = outs["coarse_rgb_gripper"]

        loss_precise_gripper = 0.5 * torch.square(precise_gripper - rgb_gripper).sum(dim=[-1, -2, -3])
        loss_coarse_gripper = 0.5 * torch.square(coarse_gripper - rgb_gripper).sum(dim=[-1, -2, -3])

        loss_precise_gripper_mean = loss_precise_gripper.mean()
        loss_coarse_gripper_mean = loss_coarse_gripper.mean()

        loss_precise_total = torch.stack([loss_precise_static_mean, loss_precise_gripper_mean]).mean()
        loss_coarse_total = torch.stack([loss_coarse_static_mean, loss_coarse_gripper_mean]).mean()

        recon_mean = loss_precise_total + loss_coarse_total

        # KL Loss
        post_logits = outs["posts"]
        precise_prior_logits = outs["precise_priors"]
        ctxt_prior_logits = outs["ctxt_priors"]

        dpost = self.lstm_core.zdistr(post_logits)
        dpost_detached = self.lstm_core.zdistr(post_logits.detach())
        dprior_p = self.lstm_core.zdistr(precise_prior_logits)
        dprior_p_detached = self.lstm_core.zdistr(precise_prior_logits.detach())
        dprior_c = self.lstm_core.ctxt_zdistr(ctxt_prior_logits)
        dprior_c_detached = self.lstm_core.ctxt_zdistr(ctxt_prior_logits.detach())

        loss_kl_c_1 = D.kl.kl_divergence(dpost_detached, dprior_c).mean()
        loss_kl_c_2 = D.kl.kl_divergence(dpost, dprior_c_detached).mean()

        loss_kl_p_1 = D.kl.kl_divergence(dpost, dprior_p_detached).mean()
        loss_kl_p_2 = D.kl.kl_divergence(dpost_detached, dprior_p).mean()

        loss_kl_c = (1 - self.kl_balance) * loss_kl_c_1 + self.kl_balance * loss_kl_c_2
        loss_kl_p = (1 - self.kl_balance) * loss_kl_p_1 + self.kl_balance * loss_kl_p_2

        # Gate-related metrics (will be dummy values since LSTM doesn't have real gates)
        gates = outs["gates"]
        gate_activity = gates.mean()
        ctxt_sparsity = gate_activity
        clamped_gates = st_clamp(gates)
        gate_time_steps = clamped_gates.sum(dim=-1).mean() * gates.shape[0]


        latent_entropy_post = dpost.entropy().mean()
        latent_entropy_prior_p = dprior_p.entropy().mean()
        latent_entropy_prior_c = dprior_c.entropy().mean()

        loss_gate_activity = self.gate_activity_weight * gate_activity
        loss_ctxt_sparsity = self.ctxt_sparsity_weight * ctxt_sparsity

        device = outs["context"].device
        loss_task_prediction = torch.tensor(0.0, device=device)
        task_accuracy = torch.tensor(0.0, device=device)
        if context_task_loss:
            context_logits = outs.get("context_logits")
            if context_logits is None:
                context_logits = self.context_task_head(outs["context"])
            flat_context_logits = context_logits.reshape(-1, context_logits.shape[-1])
            task_ids = batch["state_info"]["current_task_ids"].reshape(-1).long()
            loss_task_prediction = F.cross_entropy(flat_context_logits, task_ids)
            with torch.no_grad():
                preds = flat_context_logits.argmax(dim=-1)
                valid_mask = task_ids >= 0
                if valid_mask.any():
                    task_accuracy = (preds[valid_mask] == task_ids[valid_mask]).float().mean()

        loss = (
            self.kl_weight * (loss_kl_c + loss_kl_p)
            + self.image_weight * recon_mean
            + self.gate_activity_weight * loss_gate_activity
            + self.ctxt_sparsity_weight * loss_ctxt_sparsity
            + (self.task_weight * loss_task_prediction if context_task_loss else 0.0)
        )

        metrics = {
            "loss_total": loss,
            "loss_reconstr": recon_mean,
            "loss_precise": loss_precise_total,
            "loss_precise-static": loss_precise_static_mean,
            "loss_precise-gripper": loss_precise_gripper_mean,
            "loss_coarse": loss_coarse_total,
            "loss_coarse-static": loss_coarse_static_mean,
            "loss_coarse-gripper": loss_coarse_gripper_mean,
            "loss_kl-c": loss_kl_c,
            "loss_kl-p": loss_kl_p,
            "loss_gate-activity": loss_gate_activity,
            "loss_ctxt-sparsity": loss_ctxt_sparsity,
            "gate_activity": gate_activity,
            "gate_sparsity": ctxt_sparsity,
            "gate_time-steps": gate_time_steps,
            "latent_entropy_post": latent_entropy_post,
            "latent_entropy_precise_prior": latent_entropy_prior_p,
            "latent_entropy_coarse_prior": latent_entropy_prior_c,
            "loss_task_prediction": loss_task_prediction,
            "task_accuracy": task_accuracy,
        }

        # Optional context tracking metrics
        if self.track_context_changes or self.track_context_variance or self.track_context_update_freq:
            context = outs["context"]  # Shape: [T, B, context_dim]
            self._add_context_tracking_metrics(metrics, context, gates)

        return metrics

    def _track_context_change_images(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> None:
        """Track images before and after significant context changes.

        Args:
            batch: Input batch containing ground truth images
            outs: Model outputs containing context predictions
        """
        context = outs["context"]  # [T, B, context_dim]
        T = context.shape[0]

        if T < 2:
            return

        # Compute context changes
        context_diff = context[1:] - context[:-1]  # [T-1, B, context_dim]
        context_l2_change = torch.norm(context_diff, p=2, dim=-1)  # [T-1, B]

        # Find timesteps with significant changes (use first batch element)
        significant_changes = (context_l2_change[:, 0] > self.context_change_threshold).nonzero(as_tuple=True)[0]

        if len(significant_changes) == 0:
            return

        # Store up to 3 most significant changes
        changes_with_magnitude = [(t.item(), context_l2_change[t, 0].item()) for t in significant_changes]
        changes_with_magnitude.sort(key=lambda x: x[1], reverse=True)
        top_changes = changes_with_magnitude[:3]

        rgb_static = batch["rgb_obs"]["rgb_static"]
        rgb_gripper = batch["rgb_obs"].get("rgb_gripper")

        for t_idx, magnitude in top_changes:
            change_record = {
                "timestep": t_idx,
                "magnitude": magnitude,
                "context_before": context[t_idx, 0].detach().cpu(),
                "context_after": context[t_idx + 1, 0].detach().cpu(),
                "img_before": rgb_static[t_idx, 0].detach().cpu(),
                "img_after": rgb_static[t_idx + 1, 0].detach().cpu(),
            }

            if rgb_gripper is not None:
                change_record["img_gripper_before"] = rgb_gripper[t_idx, 0].detach().cpu()
                change_record["img_gripper_after"] = rgb_gripper[t_idx + 1, 0].detach().cpu()

            self.context_change_images.append(change_record)

    def _add_context_tracking_metrics(self, metrics: Dict[str, Tensor], context: Tensor, gates: Tensor) -> None:
        """Add optional context change tracking metrics.

        Args:
            metrics: Dictionary to add metrics to
            context: Context tensor of shape [T, B, context_dim]
            gates: Gate activations of shape [T, B, ...]
        """
        T = context.shape[0]

        if T < 2:
            # Need at least 2 timesteps to compute changes
            return

        # Compute L2 distance between consecutive timesteps
        if self.track_context_changes:
            context_diff = context[1:] - context[:-1]  # [T-1, B, context_dim]
            context_l2_change = torch.norm(context_diff, p=2, dim=-1)  # [T-1, B]
            metrics["context_l2_change_mean"] = context_l2_change.mean()
            metrics["context_l2_change_max"] = context_l2_change.max()
            metrics["context_l2_change_std"] = context_l2_change.std()

        # Compute variance over time dimension
        if self.track_context_variance:
            context_temporal_var = context.var(dim=0).mean()  # Variance across time, averaged over batch and dims
            metrics["context_temporal_variance"] = context_temporal_var

        # Track how often context significantly changes
        if self.track_context_update_freq:
            context_diff = context[1:] - context[:-1]  # [T-1, B, context_dim]
            context_l2_change = torch.norm(context_diff, p=2, dim=-1)  # [T-1, B]
            significant_changes = (context_l2_change > self.context_change_threshold).float()
            metrics["context_update_frequency"] = significant_changes.mean()
            metrics["context_update_count"] = significant_changes.sum(dim=0).mean()  # Avg updates per sequence

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        opt = self.optimizers()
        opt.zero_grad()

        rgb_static = batch["rgb_obs"]["rgb_static"]
        rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
        current_batch_size = rgb_static.shape[1]
        self._ensure_state_batch_size("train", current_batch_size)

        with self.autocast:
            outs = self(
                rgb_static,
                batch["robot_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.in_state,
                rgb_g=rgb_g_input,
            )
            losses = self.loss(batch, outs)

        self.in_state = outs["out_states"]

        self._record_train_step_metrics(losses)
        if self._should_flush_train_metrics():
            self._flush_train_metrics()

        self._maybe_log_train_images(batch, outs)
        self._step_with_scaler(opt, losses["loss_total"], losses)

        return losses["loss_total"]

    def on_train_epoch_end(self) -> None:
        super(DreamerV2LSTM, self).on_train_epoch_end()
        self._flush_train_metrics()

    def _ensure_state_batch_size(self, mode: str, batch_size: int) -> None:
        if mode == "train":
            state = getattr(self, "in_state", None)
            if state is None or state[0].shape[0] != batch_size:
                self.in_state = self.lstm_core.init_state(batch_size)
        else:
            state = getattr(self, "val_state", None)
            if state is None or state[0].shape[0] != batch_size:
                self.val_state = self.lstm_core.init_state(batch_size)

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Union[Tensor, Any]]:
        with self.autocast:
            rgb_g_input = batch["rgb_obs"].get("rgb_gripper")
            current_batch_size = batch["rgb_obs"]["rgb_static"].shape[1]
            self._ensure_state_batch_size("val", current_batch_size)
            outs = self(
                batch["rgb_obs"]["rgb_static"],
                batch["robot_obs"],
                batch["actions"]["pre_actions"],
                batch["reset"],
                self.val_state,
                rgb_g=rgb_g_input,
            )
            losses = self.loss(batch, outs)

        self.val_state = outs["out_states"]

        for key, val in losses.items():
            detach_val = val.detach()
            if key in self.val_running_metrics:
                self.val_running_metrics[key] = self.val_running_metrics[key] + detach_val
            else:
                self.val_running_metrics[key] = detach_val
        self.num_val_batches += 1

        # Track context changes and store corresponding images
        if self.log_context_heatmaps and self.track_context_changes:
            self._track_context_change_images(batch, outs)

        self.val_gt_img_s = batch["rgb_obs"]["rgb_static"][-1, 0]
        self.val_precise_img_s = outs["dcd_img_s"][-1, 0]

        coarse_rgb = outs.get("coarse_rgb")
        self.val_coarse_img_s = coarse_rgb[-1, 0] if coarse_rgb is not None else None

        self.val_gt_img_g = None
        self.val_precise_img_g = None
        self.val_coarse_img_g = None
        if self.use_gripper_camera:
            if rgb_g_input is not None:
                self.val_gt_img_g = rgb_g_input[-1, 0]

            precise_img_g = outs.get("dcd_img_g")
            if precise_img_g is not None:
                self.val_precise_img_g = precise_img_g[-1, 0]

            coarse_img_g = outs.get("coarse_rgb_gripper")
            if coarse_img_g is not None:
                self.val_coarse_img_g = coarse_img_g[-1, 0]

        return losses["loss_total"]

    def on_validation_epoch_end(self) -> None:
        if self.num_val_batches == 0:
            return

        averaged_metrics = {
            key: val / float(self.num_val_batches) for key, val in self.val_running_metrics.items()
        }
        self.log_metrics(averaged_metrics, mode="val")

        self.log_images(
            mode="val",
            gt_img_s=self.val_gt_img_s,
            precise_img_s=self.val_precise_img_s,
            coarse_img_s=self.val_coarse_img_s,
            gt_img_g=self.val_gt_img_g,
            precise_img_g=self.val_precise_img_g,
            coarse_img_g=self.val_coarse_img_g,
        )

        # Log context change images if tracked
        if self.log_context_heatmaps and len(self.context_change_images) > 0:
            self._log_context_change_images(mode="val")

    def log_images(
        self,
        mode: str,
        gt_img_s: Tensor,
        precise_img_s: Tensor,
        coarse_img_s: Optional[Tensor] = None,
        gt_img_g: Optional[Tensor] = None,
        precise_img_g: Optional[Tensor] = None,
        coarse_img_g: Optional[Tensor] = None,
    ) -> None:
        if not self.logger:
            return

        datamodule = getattr(self.trainer, "datamodule", None)
        transform: Optional[Callable[[Any], Any]] = None
        if datamodule is not None:
            train_transforms = getattr(datamodule, "train_transforms", {})
            if isinstance(train_transforms, dict):
                transform = train_transforms.get("out_rgb")

        def tensor_to_uint8(arr_tensor: Tensor) -> np.ndarray:
            data: Any = arr_tensor.to("cpu").detach()
            if transform is not None:
                try:
                    data = transform(data)
                except (AssertionError, TypeError):
                    if torch.is_tensor(data):
                        data = transform(data.cpu().numpy())
                    else:
                        data = transform(np.asarray(data))
            if torch.is_tensor(data):
                arr_np = data.to("cpu").detach().numpy()
            else:
                arr_np = np.asarray(data)
            if arr_np.ndim == 3 and arr_np.shape[0] in (1, 3):
                arr_np = np.transpose(arr_np, (1, 2, 0))
            if arr_np.dtype != np.uint8:
                max_val = float(np.max(arr_np)) if arr_np.size > 0 else 0.0
                if max_val <= 1.0:
                    arr_np = np.clip(arr_np * 255.0, 0, 255)
                else:
                    arr_np = np.clip(arr_np, 0, 255)
                arr_np = arr_np.astype(np.uint8)
            return arr_np

        rgb_static_gt = tensor_to_uint8(gt_img_s)
        rgb_static_precise = tensor_to_uint8(precise_img_s)

        saved_images = [
            ("rgb_static_gt", rgb_static_gt),
            ("rgb_static_precise", rgb_static_precise),
        ]

        coarse_static = None
        if coarse_img_s is not None:
            coarse_static = tensor_to_uint8(coarse_img_s)
            saved_images.append(("rgb_static_coarse", coarse_static))

        step = int(self.global_step)

        logger_name = self.logger.__class__.__name__.lower() if self.logger else ""
        images = []
        if "wandb" in logger_name:
            images.append(wandb.Image(rgb_static_gt, caption="gt"))
            images.append(wandb.Image(rgb_static_precise, caption="precise"))
            if coarse_static is not None:
                images.append(wandb.Image(coarse_static, caption="coarse"))

        if gt_img_g is not None and precise_img_g is not None:
            rgb_gripper_gt = tensor_to_uint8(gt_img_g)
            rgb_gripper_precise = tensor_to_uint8(precise_img_g)
            saved_images.extend(
                [
                    ("rgb_gripper_gt", rgb_gripper_gt),
                    ("rgb_gripper_precise", rgb_gripper_precise),
                ]
            )

            coarse_gripper = None
            if coarse_img_g is not None:
                coarse_gripper = tensor_to_uint8(coarse_img_g)
                saved_images.append(("rgb_gripper_coarse", coarse_gripper))

            if "wandb" in logger_name:
                images.append(wandb.Image(rgb_gripper_gt, caption="gt_g"))
                images.append(wandb.Image(rgb_gripper_precise, caption="precise_g"))
                if coarse_gripper is not None:
                    images.append(wandb.Image(coarse_gripper, caption="coarse_g"))

        if "wandb" in logger_name:
            self.logger.experiment.log({f"imgs/{mode}": images})
            return

        save_root = getattr(self.logger, "log_dir", None)
        if save_root is None:
            save_root = getattr(self.logger, "save_dir", None)
        if save_root is None:
            save_root = self.trainer.default_root_dir

        save_path = Path(save_root) / "images" / mode
        save_path.mkdir(parents=True, exist_ok=True)
        for tag, arr in saved_images:
            Image.fromarray(arr).save(save_path / f"{tag}_step{int(self.global_step)}.png")

    def _log_context_change_images(self, mode: str) -> None:
        """Log images before/after significant context changes with context heatmaps."""
        if not self.logger:
            return

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        logger_name = self.logger.__class__.__name__.lower() if self.logger else ""
        step = int(self.global_step)

        # Get transform for converting images
        datamodule = getattr(self.trainer, "datamodule", None)
        transform: Optional[Callable[[Any], Any]] = None
        if datamodule is not None:
            train_transforms = getattr(datamodule, "train_transforms", {})
            if isinstance(train_transforms, dict):
                transform = train_transforms.get("out_rgb")

        def tensor_to_uint8(arr_tensor: Tensor) -> np.ndarray:
            data: Any = arr_tensor.to("cpu").detach()
            if transform is not None:
                try:
                    data = transform(data)
                except (AssertionError, TypeError):
                    if torch.is_tensor(data):
                        data = transform(data.cpu().numpy())
                    else:
                        data = transform(np.asarray(data))
            if torch.is_tensor(data):
                arr_np = data.to("cpu").detach().numpy()
            else:
                arr_np = np.asarray(data)
            if arr_np.ndim == 3 and arr_np.shape[0] in (1, 3):
                arr_np = np.transpose(arr_np, (1, 2, 0))
            if arr_np.dtype != np.uint8:
                max_val = float(np.max(arr_np)) if arr_np.size > 0 else 0.0
                if max_val <= 1.0:
                    arr_np = np.clip(arr_np * 255.0, 0, 255)
                else:
                    arr_np = np.clip(arr_np, 0, 255)
                arr_np = arr_np.astype(np.uint8)
            return arr_np

        for idx, change_record in enumerate(self.context_change_images[:5]):  # Log up to 5 changes
            t = change_record["timestep"]
            mag = change_record["magnitude"]

            # Create figure with context visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f"Context Change at t={t}, magnitude={mag:.4f}", fontsize=16)

            # Row 1: Before
            img_before = tensor_to_uint8(change_record["img_before"])
            axes[0, 0].imshow(img_before)
            axes[0, 0].set_title(f"Static Camera (t={t})")
            axes[0, 0].axis('off')

            context_before = change_record["context_before"].numpy()
            axes[0, 1].bar(range(len(context_before)), context_before)
            axes[0, 1].set_title("Context Vector (Before)")
            axes[0, 1].set_xlabel("Dimension")
            axes[0, 1].set_ylabel("Value")

            if "img_gripper_before" in change_record:
                img_gripper_before = tensor_to_uint8(change_record["img_gripper_before"])
                axes[0, 2].imshow(img_gripper_before)
                axes[0, 2].set_title(f"Gripper Camera (t={t})")
                axes[0, 2].axis('off')
            else:
                axes[0, 2].axis('off')

            # Row 2: After
            img_after = tensor_to_uint8(change_record["img_after"])
            axes[1, 0].imshow(img_after)
            axes[1, 0].set_title(f"Static Camera (t={t+1})")
            axes[1, 0].axis('off')

            context_after = change_record["context_after"].numpy()
            axes[1, 1].bar(range(len(context_after)), context_after)
            axes[1, 1].set_title("Context Vector (After)")
            axes[1, 1].set_xlabel("Dimension")
            axes[1, 1].set_ylabel("Value")

            if "img_gripper_after" in change_record:
                img_gripper_after = tensor_to_uint8(change_record["img_gripper_after"])
                axes[1, 2].imshow(img_gripper_after)
                axes[1, 2].set_title(f"Gripper Camera (t={t+1})")
                axes[1, 2].axis('off')
            else:
                axes[1, 2].axis('off')

            plt.tight_layout()

            # Save to wandb or file system
            if "wandb" in logger_name:
                self.logger.experiment.log({
                    f"context_changes/{mode}/change_{idx}": wandb.Image(fig),
                    f"context_changes/{mode}/magnitude_{idx}": mag,
                })
            else:
                save_root = getattr(self.logger, "log_dir", None)
                if save_root is None:
                    save_root = getattr(self.logger, "save_dir", None)
                if save_root is None:
                    save_root = self.trainer.default_root_dir

                save_path = Path(save_root) / "images" / mode / "context_changes"
                save_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path / f"change_{idx}_t{t}_step{step}.png", dpi=100, bbox_inches='tight')

            plt.close(fig)

        # Clear the accumulated images
        self.context_change_images = []

    @torch.no_grad()
    def _flush_train_metrics(self) -> None:
        if not self.train_step_history:
            return
        if self.logger:
            for record in self.train_step_history:
                step = int(record.get("global_step", int(self.global_step)))
                aggregate_metrics = {}
                for key, value in record.items():
                    if key == "global_step":
                        continue
                    aggregate_metrics[key] = torch.tensor(value, device=self.device)
                if aggregate_metrics:
                    self.logger.log_metrics(
                        {self._format_metric_key(k): v.detach().cpu().item() for k, v in aggregate_metrics.items()},
                        step=step,
                    )
        self.train_step_history = []

    @torch.no_grad()
    def _record_train_step_metrics(self, metrics: Dict[str, Tensor]) -> None:
        step_record: Dict[str, float] = {"global_step": float(int(self.global_step))}
        for key, val in metrics.items():
            step_record[key] = float(val.detach().to("cpu"))
        self.train_step_history.append(step_record)

    def _format_metric_key(self, key: str) -> str:
        info = key.split("_")
        if len(info) >= 2:
            return f"{info[0]}/train-{info[1]}"
        return f"{key}/train"

    def _should_flush_train_metrics(self) -> bool:
        if not self.train_step_history:
            return False

        step = int(self.global_step) + 1
        if self.metric_flush_size and self.metric_flush_size > 0:
            if len(self.train_step_history) >= self.metric_flush_size:
                return True

        interval = None
        if self.metric_log_interval and self.metric_log_interval > 0:
            interval = self.metric_log_interval
        elif self.metric_flush_size and self.metric_flush_size > 0:
            interval = self.metric_flush_size

        if interval and interval > 0:
            return step % interval == 0

        return True

    def _should_log_images(self) -> bool:
        if not self.logger:
            return False

        step = int(self.global_step) + 1
        if self.image_log_interval and self.image_log_interval > 0:
            return step % self.image_log_interval == 0

        trainer_interval = getattr(self.trainer, "log_every_n_steps", None)
        return bool(trainer_interval and trainer_interval > 0 and step % trainer_interval == 0)

    def _maybe_log_train_images(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> None:
        if not self._should_log_images():
            return

        coarse_rgb = outs.get("coarse_rgb")
        coarse_img_s = coarse_rgb[-1, 0] if coarse_rgb is not None else None

        gt_img_g_val = None
        precise_img_g_val = None
        coarse_img_g_val = None

        if self.use_gripper_camera:
            rgb_gripper = batch["rgb_obs"].get("rgb_gripper")
            if rgb_gripper is not None:
                gt_img_g_val = rgb_gripper[-1, 0]

            precise_img_g = outs.get("dcd_img_g")
            if precise_img_g is not None:
                precise_img_g_val = precise_img_g[-1, 0]

            coarse_img_g = outs.get("coarse_rgb_gripper")
            if coarse_img_g is not None:
                coarse_img_g_val = coarse_img_g[-1, 0]

        self.log_images(
            mode="train",
            gt_img_s=batch["rgb_obs"]["rgb_static"][-1, 0],
            precise_img_s=outs["dcd_img_s"][-1, 0],
            coarse_img_s=coarse_img_s,
            gt_img_g=gt_img_g_val,
            precise_img_g=precise_img_g_val,
            coarse_img_g=coarse_img_g_val,
        )

    def _log_component_grad_norms(self, losses: Dict[str, Tensor]) -> None:
        """Log gradient norms for individual model components to diagnose vanishing gradients."""

        def compute_grad_norm(parameters):
            """Compute the L2 norm of gradients for a set of parameters."""
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5

        # Encoder gradients
        losses["grad-norm-encoder"] = torch.tensor(
            compute_grad_norm(self.encoder.parameters()), device=self.device
        )

        # LSTM core gradients
        losses["grad-norm-lstm"] = torch.tensor(
            compute_grad_norm(self.lstm_core.parameters()), device=self.device
        )

        # Break down LSTM into sub-components
        if hasattr(self.lstm_core, "cell"):
            cell = self.lstm_core.cell

            # Context LSTM gradients - important for detecting context learning issues
            if hasattr(cell, "context_lstm"):
                losses["grad-norm-context-lstm"] = torch.tensor(
                    compute_grad_norm(cell.context_lstm.parameters()), device=self.device
                )

            # GRU (precise dynamics) gradients
            if hasattr(cell, "gru"):
                losses["grad-norm-gru"] = torch.tensor(
                    compute_grad_norm(cell.gru.parameters()), device=self.device
                )

            # Posterior network gradients
            posterior_params = []
            for attr in ["post_mlp_h", "post_mlp_c", "post_mlp_e", "post_norm", "post_mlp"]:
                if hasattr(cell, attr):
                    posterior_params.extend(getattr(cell, attr).parameters())
            if posterior_params:
                losses["grad-norm-posterior"] = torch.tensor(
                    compute_grad_norm(posterior_params), device=self.device
                )

            # Prior networks gradients
            prior_params = []
            for attr in ["prior_mlp_h", "prior_mlp_c", "prior_norm", "prior_ensemble"]:
                if hasattr(cell, attr):
                    prior_params.extend(getattr(cell, attr).parameters())
            if prior_params:
                losses["grad-norm-prior"] = torch.tensor(
                    compute_grad_norm(prior_params), device=self.device
                )

            # Context prior/posterior heads gradients - important for context prediction
            ctxt_params = []
            for attr in ["ctxt_head", "ctxt_ensemble", "ctxt_post_head", "ctxt_post_ensemble"]:
                if hasattr(cell, attr):
                    ctxt_params.extend(getattr(cell, attr).parameters())
            if ctxt_params:
                losses["grad-norm-context-heads"] = torch.tensor(
                    compute_grad_norm(ctxt_params), device=self.device
                )

        # Decoder gradients
        losses["grad-norm-precise-decoder"] = torch.tensor(
            compute_grad_norm(self.precise_decoder.parameters()), device=self.device
        )
        losses["grad-norm-coarse-decoder"] = torch.tensor(
            compute_grad_norm(self.coarse_decoder.parameters()), device=self.device
        )

        # Task prediction head gradients (if using task supervision)
        if hasattr(self, "context_task_head"):
            losses["grad-norm-task-head"] = torch.tensor(
                compute_grad_norm(self.context_task_head.parameters()), device=self.device
            )

    def _step_with_scaler(
        self, opt: torch.optim.Optimizer, total_loss: Tensor, losses: Dict[str, Tensor]
    ) -> None:
        """Apply gradient scaling, clipping, and optimizer step."""
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(opt)

        # Log per-component gradient norms before clipping
        if self.global_step % self.grad_log_interval == 0:
            self._log_component_grad_norms(losses)

        grad_total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        if not isinstance(grad_total_norm, torch.Tensor):
            grad_total_norm = torch.tensor(grad_total_norm, device=self.device)
        losses["grad-total-norm"] = grad_total_norm.detach()

        self.scaler.step(opt)
        self.scaler.update()

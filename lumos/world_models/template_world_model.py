import logging
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor

from lumos.world_models.world_model import WorldModel

logger = logging.getLogger(__name__)


class TemplateWorldModel(WorldModel):
    """
    Draft implementation of a world model module.

    This class is meant to be used as a starting point when wiring up a new
    world-model variant. Replace the TODO blocks with logic that matches your
    component interfaces and data layout.
    """

    def __init__(
        self,
        encoder: Optional[DictConfig],
        dynamics: Optional[DictConfig],
        decoder: Optional[DictConfig],
        optimizer: DictConfig,
        loss: DictConfig,
        name: str,
        train_batch_size: int,
        val_batch_size: int,
        amp: Optional[DictConfig] = None,
    ):
        super().__init__(name=name)

        self.encoder_cfg = encoder
        self.dynamics_cfg = dynamics
        self.decoder_cfg = decoder
        self.optimizer_cfg = optimizer
        self.loss_cfg = loss
        self.amp_cfg = amp

        self.encoder = hydra.utils.instantiate(encoder) if encoder is not None else None
        self.dynamics = hydra.utils.instantiate(dynamics) if dynamics is not None else None
        self.decoder = hydra.utils.instantiate(decoder) if decoder is not None else None

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.kl_weight = getattr(loss, "kl_weight", 1.0)
        self.recon_weight = getattr(loss, "image_weight", 1.0)
        self.reg_weight = getattr(loss, "regularizer_weight", 0.0)

        autocast_cfg = getattr(amp, "autocast", None) if amp is not None else None
        self.autocast = hydra.utils.instantiate(autocast_cfg) if autocast_cfg else nullcontext()
        scaler_cfg = getattr(amp, "scaler", None) if amp is not None else None
        self.scaler = hydra.utils.instantiate(scaler_cfg) if scaler_cfg else None

        self.automatic_optimization = False
        self._train_state: Optional[Any] = None
        self._val_state: Optional[Any] = None

        self.loggable_metrics = [
            "loss_total",
            "loss_reconstruction",
            "loss_kl",
            "loss_regularizer",
        ]

    # --------------------------------------------------------------------- #
    # Core hooks to adapt to the new world model                            #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        observations: Tensor,
        actions: Tensor,
        resets: Tensor,
        in_state: Any,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """
        Run a single rollout step through encoder -> dynamics -> decoder.

        TODO:
            * Adjust the signature to match the datamodule outputs.
            * Populate the returned dictionary with everything your loss
              function needs (e.g. priors, posts, reconstructions, latents).
        """
        raise NotImplementedError("Implement the forward pass for your world model.")

    def dream(
        self,
        actions: Tensor,
        in_state: Any,
        temperature: float = 1.0,
    ) -> Tuple[Any, Any, Any]:
        """
        Imagine future trajectories purely from latent dynamics.

        TODO:
            * Delegate to the dynamics module's imagination routine.
            * Decide what pieces (priors, states, context) you want returned.
        """
        raise NotImplementedError("Implement imagination for your world model.")

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Combine reconstruction, KL, and any auxiliary regularizers into metrics.

        TODO:
            * Read the tensors produced in forward.
            * Compute individual loss terms and aggregate them into
              `loss_total`.
            * Return every metric you want to log.
        """
        raise NotImplementedError("Implement the training loss for your world model.")

    # --------------------------------------------------------------------- #
    # Lightning hooks with recommended structure                            #
    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        """
        Instantiate the optimizer (and optional LR schedulers) via Hydra.
        """
        optimizer = hydra.utils.instantiate(self.optimizer_cfg, params=self.parameters())
        return {"optimizer": optimizer}

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Example training loop with manual optimization.

        TODO:
            * Prepare input tensors that match your forward signature.
            * Keep the pattern of zero_grad -> forward -> loss -> backward -> step.
        """
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)

        autocast_ctx = self.autocast if hasattr(self.autocast, "__enter__") else nullcontext()
        with autocast_ctx:
            outs = self.forward(
                batch["observations"],  # TODO: adapt to datamodule keys
                batch["actions"],
                batch.get("resets", torch.zeros_like(batch["actions"][:, :, 0])),
                self._ensure_train_state(batch),
            )
            metrics = self.loss(batch, outs)
            loss = metrics["loss_total"]
            if "next_state" in outs:
                self._train_state = outs["next_state"]

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), getattr(self.loss_cfg, "grad_clip", 100.0))
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), getattr(self.loss_cfg, "grad_clip", 100.0))
            optimizer.step()

        self._log_step_metrics(metrics, mode="train")
        return metrics

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        """
        Example validation loop mirroring the training structure.
        """
        with torch.no_grad():
            outs = self.forward(
                batch["observations"],  # TODO: adapt to datamodule keys
                batch["actions"],
                batch.get("resets", torch.zeros_like(batch["actions"][:, :, 0])),
                self._ensure_val_state(batch),
            )
            metrics = self.loss(batch, outs)
            if "next_state" in outs:
                self._val_state = outs["next_state"]
            self._log_step_metrics(metrics, mode="val")
        return metrics

    # --------------------------------------------------------------------- #
    # Helper utilities you may reuse or override                            #
    # --------------------------------------------------------------------- #
    def _ensure_train_state(self, batch: Dict[str, Tensor]) -> Any:
        """
        Lazily initialize the training recurrent state.
        Override if your dynamics require a custom shape.
        """
        if self._train_state is None:
            self._train_state = self._init_state(self.train_batch_size)
        return self._train_state

    def _ensure_val_state(self, batch: Dict[str, Tensor]) -> Any:
        """
        Lazily initialize the validation recurrent state.
        """
        if self._val_state is None:
            self._val_state = self._init_state(self.val_batch_size)
        return self._val_state

    def _init_state(self, batch_size: int) -> Any:
        """
        Create an initial latent/deterministic state tuple.

        TODO: Replace this stub with a call into your dynamics module.
        """
        raise NotImplementedError("Provide an init state routine for your dynamics module.")

    def _log_step_metrics(self, metrics: Dict[str, Tensor], mode: str) -> None:
        """
        Shared logging helper.
        """
        for key, value in metrics.items():
            if key in self.loggable_metrics:
                self.log(f"{mode}/{key}", value, prog_bar=(key == "loss_total"), on_step=True, on_epoch=True)

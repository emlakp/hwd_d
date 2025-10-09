import logging
from typing import Any, Dict, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from lumos.world_models.world_model import WorldModel

logger = logging.getLogger(__name__)


class CRSSMHighLevel(WorldModel):
    """
    High-level variant of Context RSSM with dedicated heads for action, time, and reward.

    This class intentionally ships as a scaffold. It wires together the major modules
    (posterior/prior action encoders plus multiple prediction heads) and leaves the
    modelling specifics for you to fill in. Replace the TODO sections with the actual
    computation you need once the module interfaces are defined.
    """

    def __init__(
        self,
        posterior_action_encoder: Optional[DictConfig],
        prior_action_encoder: Optional[DictConfig],
        action_predictor: Optional[DictConfig],
        stochastic_state_predictor: Optional[DictConfig],
        time_predictor: Optional[DictConfig],
        reward_predictor: Optional[DictConfig],
        optimizer: DictConfig,
        loss: DictConfig,
        name: str,
        action_dim: int,
        hl_action_dim: int = 0,
        continuous_action: bool = True,
        min_std: float = 0.1,
    ):
        super().__init__(name=name)

        self.action_dim = action_dim
        self.hl_action_dim = hl_action_dim
        self.continuous_action = continuous_action
        self.min_std = min_std

        # Loss and optimization configs.
        self.loss_cfg = loss
        self.optimizer_cfg = optimizer

        # Core modules (instantiate via Hydra when configs are provided).
        self.posterior_action_encoder = self._maybe_instantiate(
            posterior_action_encoder, module_name="posterior_action_encoder"
        )
        self.prior_action_encoder = self._maybe_instantiate(
            prior_action_encoder, module_name="prior_action_encoder"
        )

        self.action_predictor = self._maybe_instantiate(
            action_predictor, module_name="action_predictor"
        )
        self.stochastic_state_predictor = self._maybe_instantiate(
            stochastic_state_predictor, module_name="stochastic_state_predictor"
        )
        self.time_predictor = self._maybe_instantiate(
            time_predictor, module_name="time_predictor"
        )
        self.reward_predictor = self._maybe_instantiate(
            reward_predictor, module_name="reward_predictor"
        )

        # Optionally collect predictors in a ModuleDict for convenience.
        self.prediction_heads = nn.ModuleDict(
            {
                "action": self.action_predictor if self.action_predictor else nn.Identity(),
                "stochastic_state": self.stochastic_state_predictor
                if self.stochastic_state_predictor
                else nn.Identity(),
                "time": self.time_predictor if self.time_predictor else nn.Identity(),
                "reward": self.reward_predictor if self.reward_predictor else nn.Identity(),
            }
        )

        self.loggable_metrics = [
            "loss_total",
            "loss_action",
            "loss_state",
            "loss_time",
            "loss_reward",
        ]

    # --------------------------------------------------------------------- #
    # Lightning hooks                                                       #
    # --------------------------------------------------------------------- #
    def configure_optimizers(self):
        return {
            "optimizer": hydra.utils.instantiate(
                self.optimizer_cfg,
                params=self.parameters(),
            )
        }

    # --------------------------------------------------------------------- #
    # High-level RSSM interface (fill in for your use case)                 #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        embeddings: Tensor,
        prev_state: Any,
        prev_action: Tensor,
        *extra_inputs: Tensor,
        **kwargs: Any,
    ) -> Dict[str, Tensor]:
        """
        Run one transition step.

        TODO:
            * Encode actions (posterior vs. prior) based on your training signal.
            * Update the latent state via `stochastic_state_predictor`.
            * Trigger prediction heads and return their outputs.
        Returns:
            Dictionary containing raw predictions, distributions, and the next latent state.
        """
        raise NotImplementedError("Implement the CRSSM high-level transition here.")

    def imagine_step(
        self,
        prev_state: Any,
        prev_action: Tensor,
        temperature: float = 1.0,
    ) -> Tuple[Any, Tensor, Dict[str, Tensor]]:
        """
        Predict the next latent state and auxiliary quantities without conditioning on observations.

        TODO:
            * Use prior encoders to sample high-level actions.
            * Roll the stochastic state forward.
        Returns:
            next_state, sampled_action, auxiliary predictions.
        """
        raise NotImplementedError("Implement the imagination rollout.")

    def loss(self, batch: Dict[str, Tensor], outs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Combine supervised/unsupervised objectives for the prediction heads.

        TODO:
            * Compare predicted actions/time/reward against targets in `batch`.
            * Add KL or auxiliary penalties for the action encoders.
        """
        raise NotImplementedError("Implement the training loss calculation.")

    # --------------------------------------------------------------------- #
    # Utilities                                                             #
    # --------------------------------------------------------------------- #
    def _maybe_instantiate(self, cfg: Optional[DictConfig], module_name: str) -> Optional[nn.Module]:
        if cfg is None:
            logger.debug("Skipping %s instantiation (config is None).", module_name)
            return None
        module = hydra.utils.instantiate(cfg)
        if not isinstance(module, nn.Module):
            raise TypeError(f"{module_name} must resolve to nn.Module, got {type(module)}")
        return module

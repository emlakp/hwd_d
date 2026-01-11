import logging
from typing import Any, Dict, List, Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.nn as nn

from lumos.world_models.world_model import WorldModel
from lumos.utils.nn_utils import MLP, init_weights


logger = logging.getLogger(__name__)


class CRSSMHighLevel(WorldModel):

    def __init__(
        self,
        hl_action_encoder: DictConfig,
        hl_heads: DictConfig,
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
        super(CRSSMHighLevel, self).__init__(name=name)
        self.use_gripper_camera = use_gripper_camera


        self.thick_layers = nn.ModuleList([nn.LazyLinear(hidden) for _ in range(self._int_layers)])

        self.hl_act_post = hydra.utils.instantiate(hl_action_encoder)

        self.hl_act_prior = hydra.utils.instantiate(hl_action_encoder)

        self.heads = nn.ModuleDict()

        
        for head_name in ['time', 'action', 'reward']:
            if head_name in hl_heads:
                self.heads[head_name] = hydra.utils.instantiate(hl_heads[head_name])

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.with_proprio = with_proprio


        self.optimizer_config = optimizer
        self.autocast = hydra.utils.instantiate(amp.autocast)
        self.scaler = hydra.utils.instantiate(amp.scaler)

        self.loss_config = loss



        self.batch_metrics = [
            "loss_total",
        ]

        for m in self.modules():
            init_weights(m)

        self.num_val_batches = 0
        self.train_step_history: List[Dict[str, float]] = []
        self.save_hyperparameters()



    def forward(self, inps):

        hl_prior_logits = self.hl_act_prior(inps)
        full_inputs = torch.cat([inps, hl_prior_logits.sample()], dim=-1)
        thick_feats = self.thick_layers(full_inputs)

        stoch = thick_feats.sample()
        outs = {'thick_stoch': stoch}
        outs['thick_act'] = self.thick_heads['action'](readout_inps).sample()
        outs['thick_time'] = self.thick_heads['time'](readout_inps).sample()
        outs['thick_time_mode'] = self.thick_heads['time'](readout_inps).mode()
        outs['reward'] = self.thick_heads['reward'](readout_inps).mode()





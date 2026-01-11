from typing import Dict, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.distributions as D
import torch.nn as nn


class ContextLSTMCore(nn.Module):
    """
    Core module for Context LSTM, handling the sequential processing.
    Similar to ContextRSSMCore but manages the additional LSTM cell state.
    """

    def __init__(self, cell: DictConfig):
        super().__init__()
        self.cell = hydra.utils.instantiate(cell)

    def forward(
        self,
        embeds: Tensor,
        actions: Tensor,
        resets: Tensor,
        in_state: Tuple[Tensor, Tensor, Tensor, Tensor],
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Tensor], Tuple[Tensor, Tensor, Tensor, Tensor]]:
        reset_masks = ~resets
        time_steps = embeds.shape[0]

        h, z, context, cell_state = in_state
        prior_logits, post_logits = [], []
        ctxt_prior_logits_all, ctxt_post_logits_all = [], []
        deter_states, stoch_states, context_states = [], [], []
        gate_traces = []

        for t in range(time_steps):
            (precise_prior_logits, ctxt_prior_logits, posterior_logits), \
            (precise_prior_sample, ctxt_prior_sample, posterior_sample), \
            (context, h, posterior_sample, cell_state), gates = self.cell(
                actions[t], (h, z, context, cell_state), reset_masks[t], embeds[t], temperature=temperature
            )

            prior_logits.append(precise_prior_logits)
            post_logits.append(posterior_logits)
            ctxt_prior_logits_all.append(ctxt_prior_logits)

            deter_states.append(h)
            stoch_states.append(posterior_sample)
            context_states.append(context)
            gate_traces.append(gates)

            z = posterior_sample

        outputs = {
            "precise_priors": torch.stack(prior_logits),
            "posts": torch.stack(post_logits),
            "ctxt_priors": torch.stack(ctxt_prior_logits_all),
            "deter": torch.stack(deter_states),
            "stoch": torch.stack(stoch_states),
            "context": torch.stack(context_states),
            "gates": torch.stack(gate_traces),
        }
        outputs["features"] = self.to_feature(outputs["deter"], outputs["stoch"], outputs["context"])

        final_state = (h.detach(), z.detach(), context.detach(), cell_state.detach())

        return outputs, final_state

    def init_state(self, batch_size):
        return self.cell.init_state(batch_size)

    def to_feature(self, h: Tensor, z: Tensor, context: Tensor) -> Tensor:
        return torch.cat((h, z, context), dim=-1)

    def feature_replace_z(self, features: Tensor, z: Tensor):
        h_dim = self.cell.deter_dim
        z_dim = z.shape[-1]
        context_dim = self.cell.context_dim
        h, _, context = features.split([h_dim, z_dim, context_dim], dim=-1)
        return self.to_feature(h, z, context)

    def zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.zdistr(pp)

    def ctxt_zdistr(self, pp: Tensor) -> D.Distribution:
        return self.cell.ctxt_zdistr(pp)

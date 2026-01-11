from typing import Optional, Tuple

import hydra
from omegaconf import DictConfig
import torch
from torch import Tensor
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F

from lumos.utils.nn_utils import NoNorm


class ContextLSTMCell(nn.Module):
    """
    Context RSSM cell using a simple LSTM instead of GateL0RD for context dynamics.

    This cell replaces the GateL0RDCell with a standard LSTM, removing the gating
    mechanism while maintaining the same overall architecture.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int,
        stoch_dim: int,
        stoch_rank: int,
        context_dim: int,
        hidden_dim: int,
        ensemble: int = 5,
        gru: DictConfig = None,
        layer_norm: bool = True,
        ablate_context: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        self.stoch_rank = stoch_rank
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.ensemble = ensemble
        self.ablate_context = ablate_context

        if gru is not None:
            self.gru = hydra.utils.instantiate(gru)
        else:
            from lumos.world_models.rssm.gru import GRUCellStack

            # Input size should be hidden_dim + context_dim since we concatenate za and context
            self.gru = GRUCellStack(hidden_dim + context_dim, deter_dim, 1)

        self.norm = nn.LayerNorm if layer_norm else NoNorm

        # Initialize state parameters
        self.init_h = nn.Parameter(torch.zeros(self.deter_dim))
        self.init_z = nn.Parameter(torch.zeros(self.stoch_dim * self.stoch_rank))
        self.init_context = nn.Parameter(torch.zeros(self.context_dim))

        # Context dynamics via simple LSTM
        self.context_lstm = nn.LSTMCell(self.hidden_dim, self.context_dim)

        # LSTM cell state (will be managed separately)
        self.init_cell_state = nn.Parameter(torch.zeros(self.context_dim))

        self.z_mlp = nn.Linear(self.stoch_dim * (self.stoch_rank or 1), self.hidden_dim)
        self.a_mlp = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.in_norm = self.norm(self.hidden_dim, eps=1e-3)

        # Precise prior network
        self.prior_mlp_h = nn.Linear(self.deter_dim, self.hidden_dim)
        self.prior_mlp_c = nn.Linear(self.context_dim, self.hidden_dim, bias=False)
        self.prior_norm = self.norm(self.hidden_dim, eps=1e-3)
        self.prior_ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.norm(self.hidden_dim, eps=1e-3),
                    nn.ELU(),
                    nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2)),
                )
                for _ in range(self.ensemble)
            ]
        )

        # Posterior network
        self.post_mlp_h = nn.Linear(self.deter_dim, self.hidden_dim)
        self.post_mlp_c = nn.Linear(self.context_dim, self.hidden_dim, bias=False)
        self.post_mlp_e = nn.Linear(self.embed_dim, self.hidden_dim, bias=False)
        self.post_norm = self.norm(self.hidden_dim, eps=1e-3)
        self.post_mlp = nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2))

        # Context prior head
        self.ctxt_head = nn.Sequential(
            nn.Linear(
                self.stoch_dim * (self.stoch_rank or 1) + self.action_dim + self.context_dim,
                self.hidden_dim,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Context posterior head (observation-conditioned)
        self.ctxt_post_head = nn.Sequential(
            nn.Linear(
                self.stoch_dim * (self.stoch_rank or 1)
                + self.action_dim
                + self.context_dim
                + self.embed_dim,
                self.hidden_dim,
            ),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        self.ctxt_ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.norm(self.hidden_dim, eps=1e-3),
                    nn.ELU(),
                    nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2)),
                )
                for _ in range(self.ensemble)
            ]
        )

        self.ctxt_post_ensemble = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    self.norm(self.hidden_dim, eps=1e-3),
                    nn.ELU(),
                    nn.Linear(self.hidden_dim, self.stoch_dim * (self.stoch_rank or 2)),
                )
                for _ in range(self.ensemble)
            ]
        )

    def init_state(self, batch_size):
        device = next(self.parameters()).device
        h = torch.tile(self.init_h, (batch_size, 1)).to(device)
        z = torch.tile(self.init_z, (batch_size, 1)).to(device)
        context = torch.tile(self.init_context, (batch_size, 1)).to(device)
        cell_state = torch.tile(self.init_cell_state, (batch_size, 1)).to(device)
        # Return (h, z, context, cell_state) - adding cell_state for LSTM
        return h, z, context, cell_state

    def forward(
        self,
        action: Tensor,
        in_state: Tuple[Tensor, Tensor, Tensor, Tensor],
        reset_mask: Optional[Tensor] = None,
        embed: Optional[Tensor] = None,
        temperature: float = 1.0,
    ):
        in_h, in_z, in_context, in_cell_state = in_state
        if reset_mask is not None:
            reset_mask = reset_mask.unsqueeze(-1)
            in_h = in_h * reset_mask
            in_z = in_z * reset_mask
            in_context = in_context * reset_mask
            in_cell_state = in_cell_state * reset_mask

        batch_size = action.shape[0]

        # Process inputs for GRU
        x = self.z_mlp(in_z) + self.a_mlp(action)
        x = self.in_norm(x)
        za = F.elu(x)

        # Context dynamics using LSTM
        context, cell_state = self.context_lstm(za, (in_context, in_cell_state))

        # Ablation: zero out context if ablate_context is True
        if self.ablate_context:
            context = torch.zeros_like(context)

        # Precise dynamics
        gru_input = torch.cat([za, context], dim=-1)
        h = self.gru(gru_input, in_h)

        # Precise prior parameters
        prior_inp = self.prior_mlp_h(h) + self.prior_mlp_c(context)
        prior_inp = self.prior_norm(prior_inp)
        prior_inp = F.elu(prior_inp)
        prior_index = torch.randint(0, self.ensemble, (), device=action.device)
        precise_prior_logits = self.prior_ensemble[prior_index](prior_inp)
        precise_prior_dist = self.zdistr(precise_prior_logits, temperature)
        precise_prior_sample = precise_prior_dist.rsample().reshape(batch_size, -1)

        # Posterior parameters (observation-conditioned when embed provided)
        post_inp = self.post_mlp_h(h) + self.post_mlp_c(context)
        if embed is not None:
            post_inp = post_inp + self.post_mlp_e(embed)
        post_inp = self.post_norm(post_inp)
        post_inp = F.elu(post_inp)
        posterior_logits = self.post_mlp(post_inp)

        posterior_dist = self.zdistr(posterior_logits, temperature)
        posterior_sample = posterior_dist.rsample().reshape(batch_size, -1)

        # Context prior parameters
        ctxt_inp_prior = torch.cat([in_z, action, context], dim=-1)
        ctxt_hidden_prior = self.ctxt_head(ctxt_inp_prior)
        ctxt_prior_index = torch.randint(0, self.ensemble, (), device=action.device)
        ctxt_prior_logits = self.ctxt_ensemble[ctxt_prior_index](ctxt_hidden_prior)

        ctxt_prior_dist = self.zdistr(ctxt_prior_logits, temperature)
        ctxt_prior_sample = ctxt_prior_dist.rsample().reshape(batch_size, -1)

        # Create dummy gates (all ones) for compatibility with ContextRSSM interface
        dummy_gates = torch.ones((batch_size, self.context_dim), device=action.device)

        return (
            (precise_prior_logits, ctxt_prior_logits, posterior_logits),
            (precise_prior_sample, ctxt_prior_sample, posterior_sample),
            (context, h, posterior_sample, cell_state),  # Include cell_state in output
            dummy_gates  # Return dummy gates instead of real gates
        )

    def zdistr(self, pp: Tensor, temperature: float = 1.0) -> D.Distribution:
        logits = pp.reshape(pp.shape[:-1] + (self.stoch_dim, self.stoch_rank)) / temperature
        dist = D.OneHotCategoricalStraightThrough(logits=logits.float())
        dist = D.Independent(dist, 1)
        return dist

    def ctxt_zdistr(self, pp: Tensor, temperature: float = 1.0) -> D.Distribution:
        return self.zdistr(pp, temperature)

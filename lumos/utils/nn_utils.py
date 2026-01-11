from typing import Tuple, Union

import torch
from torch import nn


def get_activation(activation: str):
    if activation is None:
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    else:
        raise NotImplementedError


def st_clamp(x):
    x_clamped = x.clamp(0.0, 1.0)
    # forward = x_clamped, backward dL/dx = dL/d(x) (identity)
    return x + (x_clamped - x).detach()


def init_weights(m):
    if type(m) in {nn.Conv2d, nn.ConvTranspose2d, nn.Linear}:
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)
    if isinstance(m, nn.GRUCell):
        nn.init.xavier_uniform_(m.weight_ih.data)
        nn.init.orthogonal_(m.weight_hh.data)
        nn.init.zeros_(m.bias_ih.data)
        nn.init.zeros_(m.bias_hh.data)


def transpose_collate(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    return {k: torch.transpose(v, 0, 1) for k, v in default_collate(batch).items()}


def transpose_collate_wm(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "robot_obs", "frame"]
    nested_fields = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "state_info": ["robot_obs", "pre_robot_obs", "current_task_ids"],
        "actions": ["rel_actions", "pre_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                if sub_key in value:
                    transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def transpose_collate_hybrid_wm(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "state_obs", "frame"]
    nested_fields = {
        "rgb_obs": ["rgb_static", "rgb_gripper"],
        "state_info": ["robot_obs", "pre_robot_obs"],
        "actions": ["rel_actions", "pre_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                if sub_key in value:
                    transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def transpose_collate_state_wm(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "state_obs", "frame"]
    nested_fields = {
        "state_info": ["robot_obs", "pre_robot_obs"],
        "actions": ["rel_actions", "pre_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                if sub_key in value:
                    transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def transpose_collate_ag(batch):
    """transposes batch and time dimension
    (B, T, ...) -> (T, B, ...)"""
    from torch.utils.data._utils.collate import default_collate

    collated_batch = default_collate(batch)
    transposed_batch = {}

    fields = ["reset", "feature", "zero_feature", "clip_s", "clip_g"]
    nested_fields = {
        "state_info": ["robot_obs"],
        "actions": ["rel_actions"],
    }

    for key, value in collated_batch.items():
        if key in nested_fields:
            transposed_batch[key] = {}
            for sub_key in nested_fields[key]:
                transposed_batch[key][sub_key] = torch.transpose(value[sub_key], 0, 1)
        elif key in fields:
            transposed_batch[key] = torch.transpose(value, 0, 1)
        else:
            transposed_batch[key] = value

    return transposed_batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_batch(x: torch.Tensor, nonbatch_dims=1) -> Tuple[torch.Tensor, torch.Size]:
    # (b1,b2,..., X) => (B, X)
    if nonbatch_dims > 0:
        batch_dim = x.shape[:-nonbatch_dims]
        x = torch.reshape(x, (-1,) + x.shape[-nonbatch_dims:])
        return x, batch_dim
    else:
        batch_dim = x.shape
        x = torch.reshape(x, (-1,))
        return x, batch_dim


def unflatten_batch(x: torch.Tensor, batch_dim: Union[torch.Size, Tuple]) -> torch.Tensor:
    # (B, X) => (b1,b2,..., X)
    x = torch.reshape(x, batch_dim + x.shape[1:])
    return x


class NoNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def get_norm_layer(norm: str):
    """Get normalization layer type.

    Args:
        norm: Type of normalization ('layer', 'batch', 'none')

    Returns:
        Normalization layer class
    """
    if norm == 'none' or norm is None:
        return NoNorm
    elif norm == 'layer':
        return nn.LayerNorm
    elif norm == 'batch':
        return nn.BatchNorm1d
    else:
        raise NotImplementedError(f"Normalization type '{norm}' not implemented")


class MLP(nn.Module):
    """Multi-layer perceptron with configurable output for distributions.

    Similar to TensorFlow MLP but following the pattern from cell.py:
    - Network outputs raw logits/parameters
    - Separate method converts to distributions
    - Clean separation of network and distribution logic

    Args:
        shape: Output shape (int or tuple/list of output dimensions)
        layers: Number of hidden layers
        units: Number of units per hidden layer
        act: Activation function ('elu', 'relu', 'sigmoid', etc.)
        norm: Normalization type ('none', 'layer', 'batch')
        dist: Output type ('mse', 'normal', 'tanh_normal', 'trunc_normal', 'onehot', 'onehot_st', 'binary')
        **out: Additional arguments (e.g., 'unimix' for categorical distributions)

    Example:
        # Create MLP that outputs logits for one-hot categorical
        mlp = MLP(shape=[7], layers=3, units=200, act='elu', norm='none', dist='onehot')
        logits = mlp(features)  # Returns raw logits
        dist = mlp.dist(logits)  # Convert to distribution
        sample = dist.rsample()  # Sample with reparameterization
    """

    def __init__(self, shape, layers, units, act='elu', norm='none', dist='mse', **out):
        super().__init__()
        self._shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self._layers = layers
        self._units = units
        self._norm = norm
        self._act = get_activation(act)
        self._dist = dist
        self._out = out

        # Build hidden layers
        self.hidden_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        norm_class = get_norm_layer(norm)

        for i in range(layers):
            if i == 0:
                # Use LazyLinear for first layer to infer input size
                self.hidden_layers.append(nn.LazyLinear(units))
            else:
                self.hidden_layers.append(nn.Linear(units, units))

            # Add normalization
            if norm != 'none' and norm is not None:
                if norm == 'layer':
                    self.norm_layers.append(norm_class(units, eps=1e-3))
                elif norm == 'batch':
                    self.norm_layers.append(norm_class(units))
                else:
                    self.norm_layers.append(NoNorm())
            else:
                self.norm_layers.append(NoNorm())

        # Output layer - just produces raw parameters/logits
        out_dim = self._get_out_dim()
        self.out_layer = nn.Linear(units, out_dim)

    def _get_out_dim(self):
        """Calculate output dimension based on distribution type."""
        base_dim = int(torch.tensor(self._shape).prod())

        if self._dist == 'mse':
            return base_dim
        elif self._dist in ['normal', 'tanh_normal', 'trunc_normal']:
            # Need both mean and std
            return base_dim * 2
        elif self._dist in ['onehot', 'onehot_st', 'binary']:
            return base_dim
        else:
            raise NotImplementedError(f"Distribution type '{self._dist}' not implemented")

    def forward(self, features):
        """Forward pass - returns raw logits/parameters, not distributions."""
        x = features.float()
        original_shape = x.shape
        x = x.reshape(-1, x.shape[-1])

        # Hidden layers with normalization and activation
        for i in range(self._layers):
            x = self.hidden_layers[i](x)
            x = self.norm_layers[i](x)
            x = self._act(x)

        # Output layer
        x = self.out_layer(x)

        # Restore batch dimensions
        x = x.reshape(original_shape[:-1] + (x.shape[-1],))

        return x

    def dist(self, logits, temperature=1.0):
        """Convert raw network output to a distribution.

        Args:
            logits: Raw output from forward pass
            temperature: Temperature for sampling (default 1.0)

        Returns:
            PyTorch distribution object
        """
        import torch.distributions as D

        if self._dist == 'mse':
            # Deterministic output, no distribution
            return logits.reshape(logits.shape[:-1] + self._shape)

        elif self._dist == 'normal':
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            std = torch.exp(log_std)
            mean = mean.reshape(mean.shape[:-1] + self._shape)
            std = std.reshape(std.shape[:-1] + self._shape)
            return D.Normal(mean, std)

        elif self._dist == 'tanh_normal':
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            std = torch.exp(log_std)
            mean = mean.reshape(mean.shape[:-1] + self._shape)
            std = std.reshape(std.shape[:-1] + self._shape)
            base_dist = D.Normal(mean, std)
            return D.TransformedDistribution(
                base_dist,
                [D.transforms.TanhTransform()]
            )

        elif self._dist == 'trunc_normal':
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            std = torch.exp(log_std)
            mean = mean.reshape(mean.shape[:-1] + self._shape)
            std = std.reshape(std.shape[:-1] + self._shape)
            from lumos.utils.dists import SafeTruncatedNormal
            return SafeTruncatedNormal(mean, std, -1.0, 1.0)

        elif self._dist == 'onehot':
            # Standard one-hot categorical
            logits_scaled = logits / temperature
            logits_shaped = logits_scaled.reshape(logits_scaled.shape[:-1] + self._shape)
            return D.OneHotCategorical(logits=logits_shaped.float())

        elif self._dist == 'onehot_st':
            if len(self._shape) == 2:
                logits_scaled = logits.reshape(logits.shape[:-1] + self._shape) / temperature
                dist = D.OneHotCategoricalStraightThrough(logits=logits_scaled.float())
                return D.Independent(dist, 1)  # Treat as independent categoricals
            else:
                # Single categorical
                logits_scaled = logits / temperature
                logits_shaped = logits_scaled.reshape(logits_scaled.shape[:-1] + self._shape)
                return D.OneHotCategoricalStraightThrough(logits=logits_shaped.float())

        elif self._dist == 'binary':
            logits_shaped = logits.reshape(logits.shape[:-1] + self._shape)
            return D.Bernoulli(logits=logits_shaped.float())

        else:
            raise NotImplementedError(f"Distribution type '{self._dist}' not implemented")

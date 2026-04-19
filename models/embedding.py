"""Small model-building helpers for the Rainbow stack."""

from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn


def build_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    *,
    activation: type[nn.Module] = nn.ReLU,
) -> tuple[nn.Sequential, int]:
    """Build a simple feed-forward MLP torso."""

    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    return nn.Sequential(*layers), last_dim

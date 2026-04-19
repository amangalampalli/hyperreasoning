"""Distributional RL helpers for C51."""

from __future__ import annotations

import torch


def expected_value(probabilities: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Return the expectation of a categorical value distribution."""

    return (probabilities * support).sum(dim=-1)


def categorical_projection(
    *,
    next_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    discounts: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Project the Bellman-updated distribution back onto the fixed support."""

    v_min = support[0]
    v_max = support[-1]
    delta_z = support[1] - support[0]
    target_support = rewards.unsqueeze(1) + discounts.unsqueeze(1) * support.unsqueeze(0) * (~dones).unsqueeze(1)
    target_support = target_support.clamp(v_min, v_max)

    b = (target_support - v_min) / delta_z
    lower = b.floor().long().clamp(0, support.numel() - 1)
    upper = b.ceil().long().clamp(0, support.numel() - 1)

    projection = torch.zeros_like(next_probabilities)
    eq_mask = upper == lower
    lower_weight = (upper.float() - b + eq_mask.float()) * next_probabilities
    upper_weight = (b - lower.float()) * next_probabilities
    projection.scatter_add_(1, lower, lower_weight)
    projection.scatter_add_(1, upper, upper_weight)
    projection = projection / projection.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return projection

from __future__ import annotations

import torch

from rl.categorical import categorical_projection


def test_categorical_projection_preserves_probability_mass() -> None:
    support = torch.linspace(-1.0, 1.0, 5)
    next_probabilities = torch.full((2, 5), 0.2)
    rewards = torch.tensor([0.0, 0.5])
    discounts = torch.tensor([0.9, 0.9])
    dones = torch.tensor([False, True])
    projected = categorical_projection(
        next_probabilities=next_probabilities,
        rewards=rewards,
        discounts=discounts,
        dones=dones,
        support=support,
    )
    assert projected.shape == (2, 5)
    assert torch.allclose(projected.sum(dim=1), torch.ones(2), atol=1e-4)

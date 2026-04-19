"""Distributional dueling Q-network for masked Rainbow control."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import torch
import torch.nn as nn

from models.embedding import build_mlp


@dataclass(slots=True)
class QNetworkConfig:
    input_dim: int
    num_actions: int
    num_atoms: int = 51
    v_min: float = -2.0
    v_max: float = 2.0
    hidden_dims: tuple[int, ...] = (256, 256)


class DuelingCategoricalQNetwork(nn.Module):
    """MLP-based dueling categorical network."""

    def __init__(self, config: QNetworkConfig) -> None:
        super().__init__()
        self.config = config
        self.num_actions = config.num_actions
        self.num_atoms = config.num_atoms
        support = torch.linspace(config.v_min, config.v_max, config.num_atoms)
        self.register_buffer("support", support)
        self.torso, last_dim = build_mlp(config.input_dim, config.hidden_dims)
        self.value_stream = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, config.num_atoms),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, config.num_actions * config.num_atoms),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.torso(obs)
        value = self.value_stream(features).view(-1, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(-1, self.num_actions, self.num_atoms)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def expected_q_values(self, logits: torch.Tensor) -> torch.Tensor:
        pmf = self.probabilities(logits)
        return (pmf * self.support).sum(dim=-1)

    def masked_q_values(self, logits: torch.Tensor, action_mask: torch.Tensor | None) -> torch.Tensor:
        q_values = self.expected_q_values(logits)
        if action_mask is None:
            return q_values
        mask = action_mask.bool()
        if mask.ndim != 2 or mask.shape != q_values.shape:
            raise ValueError(f"Invalid action mask shape {tuple(mask.shape)} for q_values {tuple(q_values.shape)}")
        if not mask.any(dim=1).all():
            raise ValueError("Each batch item must have at least one valid action")
        return q_values.masked_fill(~mask, torch.finfo(q_values.dtype).min)

    def select_action(self, obs: torch.Tensor, action_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        q_values = self.masked_q_values(logits, action_mask)
        actions = torch.argmax(q_values, dim=1)
        return actions, logits

    def action_probabilities(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        probs = self.probabilities(logits)
        return probs[torch.arange(probs.shape[0], device=probs.device), actions]

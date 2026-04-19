"""Project-local Rainbow implementation for masked DSL search control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from models.q_network import DuelingCategoricalQNetwork, QNetworkConfig
from rl.categorical import categorical_projection


@dataclass(slots=True)
class RainbowConfig:
    input_dim: int
    num_actions: int
    num_atoms: int = 51
    v_min: float = -2.0
    v_max: float = 2.0
    hidden_dims: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    target_update_interval: int = 250
    batch_size: int = 128
    max_grad_norm: float = 10.0


@dataclass(slots=True)
class UpdateMetrics:
    loss: float
    mean_q: float
    mean_reward: float
    mean_priority: float


class RainbowAgent:
    """Masked C51 Double DQN with a dueling network."""

    def __init__(self, config: RainbowConfig, *, device: str | torch.device = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        network_config = QNetworkConfig(
            input_dim=config.input_dim,
            num_actions=config.num_actions,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            hidden_dims=config.hidden_dims,
        )
        self.online_network = DuelingCategoricalQNetwork(network_config).to(self.device)
        self.target_network = DuelingCategoricalQNetwork(network_config).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=config.learning_rate)
        self.train_steps = 0

    @property
    def support(self) -> torch.Tensor:
        return self.online_network.support

    def act(self, obs: np.ndarray, action_mask: np.ndarray, *, epsilon: float = 0.0) -> int:
        mask = np.asarray(action_mask, dtype=np.bool_)
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size == 0:
            raise ValueError("No valid actions available")
        if random.random() < epsilon:
            return int(random.choice(valid_indices.tolist()))
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            actions, _ = self.online_network.select_action(obs_tensor, mask_tensor)
        return int(actions.item())

    def update(self, batch) -> tuple[UpdateMetrics, np.ndarray]:
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=self.device)
        truncated = torch.as_tensor(batch.truncated, dtype=torch.bool, device=self.device)
        dones = terminated | truncated
        action_masks = torch.as_tensor(batch.action_masks, dtype=torch.bool, device=self.device)
        next_action_masks = torch.as_tensor(batch.next_action_masks, dtype=torch.bool, device=self.device)
        discounts = torch.as_tensor(batch.discounts, dtype=torch.float32, device=self.device)
        weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device)

        logits = self.online_network(obs)
        chosen_logits = logits[torch.arange(logits.shape[0], device=self.device), actions]

        with torch.no_grad():
            next_online_logits = self.online_network(next_obs)
            bootstrap_mask = next_action_masks.clone()
            invalid_rows = ~bootstrap_mask.any(dim=1)
            bootstrap_mask[invalid_rows, 0] = True
            next_online_q = self.online_network.masked_q_values(next_online_logits, bootstrap_mask)
            next_actions = torch.argmax(next_online_q, dim=1)
            next_target_logits = self.target_network(next_obs)
            next_target_probs = self.target_network.action_probabilities(next_target_logits, next_actions)
            target_distribution = categorical_projection(
                next_probabilities=next_target_probs,
                rewards=rewards,
                discounts=discounts,
                dones=dones,
                support=self.support,
            )

        log_probs = F.log_softmax(chosen_logits, dim=-1)
        per_item_loss = -(target_distribution * log_probs).sum(dim=-1)
        loss = (per_item_loss * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

        priorities = per_item_loss.detach().cpu().numpy()
        expected_q = self.online_network.masked_q_values(logits, action_masks)
        metrics = UpdateMetrics(
            loss=float(loss.item()),
            mean_q=float(expected_q.max(dim=1).values.mean().item()),
            mean_reward=float(rewards.mean().item()),
            mean_priority=float(priorities.mean()),
        )
        return metrics, priorities

    def save(self, path: Path, *, encoder_state: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "online_state_dict": self.online_network.state_dict(),
            "target_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "encoder_state": encoder_state,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, *, device: str | torch.device = "cpu") -> tuple["RainbowAgent", dict[str, Any] | None]:
        payload = torch.load(path, map_location=device)
        agent = cls(RainbowConfig(**payload["config"]), device=device)
        agent.online_network.load_state_dict(payload["online_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.train_steps = int(payload.get("train_steps", 0))
        return agent, payload.get("encoder_state")

    def save_metadata(self, path: Path, extra: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(extra, indent=2, sort_keys=True), encoding="utf-8")

"""Project-local prioritized replay buffer for offline and online training."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data.replay_dataset import EncodedTransition
from rl.priorities import importance_sampling_weights, sample_probabilities, sanitize_priorities


@dataclass(slots=True)
class ReplayBatch:
    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_masks: np.ndarray
    next_action_masks: np.ndarray
    discounts: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


@dataclass(slots=True)
class NStepEncodedStep:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    task_id: str | None
    metadata: dict[str, Any]


class NStepTransitionAccumulator:
    """Online n-step transition builder over encoded step data."""

    def __init__(self, *, n_step: int, gamma: float) -> None:
        self.n_step = n_step
        self.gamma = gamma
        self._queue: deque[NStepEncodedStep] = deque()

    def _build_transition(self, length: int) -> EncodedTransition:
        reward = 0.0
        terminal = False
        truncated = False
        next_obs = self._queue[length - 1].next_obs
        next_action_mask = self._queue[length - 1].next_action_mask
        metadata = dict(self._queue[0].metadata)
        for index in range(length):
            step = self._queue[index]
            reward += (self.gamma**index) * step.reward
            terminal = step.terminated
            truncated = step.truncated
            next_obs = step.next_obs
            next_action_mask = step.next_action_mask
            if terminal or truncated:
                break
        first = self._queue[0]
        return EncodedTransition(
            obs=first.obs,
            action=first.action,
            reward=reward,
            next_obs=next_obs,
            terminated=terminal,
            truncated=truncated,
            action_mask=first.action_mask,
            next_action_mask=next_action_mask,
            discount=self.gamma**length,
            task_id=first.task_id,
            metadata={**metadata, "n_steps": length},
        )

    def push(self, step: NStepEncodedStep) -> list[EncodedTransition]:
        self._queue.append(step)
        ready: list[EncodedTransition] = []
        if step.terminated or step.truncated:
            while self._queue:
                ready.append(self._build_transition(len(self._queue)))
                self._queue.popleft()
            return ready
        if len(self._queue) >= self.n_step:
            ready.append(self._build_transition(self.n_step))
            self._queue.popleft()
        return ready


class PrioritizedReplayBuffer:
    """Simple proportional prioritized replay."""

    def __init__(self, *, capacity: int, obs_dim: int, num_actions: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=np.bool_)
        self.truncated = np.zeros(capacity, dtype=np.bool_)
        self.action_masks = np.zeros((capacity, num_actions), dtype=np.bool_)
        self.next_action_masks = np.zeros((capacity, num_actions), dtype=np.bool_)
        self.discounts = np.ones(capacity, dtype=np.float32)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self._position = 0
        self._size = 0
        self.alpha = alpha

    def __len__(self) -> int:
        return self._size

    def add(self, transition: EncodedTransition, *, priority: float | None = None) -> None:
        idx = self._position
        self.obs[idx] = transition.obs
        self.next_obs[idx] = transition.next_obs
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.terminated[idx] = transition.terminated
        self.truncated[idx] = transition.truncated
        self.action_masks[idx] = transition.action_mask
        self.next_action_masks[idx] = transition.next_action_mask
        self.discounts[idx] = transition.discount
        self.priorities[idx] = float(priority) if priority is not None else float(self.priorities.max(initial=1.0))
        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def extend(self, transitions: list[EncodedTransition]) -> None:
        for transition in transitions:
            self.add(transition)

    def sample(self, batch_size: int, *, beta: float, rng: np.random.Generator | None = None) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("Replay buffer is empty")
        rng = rng or np.random.default_rng()
        current_priorities = sanitize_priorities(self.priorities[: self._size])
        probabilities = sample_probabilities(current_priorities, self.alpha)
        indices = rng.choice(self._size, size=batch_size, replace=self._size < batch_size, p=probabilities)
        weights = importance_sampling_weights(probabilities, indices, beta=beta)
        return ReplayBatch(
            obs=self.obs[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_obs=self.next_obs[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            action_masks=self.action_masks[indices],
            next_action_masks=self.next_action_masks[indices],
            discounts=self.discounts[indices],
            weights=weights,
            indices=indices.astype(np.int64),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = sanitize_priorities(priorities)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            capacity=np.array(self.capacity, dtype=np.int64),
            obs=self.obs,
            next_obs=self.next_obs,
            actions=self.actions,
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            action_masks=self.action_masks,
            next_action_masks=self.next_action_masks,
            discounts=self.discounts,
            priorities=self.priorities,
            position=np.array(self._position, dtype=np.int64),
            size=np.array(self._size, dtype=np.int64),
            alpha=np.array(self.alpha, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "PrioritizedReplayBuffer":
        payload = np.load(path, allow_pickle=False)
        buffer = cls(
            capacity=int(payload["capacity"].item()),
            obs_dim=payload["obs"].shape[1],
            num_actions=payload["action_masks"].shape[1],
            alpha=float(payload["alpha"].item()),
        )
        buffer.obs[:] = payload["obs"]
        buffer.next_obs[:] = payload["next_obs"]
        buffer.actions[:] = payload["actions"]
        buffer.rewards[:] = payload["rewards"]
        buffer.terminated[:] = payload["terminated"]
        buffer.truncated[:] = payload["truncated"]
        buffer.action_masks[:] = payload["action_masks"]
        buffer.next_action_masks[:] = payload["next_action_masks"]
        buffer.discounts[:] = payload["discounts"]
        buffer.priorities[:] = payload["priorities"]
        buffer._position = int(payload["position"].item())
        buffer._size = int(payload["size"].item())
        return buffer

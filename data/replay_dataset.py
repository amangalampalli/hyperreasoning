"""Offline replay dataset helpers and n-step reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np

from data.datasets import TransitionRecord, load_transition_sources, normalize_collected_transition
from env.dsl_env import encode_action_mask
from env.state_encoder import StateEncoder


@dataclass(slots=True)
class EncodedTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    terminated: bool
    truncated: bool
    action_mask: np.ndarray
    next_action_mask: np.ndarray
    discount: float
    task_id: str | None
    metadata: dict[str, Any]


def _episode_transition_files(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/*/episodes/*.json"))


def _load_episode_file(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("transitions", []))


def _build_n_step_episode(episode_rows: list[dict[str, Any]], *, gamma: float, n_step: int) -> list[TransitionRecord]:
    normalized = [normalize_collected_transition(row, gamma=gamma) for row in episode_rows]
    if n_step <= 1:
        return normalized
    rebuilt: list[TransitionRecord] = []
    for start in range(len(normalized)):
        reward = 0.0
        steps = 0
        terminated = False
        truncated = False
        next_obs = normalized[start].next_obs
        next_action_mask = normalized[start].next_action_mask
        metadata = dict(normalized[start].metadata)
        for offset in range(n_step):
            index = start + offset
            if index >= len(normalized):
                break
            current = normalized[index]
            reward += (gamma**offset) * current.reward
            steps += 1
            next_obs = current.next_obs
            next_action_mask = current.next_action_mask
            terminated = current.terminated
            truncated = current.truncated
            if terminated or truncated:
                break
        rebuilt.append(
            TransitionRecord.model_validate(
                {
                    **normalized[start].model_dump(),
                    "reward": reward,
                    "next_obs": next_obs,
                    "terminated": terminated,
                    "truncated": truncated,
                    "next_action_mask": next_action_mask,
                    "discount": gamma**steps,
                    "n_steps": steps,
                    "metadata": {**metadata, "n_step_rebuilt": True},
                }
            )
        )
    return rebuilt


def load_offline_replay_dataset(
    run_dirs: Iterable[Path],
    *,
    gamma: float = 0.99,
    n_step: int = 1,
    allowed_task_ids: set[str] | None = None,
) -> list[TransitionRecord]:
    run_dirs = [Path(path) for path in run_dirs]
    if n_step <= 1:
        transitions = load_transition_sources(run_dirs, gamma=gamma)
        if allowed_task_ids is not None:
            transitions = [transition for transition in transitions if transition.task_id in allowed_task_ids]
        return transitions
    rebuilt: list[TransitionRecord] = []
    for run_dir in run_dirs:
        for episode_file in _episode_transition_files(run_dir):
            episode = _build_n_step_episode(_load_episode_file(episode_file), gamma=gamma, n_step=n_step)
            if allowed_task_ids is not None:
                episode = [transition for transition in episode if transition.task_id in allowed_task_ids]
            rebuilt.extend(episode)
    return rebuilt


def encode_replay_dataset(transitions: Iterable[TransitionRecord], encoder: StateEncoder) -> list[EncodedTransition]:
    feature_dim = encoder.feature_dim
    encoded: list[EncodedTransition] = []
    zero_next = np.zeros(feature_dim, dtype=np.float32)
    zero_mask = encode_action_mask([]).astype(np.bool_)
    for transition in transitions:
        encoded.append(
            EncodedTransition(
                obs=encoder.encode_state(transition.obs),
                action=transition.action_id,
                reward=transition.reward,
                next_obs=zero_next.copy() if transition.next_obs is None else encoder.encode_state(transition.next_obs),
                terminated=transition.terminated,
                truncated=transition.truncated,
                action_mask=np.asarray(transition.action_mask, dtype=np.bool_),
                next_action_mask=zero_mask.copy()
                if transition.next_obs is None
                else np.asarray(transition.next_action_mask, dtype=np.bool_),
                discount=transition.discount,
                task_id=transition.task_id,
                metadata=dict(transition.metadata),
            )
        )
    return encoded

"""Transition schema and dataset loading for offline RL."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import json
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from data.task_manifest import load_task_manifest
from env.dsl_env import ACTION_SPACE, action_name_to_id, encode_action_mask


class TransitionRecord(BaseModel):
    """Canonical offline RL transition."""

    model_config = ConfigDict(extra="forbid")

    obs: dict[str, Any]
    action: str
    action_id: int
    reward: float
    next_obs: dict[str, Any] | None
    terminated: bool
    truncated: bool = False
    action_mask: list[int]
    next_action_mask: list[int]
    task_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    discount: float = 1.0
    n_steps: int = 1


def normalize_collected_transition(raw: dict[str, Any], *, gamma: float = 0.99, n_steps: int = 1) -> TransitionRecord:
    next_obs = raw.get("next_state")
    next_valid_actions = [] if next_obs is None else list(next_obs.get("valid_actions", []))
    return TransitionRecord.model_validate(
        {
            "obs": raw["state"],
            "action": raw["action"],
            "action_id": action_name_to_id(raw["action"]),
            "reward": float(raw["reward"]),
            "next_obs": next_obs,
            "terminated": bool(raw["done"]),
            "truncated": False,
            "action_mask": encode_action_mask(list(raw["valid_actions"])).astype(np.int8).tolist(),
            "next_action_mask": encode_action_mask(next_valid_actions).astype(np.int8).tolist(),
            "task_id": raw["state"].get("task_id"),
            "metadata": dict(raw.get("info", {})),
            "discount": gamma**n_steps,
            "n_steps": n_steps,
        }
    )


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl_transitions(path: Path, *, gamma: float = 0.99) -> list[TransitionRecord]:
    return [normalize_collected_transition(row, gamma=gamma) for row in _read_jsonl(path)]


def load_npz_transitions(path: Path) -> list[TransitionRecord]:
    payload = np.load(path, allow_pickle=True)
    rows = payload["transitions"].tolist()
    return [TransitionRecord.model_validate(row) for row in rows]


def load_parquet_transitions(path: Path) -> list[TransitionRecord]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Parquet support requires pandas/pyarrow") from exc
    frame = pd.read_parquet(path)
    return [TransitionRecord.model_validate(row) for row in frame.to_dict(orient="records")]


def load_transition_file(path: Path, *, gamma: float = 0.99) -> list[TransitionRecord]:
    if path.is_dir():
        return load_transition_file(path / "dataset.jsonl", gamma=gamma)
    if path.suffix == ".jsonl":
        return load_jsonl_transitions(path, gamma=gamma)
    if path.suffix == ".npz":
        return load_npz_transitions(path)
    if path.suffix == ".parquet":
        return load_parquet_transitions(path)
    raise ValueError(f"Unsupported transition file format: {path}")


def load_transition_sources(paths: Iterable[Path], *, gamma: float = 0.99) -> list[TransitionRecord]:
    transitions: list[TransitionRecord] = []
    for path in paths:
        transitions.extend(load_transition_file(path, gamma=gamma))
    return transitions


def action_names() -> tuple[str, ...]:
    return ACTION_SPACE


def load_task_ids_from_manifest(path: Path) -> set[str]:
    """Load task ids from a task manifest file."""

    return {task_dir.name for task_dir in load_task_manifest(path)}

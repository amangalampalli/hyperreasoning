"""Compact numeric state encoder for Rainbow training."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCALAR_KEYS: tuple[str, ...] = (
    "current_depth",
    "path_length",
    "remaining_steps",
    "visible_child_count",
    "hidden_child_count",
    "known_result_count",
    "compile_budget_remaining",
    "current_heuristic_score",
)

BOOLEAN_KEYS: tuple[str, ...] = (
    "current_compile_success",
    "current_visible_test_passed",
    "best_compile_success",
    "best_visible_test_passed",
)


@dataclass(slots=True)
class StateEncoderConfig:
    max_children: int = 4


@dataclass
class StateEncoder:
    """Deterministic one-hot + scalar encoder for environment states."""

    config: StateEncoderConfig = field(default_factory=StateEncoderConfig)
    family_vocab: dict[str, int] = field(default_factory=lambda: {"<unk>": 0})
    strategy_vocab: dict[str, int] = field(default_factory=lambda: {"<unk>": 0})

    def fit(self, states: Iterable[dict[str, Any]]) -> "StateEncoder":
        families: set[str] = set()
        strategies: set[str] = set()
        for state in states:
            family = str(state.get("family", "<unk>"))
            families.add(family)
            strategies.add(str(state.get("current_strategy", "<unk>")))
            for slot in state.get("child_slots", [])[: self.config.max_children]:
                strategies.add(str(slot.get("strategy", "<unk>")))
        self.family_vocab = {"<unk>": 0}
        for family in sorted(families):
            self.family_vocab.setdefault(family, len(self.family_vocab))
        self.strategy_vocab = {"<unk>": 0}
        for strategy in sorted(strategies):
            self.strategy_vocab.setdefault(strategy, len(self.strategy_vocab))
        return self

    @property
    def feature_dim(self) -> int:
        per_child = len(self.strategy_vocab) + 5
        return len(SCALAR_KEYS) + len(BOOLEAN_KEYS) + len(self.family_vocab) + len(self.strategy_vocab) + self.config.max_children * per_child

    def _one_hot(self, vocab: dict[str, int], key: str) -> np.ndarray:
        vec = np.zeros(len(vocab), dtype=np.float32)
        vec[vocab.get(key, 0)] = 1.0
        return vec

    def encode_state(self, state: dict[str, Any]) -> np.ndarray:
        features: list[np.ndarray] = []
        scalar_block = np.array([float(state.get(key, 0.0) or 0.0) for key in SCALAR_KEYS], dtype=np.float32)
        bool_block = np.array([1.0 if state.get(key) is True else 0.0 for key in BOOLEAN_KEYS], dtype=np.float32)
        features.append(scalar_block)
        features.append(bool_block)
        features.append(self._one_hot(self.family_vocab, str(state.get("family", "<unk>"))))
        features.append(self._one_hot(self.strategy_vocab, str(state.get("current_strategy", "<unk>"))))
        child_slots = list(state.get("child_slots", []))
        for index in range(self.config.max_children):
            if index < len(child_slots):
                slot = child_slots[index]
                slot_block = [
                    self._one_hot(self.strategy_vocab, str(slot.get("strategy", "<unk>"))),
                    np.array(
                        [
                            float(slot.get("heuristic_score", 0.0) or 0.0),
                            1.0 if slot.get("compile_known") else 0.0,
                            1.0 if slot.get("compile_success") is True else 0.0,
                            1.0 if slot.get("visible_test_passed") is True else 0.0,
                            1.0,
                        ],
                        dtype=np.float32,
                    ),
                ]
                features.extend(slot_block)
            else:
                features.append(np.zeros(len(self.strategy_vocab), dtype=np.float32))
                features.append(np.zeros(5, dtype=np.float32))
        return np.concatenate(features).astype(np.float32, copy=False)

    def encode_batch(self, states: Iterable[dict[str, Any]]) -> np.ndarray:
        encoded = [self.encode_state(state) for state in states]
        if not encoded:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack(encoded, axis=0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {"max_children": self.config.max_children},
            "family_vocab": self.family_vocab,
            "strategy_vocab": self.strategy_vocab,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StateEncoder":
        config = StateEncoderConfig(max_children=int(payload.get("config", {}).get("max_children", 4)))
        return cls(
            config=config,
            family_vocab={str(k): int(v) for k, v in payload.get("family_vocab", {"<unk>": 0}).items()},
            strategy_vocab={str(k): int(v) for k, v in payload.get("strategy_vocab", {"<unk>": 0}).items()},
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "StateEncoder":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

"""Priority sampling helpers for replay."""

from __future__ import annotations

import numpy as np


def sanitize_priorities(values: np.ndarray, *, eps: float = 1e-5) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.abs(values) + eps


def sample_probabilities(priorities: np.ndarray, alpha: float) -> np.ndarray:
    scaled = priorities.astype(np.float64) ** alpha
    total = scaled.sum()
    if total <= 0.0:
        return np.full_like(scaled, 1.0 / len(scaled), dtype=np.float64)
    return scaled / total


def importance_sampling_weights(
    probabilities: np.ndarray,
    indices: np.ndarray,
    *,
    beta: float,
) -> np.ndarray:
    sample_probs = probabilities[indices]
    weights = (len(probabilities) * sample_probs) ** (-beta)
    weights /= weights.max(initial=1.0)
    return weights.astype(np.float32)

"""Utility helpers shared by task templates and scripts."""

from __future__ import annotations

import json
import random
import textwrap
from pathlib import Path
from typing import Iterable, Mapping, Sequence, TypeVar


T = TypeVar("T")


def build_rng(seed: int, family: str, difficulty: str) -> random.Random:
    """Create a deterministic RNG keyed by family, difficulty, and seed."""

    scoped_seed = hash((family, difficulty, seed)) & 0xFFFFFFFF
    return random.Random(scoped_seed)


def make_task_id(family: str, seed: int) -> str:
    """Generate a stable task identifier."""

    return f"{family}_{seed:04d}"


def choose_variant(rng: random.Random, options: Sequence[T]) -> T:
    """Pick a deterministic option from a non-empty sequence."""

    if not options:
        raise ValueError("Expected at least one option")
    return options[rng.randrange(len(options))]


def choose_many(rng: random.Random, options: Sequence[T], count: int) -> list[T]:
    """Pick a deterministic sample without replacement."""

    if count > len(options):
        raise ValueError("Sample size exceeds option count")
    return rng.sample(list(options), count)


def dedent(text: str) -> str:
    """Normalize triple-quoted Python snippets."""

    return textwrap.dedent(text).strip() + "\n"


def indent(text: str, prefix: str) -> str:
    """Indent a string while preserving trailing newlines."""

    stripped = text.rstrip("\n")
    return textwrap.indent(stripped, prefix) + ("\n" if text.endswith("\n") else "")


def render_test_module(body: str) -> str:
    """Normalize a test module body so pytest can collect it cleanly."""

    normalized_body = textwrap.dedent(body).strip()
    return f"import unittest\n\n{normalized_body}\n"


def render_json(data: Mapping[str, object]) -> str:
    """Serialize a JSON mapping in a stable human-readable form."""

    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def ensure_dir(path: Path) -> None:
    """Create a directory tree if missing."""

    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text after creating parent directories."""

    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")


def parse_mix(mix_text: str) -> dict[str, float]:
    """Parse difficulty mix text like ``medium=0.2,hard=0.8``."""

    entries: dict[str, float] = {}
    for raw_part in mix_text.split(","):
        part = raw_part.strip()
        if not part:
            continue
        name, _, value = part.partition("=")
        if not _:
            raise ValueError(f"Invalid difficulty mix entry: {part!r}")
        entries[name.strip()] = float(value.strip())
    total = sum(entries.values())
    if total <= 0:
        raise ValueError("Difficulty mix must have positive total weight")
    return {name: weight / total for name, weight in entries.items()}


def weighted_choice(rng: random.Random, weighted_items: Mapping[str, float]) -> str:
    """Sample from a normalized weight mapping."""

    roll = rng.random()
    cumulative = 0.0
    last_key = ""
    for key, weight in weighted_items.items():
        cumulative += weight
        last_key = key
        if roll <= cumulative:
            return key
    return last_key


def stable_slug(parts: Iterable[str]) -> str:
    """Join a sequence of slug fragments."""

    return "_".join(part.strip().replace("-", "_") for part in parts if part.strip())

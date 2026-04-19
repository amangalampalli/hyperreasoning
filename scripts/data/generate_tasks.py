#!/usr/bin/env python3
"""Generate procedurally varied coding tasks for the RL/verifier pipeline.

README
------
Generate tasks:
    python3 scripts/data/generate_tasks.py --num-tasks 100 --seed 123
    conda run -n hyperreasoning python scripts/data/generate_tasks.py --num-tasks 100 --seed 123

Generate a subset of families:
    python3 scripts/data/generate_tasks.py --families streaming_parser_reentrancy,async_retry_contract

Run with a custom difficulty mix or output directory:
    python3 scripts/data/generate_tasks.py --difficulty-mix medium=0.2,hard=0.8 --output-dir data/generated_tasks

Generated tasks are written under:
    data/generated_tasks/<difficulty>/<task_id>/

Sanity checks use pytest via:
    conda run -n hyperreasoning python scripts/data/sanity_check_tasks.py --run-visible-tests --run-hidden-tests
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_templates.registry import get_template, list_families
from data.task_templates.utils import parse_mix, weighted_choice
from data.task_templates.writer import write_task


@dataclass(slots=True)
class GenerationResult:
    """Compact result returned from each generation job."""

    family: str
    difficulty: str
    task_id: str
    output_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tasks", type=int, default=100, help="Number of task instances to generate")
    parser.add_argument(
        "--difficulty-mix",
        default="medium=0.2,hard=0.8",
        help="Difficulty sampling weights such as medium=0.2,hard=0.8",
    )
    parser.add_argument(
        "--families",
        default="all",
        help="Comma-separated family list or 'all'",
    )
    parser.add_argument("--seed", type=int, default=123, help="Base RNG seed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/generated_tasks",
        help="Root directory where generated tasks will be written",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use; 1 keeps generation sequential",
    )
    return parser.parse_args()


def resolve_families(raw_families: str) -> list[str]:
    available = list_families()
    if raw_families == "all":
        return available
    requested = [item.strip() for item in raw_families.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown families: {', '.join(unknown)}")
    return requested


def choose_family(base_seed: int, index: int, families: list[str]) -> str:
    rng = random.Random(hash(("family", base_seed, index)) & 0xFFFFFFFF)
    return families[rng.randrange(len(families))]


def choose_difficulty(base_seed: int, index: int, mix: dict[str, float]) -> str:
    rng = random.Random(hash(("difficulty", base_seed, index)) & 0xFFFFFFFF)
    return weighted_choice(rng, mix)


def generate_one(index: int, base_seed: int, families: list[str], mix: dict[str, float], output_dir: str) -> GenerationResult:
    family = choose_family(base_seed, index, families)
    difficulty = choose_difficulty(base_seed, index, mix)
    task_seed = base_seed + index
    template = get_template(family)
    task = template.generate_instance(seed=task_seed, difficulty=difficulty)
    output_path = write_task(task, Path(output_dir))
    return GenerationResult(
        family=task.family,
        difficulty=task.difficulty,
        task_id=task.task_id,
        output_path=str(output_path),
    )


def print_summary(results: list[GenerationResult]) -> None:
    family_counts = Counter(result.family for result in results)
    difficulty_counts = Counter(result.difficulty for result in results)
    pair_counts = Counter((result.family, result.difficulty) for result in results)

    print(f"Generated {len(results)} tasks")
    print("By difficulty:")
    for difficulty, count in sorted(difficulty_counts.items()):
        print(f"  {difficulty}: {count}")
    print("By family:")
    for family, count in sorted(family_counts.items()):
        print(f"  {family}: {count}")
    print("Family x difficulty:")
    for family, difficulty in sorted(pair_counts):
        print(f"  {family} / {difficulty}: {pair_counts[(family, difficulty)]}")


def main() -> int:
    args = parse_args()
    families = resolve_families(args.families)
    mix = parse_mix(args.difficulty_mix)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "medium").mkdir(parents=True, exist_ok=True)
    (args.output_dir / "hard").mkdir(parents=True, exist_ok=True)

    if args.num_tasks < 1:
        raise ValueError("--num-tasks must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.workers == 1:
        results = [
            generate_one(index, args.seed, families, mix, str(args.output_dir))
            for index in range(args.num_tasks)
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    generate_one,
                    range(args.num_tasks),
                    [args.seed] * args.num_tasks,
                    [families] * args.num_tasks,
                    [mix] * args.num_tasks,
                    [str(args.output_dir)] * args.num_tasks,
                )
            )

    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

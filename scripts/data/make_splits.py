#!/usr/bin/env python3
"""Generate deterministic task-level train/eval split manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_manifest import discover_task_dirs, filter_task_dirs, load_task_metadata, shuffle_task_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "data/generated_tasks")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--eval-count", type=int, default=10, help="Total number of held-out eval tasks")
    parser.add_argument("--train-fraction", type=float, default=None, help="Optional fallback fraction of tasks assigned to train")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--families", default=None, help="Optional comma-separated family filter")
    parser.add_argument("--difficulties", default=None, help="Optional comma-separated difficulty filter")
    parser.add_argument("--train-name", default="train_90.txt")
    parser.add_argument("--eval-name", default="eval_10.txt")
    return parser.parse_args()


def _parse_filter_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def _write_manifest(path: Path, task_dirs: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(task_dir.as_posix() for task_dir in task_dirs) + ("\n" if task_dirs else ""), encoding="utf-8")


def _allocate_eval_counts(grouped: dict[str, list[Path]], total_eval_count: int) -> dict[str, int]:
    total_tasks = sum(len(items) for items in grouped.values())
    if total_eval_count <= 0 or total_eval_count >= total_tasks:
        raise SystemExit(f"--eval-count must be in [1, {max(1, total_tasks - 1)}]")

    base_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for key, items in grouped.items():
        exact = total_eval_count * (len(items) / total_tasks)
        count = min(len(items), int(exact))
        base_counts[key] = count
        assigned += count
        remainders.append((exact - count, key))

    remaining = total_eval_count - assigned
    for _, key in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        if base_counts[key] < len(grouped[key]):
            base_counts[key] += 1
            remaining -= 1
    return base_counts


def main() -> int:
    args = parse_args()

    families = _parse_filter_set(args.families)
    difficulties = _parse_filter_set(args.difficulties)
    task_dirs = discover_task_dirs(args.tasks_root)
    task_dirs = filter_task_dirs(task_dirs, families=families, difficulties=difficulties)
    if not task_dirs:
        raise SystemExit("No tasks found for split generation")

    grouped: dict[str, list[Path]] = defaultdict(list)
    for task_dir in task_dirs:
        metadata = load_task_metadata(task_dir) or {}
        grouped[str(metadata.get("difficulty", "unknown"))].append(task_dir)
    for difficulty, items in grouped.items():
        grouped[difficulty] = shuffle_task_dirs(items, seed=args.seed + sum(ord(ch) for ch in difficulty))

    if args.train_fraction is not None:
        if not 0.0 < args.train_fraction < 1.0:
            raise SystemExit("--train-fraction must be in (0, 1)")
        eval_count = len(task_dirs) - int(len(task_dirs) * args.train_fraction)
    else:
        eval_count = args.eval_count

    eval_allocations = _allocate_eval_counts(grouped, eval_count)
    train_tasks: list[Path] = []
    eval_tasks: list[Path] = []
    for difficulty in sorted(grouped):
        items = grouped[difficulty]
        difficulty_eval_count = eval_allocations.get(difficulty, 0)
        eval_tasks.extend(items[:difficulty_eval_count])
        train_tasks.extend(items[difficulty_eval_count:])

    train_tasks = shuffle_task_dirs(train_tasks, seed=args.seed)
    eval_tasks = shuffle_task_dirs(eval_tasks, seed=args.seed + 1)

    train_path = args.out_dir / args.train_name
    eval_path = args.out_dir / args.eval_name
    _write_manifest(train_path, train_tasks)
    _write_manifest(eval_path, eval_tasks)

    by_difficulty = {difficulty: {"total": len(grouped[difficulty]), "eval": eval_allocations.get(difficulty, 0)} for difficulty in sorted(grouped)}
    print(f"total={len(task_dirs)} train={len(train_tasks)} eval={len(eval_tasks)} seed={args.seed}")
    print(f"stratified_by_difficulty={by_difficulty}")
    print(f"train_manifest={train_path}")
    print(f"eval_manifest={eval_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

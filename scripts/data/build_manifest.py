#!/usr/bin/env python3
"""Build a reproducible task manifest from generated tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.task_manifest import discover_task_dirs, filter_task_dirs, shuffle_task_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "data/generated_tasks")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default=None)
    parser.add_argument("--difficulties", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _parse_filter_set(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def main() -> int:
    args = parse_args()
    families = _parse_filter_set(args.families)
    difficulties = _parse_filter_set(args.difficulties)
    task_dirs = discover_task_dirs(args.tasks_root)
    task_dirs = filter_task_dirs(task_dirs, families=families, difficulties=difficulties)
    if args.shuffle:
        task_dirs = shuffle_task_dirs(task_dirs, seed=args.seed)
    if args.limit is not None:
        task_dirs = task_dirs[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(path.as_posix() for path in task_dirs) + ("\n" if task_dirs else ""), encoding="utf-8")
    print(f"Wrote {len(task_dirs)} task paths to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

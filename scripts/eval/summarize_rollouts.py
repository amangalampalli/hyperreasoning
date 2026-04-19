#!/usr/bin/env python3
"""Summarize a completed offline rollout run directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.io import save_json
from data.task_manifest import aggregate_task_summaries, load_task_summary, print_run_summary, write_task_csv


def _top_ranked(task_summaries: list[dict], *, successes: bool, limit: int) -> list[dict]:
    key_fn = lambda item: (
        bool(item.get("had_any_visible_pass")),
        -999.0 if item.get("best_node_score") is None else item["best_node_score"],
    )
    ordered = sorted(task_summaries, key=key_fn, reverse=successes)
    return ordered[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory under artifacts/runs/<run_id>")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--csv-out", type=Path, default=None)
    parser.add_argument("--print-per-family", action="store_true")
    parser.add_argument("--print-per-difficulty", action="store_true")
    parser.add_argument("--top-failures", type=int, default=0)
    parser.add_argument("--top-successes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.run_dir.exists():
        print(f"Run directory does not exist: {args.run_dir}")
        return 1

    task_summaries: list[dict] = []
    for child in sorted(path for path in args.run_dir.iterdir() if path.is_dir()):
        summary = load_task_summary(child)
        if summary is not None:
            task_summaries.append(summary)

    if not task_summaries:
        print(f"No valid task summaries found under {args.run_dir}")
        return 0

    run_id = args.run_dir.name
    config = {}
    run_summary_path = args.run_dir / "run_summary.json"
    if run_summary_path.exists():
        import orjson

        try:
            existing = orjson.loads(run_summary_path.read_bytes())
            config = existing.get("config", {})
        except orjson.JSONDecodeError:
            config = {}

    summary = aggregate_task_summaries(
        task_summaries,
        run_id=run_id,
        config=config,
        tasks_selected=len(task_summaries),
        tasks_processed=len(task_summaries),
        tasks_skipped_resume=0,
        tasks_failed_to_run=0,
    )
    if args.top_successes:
        summary["top_successes"] = _top_ranked(task_summaries, successes=True, limit=args.top_successes)
    if args.top_failures:
        summary["top_failures"] = _top_ranked(task_summaries, successes=False, limit=args.top_failures)
    print_run_summary(
        summary,
        print_per_family=args.print_per_family,
        print_per_difficulty=args.print_per_difficulty,
        top_failures=args.top_failures,
        top_successes=args.top_successes,
    )

    if args.json_out is not None:
        save_json(args.json_out, summary)
        print(f"Saved JSON summary to {args.json_out}")
    if args.csv_out is not None:
        write_task_csv(task_summaries, args.csv_out)
        print(f"Saved task CSV to {args.csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

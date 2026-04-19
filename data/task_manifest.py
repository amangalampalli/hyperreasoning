"""Task discovery, manifest loading, filtering, and summary aggregation."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
import random
from typing import Any

import orjson


def load_task_manifest(path: Path) -> list[Path]:
    """Load one task directory path per line from a manifest file."""

    task_dirs: list[Path] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        task_dirs.append(Path(line))
    return task_dirs


def discover_task_dirs(tasks_root: Path) -> list[Path]:
    """Walk a task root and return directories that contain task.json.

    Supports either:
    - a root containing difficulty subdirectories, e.g. data/generated_tasks/
    - a root that is itself a difficulty directory, e.g. data/generated_tasks/hard/
    """

    task_dirs: list[Path] = []
    direct_children = [path for path in sorted(tasks_root.iterdir()) if path.is_dir()]

    if any((child / "task.json").exists() for child in direct_children):
        return [child for child in direct_children if (child / "task.json").exists()]

    for difficulty_dir in sorted(tasks_root.iterdir()):
        if not difficulty_dir.is_dir():
            continue
        for task_dir in sorted(difficulty_dir.iterdir()):
            if task_dir.is_dir() and (task_dir / "task.json").exists():
                task_dirs.append(task_dir)
    return task_dirs


def load_task_metadata(task_dir: Path) -> dict[str, Any] | None:
    """Read task.json safely and return parsed metadata."""

    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        print(f"Warning: skipping {task_dir} because task.json is missing")
        return None
    try:
        return orjson.loads(task_json_path.read_bytes())
    except orjson.JSONDecodeError:
        print(f"Warning: skipping {task_dir} because task.json is invalid")
        return None


def _is_valid_task_dir(task_dir: Path, metadata: dict[str, Any]) -> bool:
    if not isinstance(metadata.get("task_id"), str):
        print(f"Warning: skipping {task_dir} because task_id is missing")
        return False
    target_files = metadata.get("target_files") or metadata.get("metadata", {}).get("target_files", [])
    if not isinstance(target_files, list) or not any(isinstance(item, str) for item in target_files):
        print(f"Warning: skipping {task_dir} because no editable target files were found")
        return False
    return True


def filter_task_dirs(
    task_dirs: list[Path],
    *,
    families: set[str] | None = None,
    difficulties: set[str] | None = None,
) -> list[Path]:
    """Filter task directories by family/difficulty and validity."""

    results: list[Path] = []
    for task_dir in task_dirs:
        metadata = load_task_metadata(task_dir)
        if metadata is None or not _is_valid_task_dir(task_dir, metadata):
            continue
        family = metadata.get("family")
        difficulty = metadata.get("difficulty")
        if families and family not in families:
            continue
        if difficulties and difficulty not in difficulties:
            continue
        results.append(task_dir)
    return results


def shuffle_task_dirs(task_dirs: list[Path], *, seed: int) -> list[Path]:
    """Deterministically shuffle task directories."""

    items = list(task_dirs)
    random.Random(seed).shuffle(items)
    return items


def is_task_completed(run_dir: Path, task_id: str) -> bool:
    """Determine whether a task already has a completed rollout artifact."""

    summary_path = run_dir / task_id / "summary.json"
    if not summary_path.exists():
        return False
    try:
        payload = orjson.loads(summary_path.read_bytes())
    except orjson.JSONDecodeError:
        return False
    return bool(payload.get("completed", False))


def load_task_summary(task_run_dir: Path) -> dict[str, Any] | None:
    """Load one per-task summary from a rollout directory."""

    summary_path = task_run_dir / "summary.json"
    if summary_path.exists():
        try:
            payload = orjson.loads(summary_path.read_bytes())
            if isinstance(payload, dict):
                return payload
        except orjson.JSONDecodeError:
            print(f"Warning: invalid summary.json in {task_run_dir}")
            return None

    tree_path = task_run_dir / "tree.json"
    if tree_path.exists():
        try:
            payload = orjson.loads(tree_path.read_bytes())
            if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
                return payload["summary"]
        except orjson.JSONDecodeError:
            print(f"Warning: invalid tree.json in {task_run_dir}")
            return None
    print(f"Warning: no task summary found in {task_run_dir}")
    return None


def _aggregate_breakdown(task_summaries: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in task_summaries:
        group_value = summary.get(key) or "unknown"
        grouped.setdefault(str(group_value), []).append(summary)

    breakdown: dict[str, dict[str, Any]] = {}
    for group_value, items in grouped.items():
        best_scores = [item.get("best_node_score") for item in items if item.get("best_node_score") is not None]
        breakdown[group_value] = {
            "tasks": len(items),
            "compile_success": sum(1 for item in items if item.get("had_any_compile_success")),
            "visible_pass": sum(1 for item in items if item.get("had_any_visible_pass")),
            "avg_best_score": (sum(best_scores) / len(best_scores)) if best_scores else None,
        }
    return breakdown


def aggregate_task_summaries(
    task_summaries: list[dict[str, Any]],
    *,
    run_id: str,
    config: dict[str, Any],
    tasks_selected: int,
    tasks_processed: int,
    tasks_skipped_resume: int,
    tasks_failed_to_run: int,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Aggregate per-task summaries into a run-level summary."""

    total_nodes = sum(item.get("num_nodes", 0) for item in task_summaries)
    total_compiled_nodes = sum(item.get("num_compiled_nodes", 0) for item in task_summaries)
    total_passing_nodes = sum(item.get("num_passing_nodes", 0) for item in task_summaries)
    tasks_with_compile_success = sum(1 for item in task_summaries if item.get("had_any_compile_success"))
    tasks_with_visible_pass = sum(1 for item in task_summaries if item.get("had_any_visible_pass"))
    best_scores = [item.get("best_node_score") for item in task_summaries if item.get("best_node_score") is not None]

    return {
        "run_id": run_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "config": config,
        "tasks_selected": tasks_selected,
        "tasks_processed": tasks_processed,
        "tasks_skipped_resume": tasks_skipped_resume,
        "tasks_failed_to_run": tasks_failed_to_run,
        "tasks_with_compile_success": tasks_with_compile_success,
        "tasks_with_visible_pass": tasks_with_visible_pass,
        "total_nodes": total_nodes,
        "total_compiled_nodes": total_compiled_nodes,
        "total_passing_nodes": total_passing_nodes,
        "avg_nodes_per_task": (total_nodes / tasks_processed) if tasks_processed else 0.0,
        "avg_depth_reached": (
            sum(item.get("max_depth_reached", 0) for item in task_summaries) / tasks_processed
            if tasks_processed
            else 0.0
        ),
        "compile_success_rate": (tasks_with_compile_success / tasks_processed) if tasks_processed else 0.0,
        "visible_pass_rate": (tasks_with_visible_pass / tasks_processed) if tasks_processed else 0.0,
        "avg_best_score": (sum(best_scores) / len(best_scores)) if best_scores else None,
        "families_breakdown": _aggregate_breakdown(task_summaries, "family"),
        "difficulty_breakdown": _aggregate_breakdown(task_summaries, "difficulty"),
        "per_task_index": [
            {
                "task_id": item.get("task_id"),
                "family": item.get("family"),
                "difficulty": item.get("difficulty"),
                "had_any_compile_success": item.get("had_any_compile_success"),
                "had_any_visible_pass": item.get("had_any_visible_pass"),
                "best_node_score": item.get("best_node_score"),
                "best_node_id": item.get("best_node_id"),
            }
            for item in task_summaries
        ],
    }


def _format_breakdown_table(title: str, breakdown: dict[str, dict[str, Any]]) -> list[str]:
    lines = [title]
    lines.append(f"{'group':28} {'tasks':>6} {'compile_success':>16} {'visible_pass':>14} {'avg_best_score':>15}")
    for group, metrics in sorted(breakdown.items()):
        avg_best_score = metrics["avg_best_score"]
        avg_text = "-" if avg_best_score is None else f"{avg_best_score:.2f}"
        lines.append(
            f"{group:28} {metrics['tasks']:>6} {metrics['compile_success']:>16} "
            f"{metrics['visible_pass']:>14} {avg_text:>15}"
        )
    return lines


def print_run_summary(
    summary: dict[str, Any],
    *,
    print_per_family: bool = False,
    print_per_difficulty: bool = False,
    top_failures: int = 0,
    top_successes: int = 0,
) -> None:
    """Print a readable aggregate run summary."""

    print(f"Run: {summary['run_id']}")
    print(f"Tasks selected: {summary['tasks_selected']}")
    print(f"Tasks processed: {summary['tasks_processed']}")
    print(f"Tasks skipped by resume: {summary['tasks_skipped_resume']}")
    print(f"Tasks failed to run: {summary['tasks_failed_to_run']}")
    print(f"Tasks with compile success: {summary['tasks_with_compile_success']}")
    print(f"Tasks with visible pass: {summary['tasks_with_visible_pass']}")
    print(f"Total nodes: {summary['total_nodes']}")
    print(f"Total compiled nodes: {summary['total_compiled_nodes']}")
    print(f"Total passing nodes: {summary['total_passing_nodes']}")
    print(f"Avg nodes/task: {summary['avg_nodes_per_task']:.2f}")
    print(f"Avg depth reached: {summary['avg_depth_reached']:.2f}")
    print(f"Compile success rate: {summary['compile_success_rate']:.2f}")
    print(f"Visible pass rate: {summary['visible_pass_rate']:.2f}")
    if print_per_family:
        print()
        print("\n".join(_format_breakdown_table("Per-family:", summary["families_breakdown"])))
    if print_per_difficulty:
        print()
        print("\n".join(_format_breakdown_table("Per-difficulty:", summary["difficulty_breakdown"])))

    ranked = summary.get("per_task_index", [])
    if top_successes:
        successes = sorted(
            ranked,
            key=lambda item: (
                bool(item.get("had_any_visible_pass")),
                -999.0 if item.get("best_node_score") is None else item["best_node_score"],
            ),
            reverse=True,
        )[:top_successes]
        print("\nTop successes:")
        for item in successes:
            print(
                f"- {item['task_id']} family={item['family']} difficulty={item['difficulty']} "
                f"visible_pass={item['had_any_visible_pass']} best_score={item['best_node_score']}"
            )
    if top_failures:
        failures = sorted(
            ranked,
            key=lambda item: (
                bool(item.get("had_any_visible_pass")),
                999.0 if item.get("best_node_score") is None else item["best_node_score"],
            ),
        )[:top_failures]
        print("\nTop failures:")
        for item in failures:
            print(
                f"- {item['task_id']} family={item['family']} difficulty={item['difficulty']} "
                f"visible_pass={item['had_any_visible_pass']} best_score={item['best_node_score']}"
            )


def write_task_csv(task_summaries: list[dict[str, Any]], path: Path) -> None:
    """Write one CSV row per task summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "family",
        "difficulty",
        "task_dir",
        "num_nodes",
        "num_compiled_nodes",
        "num_compile_success",
        "num_passing_nodes",
        "max_depth_reached",
        "had_any_compile_success",
        "had_any_visible_pass",
        "best_node_score",
        "best_node_id",
        "num_expanded_nodes",
        "avg_compile_latency_s",
        "avg_test_latency_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in task_summaries:
            writer.writerow({field: summary.get(field) for field in fieldnames})

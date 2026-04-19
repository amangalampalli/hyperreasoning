#!/usr/bin/env python3
"""Recompute task comparisons and find where Rainbow beats baselines."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
from typing import Any
from datetime import datetime

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_TASK_MANIFEST = ROOT / "data/splits/eval_10.txt"
DEFAULT_CHECKPOINT = ROOT / "artifacts/models/rainbow_offline_v1_1776237078/best.pt"
DEFAULT_LOCAL_OUTPUT_DIR = ROOT / ".hyper/rainbow_edge_scans"
DEFAULT_COMPETITORS = ("heuristic", "one_shot", "random")
EFFICIENCY_FIELDS = ("llm_total_tokens", "elapsed_time_ms", "branches_explored")
_WORKER_RAINBOW_AGENT: Any | None = None
_WORKER_RAINBOW_ENCODER: Any | None = None


def canonical_method_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "rainbow": "rainbow",
        "heuristic": "heuristic",
        "random": "random",
        "oneshot": "one_shot",
        "one_shot": "one_shot",
    }
    return aliases.get(normalized)


def _init_parallel_worker(checkpoint: Path) -> None:
    global _WORKER_RAINBOW_AGENT, _WORKER_RAINBOW_ENCODER

    from scripts.eval.run_eval_pipeline import load_rainbow_artifacts

    _WORKER_RAINBOW_AGENT, _WORKER_RAINBOW_ENCODER = load_rainbow_artifacts(checkpoint)


def _evaluate_task_method_worker(
    index: int,
    task: Any,
    method: str,
    args: argparse.Namespace,
) -> tuple[int, dict[str, Any]]:
    from scripts.eval.run_eval_pipeline import evaluate_task_method

    try:
        record = evaluate_task_method(
            task,
            method=method,
            args=args,
            rainbow_agent=_WORKER_RAINBOW_AGENT,
            rainbow_encoder=_WORKER_RAINBOW_ENCODER,
        )
    except Exception as exc:  # pragma: no cover - exercised only in live scan failures
        record = {
            "kind": "task_eval_error_v1",
            "task_id": task.task_id,
            "family": task.family,
            "method": method,
            "seed": args.seed,
            "error": str(exc),
        }
    return index, record


def evaluate_records(args: argparse.Namespace, *, records_output: Path | None = None) -> list[dict[str, Any]]:
    """Run fresh comparisons directly from task workspaces."""

    from data.task_store import TaskStore
    from llm.prompt_utils import load_task_context
    from scripts.eval.run_eval_pipeline import evaluate_task_method, load_rainbow_artifacts

    methods = ("rainbow", *args.competitors)
    if args.task_dirs:
        task_contexts = [load_task_context(path) for path in args.task_dirs]
        if args.num_tasks is not None:
            task_contexts = task_contexts[: args.num_tasks]
    else:
        task_contexts = list(TaskStore.from_manifest(args.task_manifest, limit=args.num_tasks).iter_contexts())

    run_plan = [(task, method) for task in task_contexts for method in methods]
    progress = None
    if not args.no_progress:
        progress = tqdm(total=len(run_plan), desc="Rainbow edge scan", unit="run", dynamic_ncols=True)

    if args.jobs > 1:
        records_by_index: list[dict[str, Any] | None] = [None] * len(run_plan)
        written_indexes: set[int] = set()
        try:
            with ProcessPoolExecutor(
                max_workers=args.jobs,
                initializer=_init_parallel_worker,
                initargs=(args.checkpoint,),
            ) as executor:
                futures = [
                    executor.submit(_evaluate_task_method_worker, index, task, method, args)
                    for index, (task, method) in enumerate(run_plan)
                ]
                for future in as_completed(futures):
                    index, record = future.result()
                    records_by_index[index] = record
                    if records_output is not None:
                        write_records_snapshot(records_output, records_by_index)
                        written_indexes.add(index)
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            task=record.get("task_id", "-"),
                            method=record.get("method", "-"),
                            solved=record.get("solved", "-"),
                            refresh=False,
                        )
        finally:
            if progress is not None:
                progress.close()
        return [record for record in records_by_index if record is not None]

    rainbow_agent, rainbow_encoder = load_rainbow_artifacts(args.checkpoint)
    records: list[dict[str, Any]] = []
    if records_output is not None:
        records_output.parent.mkdir(parents=True, exist_ok=True)
        records_output.write_text("", encoding="utf-8")
    try:
        for task, method in run_plan:
            if progress is not None:
                progress.set_postfix(task=task.task_id, method=method, refresh=False)
            try:
                record = evaluate_task_method(
                    task,
                    method=method,
                    args=args,
                    rainbow_agent=rainbow_agent,
                    rainbow_encoder=rainbow_encoder,
                )
            except Exception as exc:  # pragma: no cover - exercised only in live scan failures
                record = {
                    "kind": "task_eval_error_v1",
                    "task_id": task.task_id,
                    "family": task.family,
                    "method": method,
                    "seed": args.seed,
                    "error": str(exc),
                }
            records.append(record)
            if records_output is not None:
                with records_output.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    handle.flush()
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return records


def write_records_snapshot(path: Path, records_by_index: list[dict[str, Any] | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records_by_index if record is not None),
        encoding="utf-8",
    )


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") != "task_eval_record_v1":
            continue
        method = canonical_method_name(row.get("method") or row.get("policy"))
        task_id = row.get("task_id")
        if method is None or not isinstance(task_id, str):
            continue
        normalized = dict(row)
        normalized["method"] = method
        by_task.setdefault(task_id, {})[method] = normalized
    return by_task


def find_rainbow_edges(
    rows: list[dict[str, Any]],
    *,
    competitors: tuple[str, ...] = DEFAULT_COMPETITORS,
) -> dict[str, list[dict[str, Any]]]:
    by_task = index_rows(rows)
    strict_solve_wins: list[dict[str, Any]] = []
    any_solve_wins: list[dict[str, Any]] = []
    test_fraction_wins: list[dict[str, Any]] = []
    quality_tie_efficiency_wins: list[dict[str, Any]] = []

    for task_id, task_rows in sorted(by_task.items()):
        rainbow = task_rows.get("rainbow")
        if rainbow is None:
            continue
        present_competitors = tuple(method for method in competitors if method in task_rows)
        if not present_competitors:
            continue
        comparison = build_comparison(task_id, rainbow, {method: task_rows[method] for method in present_competitors})
        solve_better = [
            method
            for method in present_competitors
            if numeric(rainbow.get("solved"), default=0.0) > numeric(task_rows[method].get("solved"), default=0.0)
        ]
        fraction_better = [
            method
            for method in present_competitors
            if numeric(rainbow.get("fraction_tests_passed"), default=0.0)
            > numeric(task_rows[method].get("fraction_tests_passed"), default=0.0)
        ]
        efficiency_better = [
            method for method in present_competitors if quality_tied_and_more_efficient(rainbow, task_rows[method])
        ]

        if len(solve_better) == len(present_competitors):
            strict_solve_wins.append({**comparison, "beaten_methods": list(solve_better)})
        if solve_better:
            any_solve_wins.append({**comparison, "beaten_methods": list(solve_better)})
        if len(fraction_better) == len(present_competitors):
            test_fraction_wins.append({**comparison, "beaten_methods": list(fraction_better)})
        if len(efficiency_better) == len(present_competitors):
            quality_tie_efficiency_wins.append({**comparison, "beaten_methods": list(efficiency_better)})

    return {
        "strict_solve_wins": strict_solve_wins,
        "any_solve_wins": any_solve_wins,
        "test_fraction_wins": test_fraction_wins,
        "quality_tie_efficiency_wins": quality_tie_efficiency_wins,
    }


def build_comparison(
    task_id: str,
    rainbow: dict[str, Any],
    competitors: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "family": rainbow.get("family"),
        "rainbow": summarize_row(rainbow),
        "competitors": {method: summarize_row(row) for method, row in sorted(competitors.items())},
    }


def summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "solved": int(numeric(row.get("solved"), default=0.0)),
        "tests_passed": row.get("tests_passed"),
        "tests_total": row.get("tests_total"),
        "fraction_tests_passed": row.get("fraction_tests_passed"),
        "visible_test_passed": row.get("visible_test_passed"),
        "hidden_test_passed": row.get("hidden_test_passed"),
        "elapsed_time_ms": row.get("elapsed_time_ms"),
        "llm_total_tokens": row.get("llm_total_tokens"),
        "branches_explored": row.get("branches_explored"),
        "steps_to_success": row.get("steps_to_success"),
    }


def quality_tied_and_more_efficient(rainbow: dict[str, Any], competitor: dict[str, Any]) -> bool:
    if numeric(rainbow.get("solved"), default=0.0) != numeric(competitor.get("solved"), default=0.0):
        return False
    if numeric(rainbow.get("fraction_tests_passed"), default=-1.0) != numeric(
        competitor.get("fraction_tests_passed"), default=-2.0
    ):
        return False
    available = [
        field
        for field in EFFICIENCY_FIELDS
        if rainbow.get(field) is not None and competitor.get(field) is not None
    ]
    if not available:
        return False
    return any(numeric(rainbow.get(field)) < numeric(competitor.get(field)) for field in available)


def numeric(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    return default


def render_markdown(report: dict[str, list[dict[str, Any]]], *, competitors: tuple[str, ...]) -> str:
    lines = [
        "# Rainbow Fresh Edge Scan",
        "",
        "Source: freshly recomputed task/method records from task workspaces.",
        f"Competitors: {', '.join(competitors)}",
        "",
    ]
    sections = [
        ("Rainbow solved, all selected competitors failed", "strict_solve_wins"),
        ("Rainbow solved, at least one selected competitor failed", "any_solve_wins"),
        ("Rainbow passed a higher test fraction than all selected competitors", "test_fraction_wins"),
        ("Same quality, Rainbow more efficient than all selected competitors", "quality_tie_efficiency_wins"),
    ]
    for title, key in sections:
        items = report[key]
        lines += [f"## {title}", "", f"Count: {len(items)}", ""]
        if not items:
            lines += ["No tasks found.", ""]
            continue
        lines += [
            "| Task | Family | Rainbow | Competitors |",
            "| --- | --- | --- | --- |",
        ]
        for item in items:
            lines.append(
                "| {task} | {family} | {rainbow} | {competitors} |".format(
                    task=item["task_id"],
                    family=item.get("family") or "-",
                    rainbow=compact_summary(item["rainbow"]),
                    competitors="<br>".join(
                        f"{method}: {compact_summary(summary)}"
                        for method, summary in item["competitors"].items()
                    ),
                )
            )
        lines.append("")
    return "\n".join(lines)


def compact_summary(summary: dict[str, Any]) -> str:
    tests = f"{summary.get('tests_passed')}/{summary.get('tests_total')}"
    time_ms = summary.get("elapsed_time_ms")
    time_text = "-" if time_ms is None else f"{float(time_ms):.0f}ms"
    return (
        f"solved={summary.get('solved')}, tests={tests}, "
        f"frac={summary.get('fraction_tests_passed')}, tokens={summary.get('llm_total_tokens')}, "
        f"time={time_text}"
    )


def render_csv(report: dict[str, list[dict[str, Any]]]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "category",
            "task_id",
            "family",
            "beaten_methods",
            "rainbow_solved",
            "rainbow_tests",
            "rainbow_fraction",
            "rainbow_tokens",
            "rainbow_time_ms",
        ],
    )
    writer.writeheader()
    for category, items in report.items():
        for item in items:
            rainbow = item["rainbow"]
            writer.writerow(
                {
                    "category": category,
                    "task_id": item["task_id"],
                    "family": item.get("family") or "",
                    "beaten_methods": " ".join(item.get("beaten_methods") or []),
                    "rainbow_solved": rainbow.get("solved"),
                    "rainbow_tests": f"{rainbow.get('tests_passed')}/{rainbow.get('tests_total')}",
                    "rainbow_fraction": rainbow.get("fraction_tests_passed"),
                    "rainbow_tokens": rainbow.get("llm_total_tokens"),
                    "rainbow_time_ms": rainbow.get("elapsed_time_ms"),
                }
            )
    return buffer.getvalue()


def default_records_output_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_LOCAL_OUTPUT_DIR / f"fresh_eval_records_{timestamp}.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--task-dirs", nargs="*", type=Path, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--competitors", nargs="+", default=list(DEFAULT_COMPETITORS))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-verified-plans-per-task", type=int, default=1)
    parser.add_argument("--proposal-source", choices=["heuristic", "llm", "hybrid"], default="heuristic")
    parser.add_argument("--allow-full-file-fallback", action="store_true")
    parser.add_argument("--compiler-temp", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel worker processes. Default is 1; increase if your LLM server can handle concurrent requests.",
    )
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--format", choices=["markdown", "json", "csv"], default="markdown")
    parser.add_argument("--output", type=Path, default=None, help="Optional report output path.")
    parser.add_argument(
        "--records-output",
        type=Path,
        default=None,
        help="Path for freshly computed records. Defaults to a timestamped file under .hyper/rainbow_edge_scans/.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    competitors = tuple(
        method
        for competitor_name in args.competitors
        if (method := canonical_method_name(competitor_name)) is not None and method != "rainbow"
    )
    if not competitors:
        raise SystemExit("At least one non-Rainbow competitor is required.")
    args.competitors = competitors
    args.jobs = max(1, int(args.jobs))
    records_output = args.records_output or default_records_output_path()
    rows = evaluate_records(args, records_output=records_output)
    print(f"fresh_records={records_output}")
    report = find_rainbow_edges(rows, competitors=competitors)
    if args.format == "json":
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    elif args.format == "csv":
        rendered = render_csv(report)
    else:
        rendered = render_markdown(report, competitors=competitors)
    if args.output is None:
        print(rendered, end="" if rendered.endswith("\n") else "\n")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

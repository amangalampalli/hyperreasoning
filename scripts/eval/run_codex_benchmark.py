#!/usr/bin/env python3
"""Run a separate Codex-vs-Rainbow benchmark on the eval split."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval.codex_benchmark_lib import (
    CODEX_METHODS,
    CODEX_TIERS,
    PAIRED_HEADERS,
    RAW_HEADERS,
    SUMMARY_HEADERS,
    build_codex_command,
    build_codex_prompt,
    build_results_report,
    compute_paired_deltas,
    compute_summary_metrics,
    copy_task_workspace,
    extract_usage_fields,
    load_codex_task_contexts,
    load_existing_codex_records,
    load_rainbow_baseline_rows,
    parse_codex_jsonl_output,
    plot_pairwise_delta,
    plot_summary_bars,
    plot_token_breakdown,
    summarize_codex_events,
    run_external_tests,
    write_csv,
    write_raw_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "data/splits/eval_10.txt")
    parser.add_argument("--rainbow-raw-input", type=Path, default=ROOT / "docs/raw_eval_results.jsonl")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs/codex_benchmark")
    parser.add_argument("--codex-raw-output", type=Path, default=None)
    parser.add_argument("--codex-raw-input", type=Path, default=None)
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=CODEX_METHODS, default=list(CODEX_METHODS))
    parser.add_argument("--codex-model", default="gpt-5.4")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--codex-timeout-s", type=float, default=1800.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of Codex task/method runs to execute concurrently.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def _warn(message: str) -> None:
    print(f"[codex-benchmark] {message}", file=sys.stderr)


def _tier_for_method(method: str):
    return next(tier for tier in CODEX_TIERS if tier.method == method)


def _run_codex_method_for_task(task: Any, method: str, args: argparse.Namespace) -> dict[str, Any]:
    workspace = copy_task_workspace(task)
    output_last_message = workspace / ".codex_last_message.txt"
    prompt = build_codex_prompt(task)
    tier = _tier_for_method(method)
    command = build_codex_command(
        workspace=workspace,
        prompt=prompt,
        tier=tier,
        output_last_message=output_last_message,
        model=args.codex_model,
    )

    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=args.codex_timeout_s,
            check=False,
        )
        codex_exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        codex_exit_code = 124
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {args.codex_timeout_s} seconds"
    elapsed_time_ms = (time.perf_counter() - started) * 1000.0

    events, warnings = parse_codex_jsonl_output(stdout)
    if stderr.strip():
        warnings.append(stderr.strip())
    events_summary, usage_payload = summarize_codex_events(events, warnings)
    usage_fields = extract_usage_fields(usage_payload)
    codex_last_message = output_last_message.read_text(encoding="utf-8").strip() if output_last_message.exists() else None

    score = run_external_tests(
        task,
        workspace,
        python_bin=args.python_bin,
        timeout_s=args.timeout_s,
    )
    shutil.rmtree(workspace.parent, ignore_errors=True)

    return {
        "kind": "task_eval_record_v1",
        "task_id": task.task_id,
        "method": method,
        "solved": score["solved"],
        "tests_passed": score["tests_passed"],
        "tests_total": score["tests_total"],
        "fraction_tests_passed": score["fraction_tests_passed"],
        "elapsed_time_ms": elapsed_time_ms,
        "llm_input_tokens": usage_fields["llm_input_tokens"],
        "llm_output_tokens": usage_fields["llm_output_tokens"],
        "llm_total_tokens": usage_fields["llm_total_tokens"],
        "llm_reasoning_tokens": usage_fields["llm_reasoning_tokens"],
        "llm_execution_tokens": usage_fields["llm_execution_tokens"],
        "llm_usage_payload": usage_fields["llm_usage_payload"],
        "branches_explored": None,
        "steps_to_success": None,
        "seed": args.seed,
        "codex_exit_code": codex_exit_code,
        "codex_last_message": codex_last_message,
        "codex_json_events_summary": events_summary,
        "visible_test_passed": score["visible_test_passed"],
        "hidden_test_passed": score["hidden_test_passed"],
        "visible_test_stdout": score["visible_test_stdout"],
        "visible_test_stderr": score["visible_test_stderr"],
        "hidden_test_stdout": score["hidden_test_stdout"],
        "hidden_test_stderr": score["hidden_test_stderr"],
    }


def _load_or_run_codex_rows(args: argparse.Namespace, task_ids: set[str]) -> list[dict[str, Any]]:
    raw_output = args.codex_raw_output or (args.docs_dir / "raw_codex_results.jsonl")
    if args.codex_raw_input is not None:
        rows = []
        for line in args.codex_raw_input.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("kind") == "task_eval_record_v1" and payload.get("task_id") in task_ids:
                rows.append(payload)
        return rows

    existing = load_existing_codex_records(raw_output) if args.resume else {}
    rows: list[dict[str, Any]] = list(existing.values()) if args.resume else []
    seen = set(existing)
    tasks = load_codex_task_contexts(args.task_manifest, limit=args.num_tasks)
    run_plan: list[tuple[Any, str]] = []
    for task in tasks:
        for method in args.methods:
            key = (task.task_id, method, args.seed)
            if key in seen:
                continue
            run_plan.append((task, method))
    progress = None
    if not args.no_progress:
        total = len(tasks) * len(args.methods)
        progress = __import__("tqdm.auto").auto.tqdm(total=total, desc="Codex benchmark", unit="run", dynamic_ncols=True)
    try:
        if progress is not None and args.resume:
            progress.update(len(seen))
        if not run_plan:
            return sorted(rows, key=lambda row: (row["task_id"], row["method"]))
        max_workers = max(1, min(args.parallelism, len(run_plan)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(_run_codex_method_for_task, task, method, args): (task.task_id, method)
                for task, method in run_plan
            }
            for future in as_completed(future_map):
                task_id, method = future_map[future]
                if progress is not None:
                    progress.set_postfix(task=task_id, method=method, refresh=False)
                try:
                    record = future.result()
                except Exception as exc:
                    _warn(f"Codex benchmark failed for task={task_id} method={method}: {exc}")
                    record = {
                        "kind": "task_eval_error_v1",
                        "task_id": task_id,
                        "method": method,
                        "seed": args.seed,
                        "error": str(exc),
                    }
                rows.append(record)
                seen.add((task_id, method, args.seed))
                write_raw_jsonl(
                    raw_output,
                    sorted(
                        [row for row in rows if row.get("kind") == "task_eval_record_v1"],
                        key=lambda row: (row["task_id"], row["method"]),
                    ),
                )
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(task=task_id, method=method, solved=record.get("solved"), refresh=False)
    finally:
        if progress is not None:
            progress.close()
    return sorted([row for row in rows if row.get("kind") == "task_eval_record_v1"], key=lambda row: (row["task_id"], row["method"]))


def main() -> int:
    args = parse_args()
    docs_dir = args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    task_contexts = load_codex_task_contexts(args.task_manifest, limit=args.num_tasks)
    task_ids = {task.task_id for task in task_contexts}
    rainbow_rows = load_rainbow_baseline_rows(args.rainbow_raw_input, task_ids=task_ids, seed=args.seed)
    if len(rainbow_rows) != len(task_ids):
        _warn(
            f"Expected {len(task_ids)} Rainbow baseline rows from {args.rainbow_raw_input}, found {len(rainbow_rows)}."
        )

    codex_rows = _load_or_run_codex_rows(args, task_ids)
    combined_rows = rainbow_rows + codex_rows

    normalized_path = docs_dir / "normalized_results.csv"
    summary_path = docs_dir / "summary_metrics.csv"
    paired_path = docs_dir / "paired_deltas.csv"
    raw_path = args.codex_raw_output or (docs_dir / "raw_codex_results.jsonl")

    write_raw_jsonl(raw_path, codex_rows)
    write_csv(normalized_path, RAW_HEADERS, combined_rows)

    summary_rows = compute_summary_metrics(combined_rows)
    write_csv(summary_path, SUMMARY_HEADERS, summary_rows)

    paired_rows = compute_paired_deltas(combined_rows, resamples=args.bootstrap_resamples, seed=args.seed)
    write_csv(paired_path, PAIRED_HEADERS, paired_rows)

    plot_summary_bars(
        combined_rows,
        summary_rows,
        metric_key="solve_rate",
        title="Solve rate by method",
        ylabel="Solve rate",
        output_stem=docs_dir / "solve_rate",
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    plot_summary_bars(
        combined_rows,
        summary_rows,
        metric_key="mean_elapsed_time_ms",
        title="Time per task by method",
        ylabel="Mean elapsed time (ms)",
        output_stem=docs_dir / "time_per_task",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 100,
    )
    plot_summary_bars(
        combined_rows,
        summary_rows,
        metric_key="mean_llm_total_tokens",
        title="Tokens per task by method",
        ylabel="Mean total tokens",
        output_stem=docs_dir / "tokens_per_task",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 200,
    )
    token_breakdown_available = plot_token_breakdown(summary_rows, output_stem=docs_dir / "token_breakdown")
    for baseline in CODEX_METHODS:
        plot_pairwise_delta(baseline, paired_rows, output_stem=docs_dir / f"rainbow_vs_{baseline}_delta")

    report = build_results_report(
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        token_breakdown_available=token_breakdown_available,
        task_count=len(task_ids),
        rainbow_raw_input=args.rainbow_raw_input,
    )
    (docs_dir / "results.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

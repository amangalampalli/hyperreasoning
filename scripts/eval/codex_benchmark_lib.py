"""Codex benchmark helpers for isolated task execution and reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
from itertools import product
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required for Codex benchmark graph generation."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm.prompt_utils import TaskContext, load_task_context
from scripts.eval.run_eval_pipeline import (
    _coerce_float,
    _coerce_int,
    _warn,
    discover_visible_test_total,
    run_counted_visible_tests,
)


METHOD_ORDER = ["rainbow", "codex_5_4_low", "codex_5_4_medium", "codex_5_4_high"]
CODEX_METHODS = METHOD_ORDER[1:]
METHOD_LABELS = {
    "rainbow": "Rainbow",
    "codex_5_4_low": "Codex 5.4 Low",
    "codex_5_4_medium": "Codex 5.4 Medium",
    "codex_5_4_high": "Codex 5.4 High",
}
METHOD_COLORS = {
    "rainbow": "#1E5BFF",
    "codex_5_4_low": "#6B7C93",
    "codex_5_4_medium": "#C28C43",
    "codex_5_4_high": "#A25555",
}
PAIRWISE_METRICS = [
    {"name": "solved", "label": "Solve rate", "better": "higher"},
    {"name": "elapsed_time_ms", "label": "Elapsed time (ms)", "better": "lower"},
    {"name": "llm_total_tokens", "label": "Total tokens", "better": "lower"},
]
RAW_HEADERS = [
    "task_id",
    "method",
    "solved",
    "tests_passed",
    "tests_total",
    "fraction_tests_passed",
    "elapsed_time_ms",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_total_tokens",
    "llm_reasoning_tokens",
    "llm_execution_tokens",
    "branches_explored",
    "steps_to_success",
    "seed",
    "codex_exit_code",
    "codex_last_message",
    "codex_json_events_summary",
    "llm_usage_payload",
]
SUMMARY_HEADERS = [
    "method",
    "task_count",
    "solved_tasks",
    "solve_rate",
    "mean_fraction_tests_passed",
    "median_fraction_tests_passed",
    "mean_elapsed_time_ms",
    "median_elapsed_time_ms",
    "mean_llm_input_tokens",
    "mean_llm_total_tokens",
    "median_llm_total_tokens",
    "mean_llm_reasoning_tokens",
    "mean_llm_execution_tokens",
    "tokens_per_solved_task",
    "time_per_solved_task",
]
PAIRED_HEADERS = [
    "baseline",
    "metric",
    "n_tasks",
    "rainbow_mean",
    "baseline_mean",
    "mean_delta",
    "ci_low",
    "ci_high",
    "p_value",
    "test_name",
    "rainbow_better_tasks",
    "baseline_better_tasks",
    "tied_tasks",
    "summary",
]


@dataclass(frozen=True)
class CodexTier:
    method: str
    reasoning_effort: str


CODEX_TIERS = [
    CodexTier(method="codex_5_4_low", reasoning_effort="low"),
    CodexTier(method="codex_5_4_medium", reasoning_effort="medium"),
    CodexTier(method="codex_5_4_high", reasoning_effort="high"),
]


def _json_ready(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    header: (
                        ""
                        if row.get(header) is None
                        else json.dumps(row.get(header), sort_keys=True)
                        if isinstance(row.get(header), (dict, list))
                        else f"{row[header]:.6f}" if isinstance(row.get(header), float)
                        else row.get(header)
                    )
                    for header in headers
                }
            )


def copy_task_workspace(task: TaskContext) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix=f"codex_benchmark_{task.task_id}_"))
    workspace = temp_root / task.task_id
    shutil.copytree(task.task_dir, workspace)
    if task.hidden_test_file:
        hidden_path = workspace / task.hidden_test_file
        if hidden_path.exists():
            hidden_path.unlink()
        task_json_path = workspace / "task.json"
        if task_json_path.exists():
            payload = json.loads(task_json_path.read_text(encoding="utf-8"))
            payload["hidden_test_file"] = None
            task_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return workspace


def inject_hidden_test_file(task: TaskContext, workspace: Path) -> None:
    if not task.hidden_test_file:
        return
    source = task.task_dir / task.hidden_test_file
    destination = workspace / task.hidden_test_file
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_codex_prompt(task: TaskContext) -> str:
    lines = [
        "You are working inside an isolated benchmark task workspace.",
        "Fix the task by editing files in this workspace.",
        "Keep the change minimal and focused.",
        "Do not add dependencies.",
        "You may run the visible tests in the workspace.",
        "Hidden tests are not available in this workspace and will be run after you finish.",
        "",
        f"task_id: {task.task_id}",
        f"family: {task.family}",
        f"prompt: {task.prompt}",
        f"target_files: {', '.join(task.target_files)}",
        f"visible_test_file: {task.visible_test_file or 'n/a'}",
    ]
    return "\n".join(lines)


def build_codex_command(
    *,
    workspace: Path,
    prompt: str,
    tier: CodexTier,
    output_last_message: Path,
    model: str,
) -> list[str]:
    return [
        "codex",
        "exec",
        "--skip-git-repo-check",
        "--full-auto",
        "--ephemeral",
        "--json",
        "--output-last-message",
        str(output_last_message),
        "-C",
        str(workspace),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{tier.reasoning_effort}"',
        prompt,
    ]


def parse_codex_jsonl_output(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line in stdout.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            warnings.append(text)
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events, warnings


def _nested_get(payload: dict[str, Any], path: str) -> Any:
    current: Any = payload
    for segment in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(segment)
    return current


def extract_usage_fields(usage: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(usage, dict):
        return {
            "llm_input_tokens": None,
            "llm_output_tokens": None,
            "llm_total_tokens": None,
            "llm_reasoning_tokens": None,
            "llm_execution_tokens": None,
            "llm_usage_payload": None,
        }

    input_tokens = _coerce_int(usage.get("input_tokens") or usage.get("prompt_tokens"))
    output_tokens = _coerce_int(usage.get("output_tokens") or usage.get("completion_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    reasoning_tokens = None
    for key in (
        "reasoning_tokens",
        "output_tokens_details.reasoning_tokens",
        "completion_tokens_details.reasoning_tokens",
        "output_token_details.reasoning_tokens",
    ):
        reasoning_tokens = _coerce_int(_nested_get(usage, key))
        if reasoning_tokens is not None:
            break

    execution_tokens = None
    for key in (
        "execution_tokens",
        "output_tokens_details.execution_tokens",
        "completion_tokens_details.execution_tokens",
        "output_token_details.execution_tokens",
    ):
        execution_tokens = _coerce_int(_nested_get(usage, key))
        if execution_tokens is not None:
            break
    if execution_tokens is None and reasoning_tokens is not None and output_tokens is not None and output_tokens >= reasoning_tokens:
        execution_tokens = output_tokens - reasoning_tokens

    return {
        "llm_input_tokens": input_tokens,
        "llm_output_tokens": output_tokens,
        "llm_total_tokens": total_tokens,
        "llm_reasoning_tokens": reasoning_tokens,
        "llm_execution_tokens": execution_tokens,
        "llm_usage_payload": usage,
    }


def summarize_codex_events(events: list[dict[str, Any]], warnings: list[str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    event_counts = Counter(str(event.get("type", "unknown")) for event in events)
    usage_payload = None
    last_agent_message = None
    for event in events:
        if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
            usage_payload = event["usage"]
        if event.get("type") == "item.completed":
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                last_agent_message = item.get("text")
    summary = {
        "event_counts": dict(event_counts),
        "warning_count": len(warnings),
        "warning_sample": warnings[:10],
        "last_agent_message": last_agent_message,
    }
    return summary, usage_payload


def run_external_tests(task: TaskContext, workspace: Path, *, python_bin: str, timeout_s: float) -> dict[str, Any]:
    visible_total = discover_visible_test_total(task, python_bin=python_bin, timeout_s=timeout_s)
    visible_passed = 0
    visible_passed_flag = None
    visible_stdout = None
    visible_stderr = None
    if task.visible_test_file:
        visible_passed_flag, _, visible_stdout, visible_stderr, visible_counts = run_counted_visible_tests(
            workspace_dir=workspace,
            test_file=task.visible_test_file,
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        visible_passed = visible_counts.passed
        visible_total = visible_counts.total or visible_total

    hidden_total = 0
    hidden_passed = 0
    hidden_passed_flag = None
    hidden_stdout = None
    hidden_stderr = None
    if task.hidden_test_file:
        hidden_total = discover_visible_test_total(
            TaskContext.model_validate({**task.model_dump(), "visible_test_file": task.hidden_test_file}),
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        inject_hidden_test_file(task, workspace)
        hidden_passed_flag, _, hidden_stdout, hidden_stderr, hidden_counts = run_counted_visible_tests(
            workspace_dir=workspace,
            test_file=task.hidden_test_file,
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        hidden_passed = hidden_counts.passed
        hidden_total = hidden_counts.total or hidden_total

    tests_passed = visible_passed + hidden_passed
    tests_total = visible_total + hidden_total
    solved = int(tests_total > 0 and tests_passed == tests_total)
    fraction = (tests_passed / tests_total) if tests_total else 0.0
    return {
        "solved": solved,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "fraction_tests_passed": fraction,
        "visible_test_passed": visible_passed_flag,
        "hidden_test_passed": hidden_passed_flag,
        "visible_test_stdout": visible_stdout,
        "visible_test_stderr": visible_stderr,
        "hidden_test_stdout": hidden_stdout,
        "hidden_test_stderr": hidden_stderr,
    }


def load_codex_task_contexts(manifest_path: Path, limit: int | None = None) -> list[TaskContext]:
    task_dirs = [Path(line.strip()) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        task_dirs = task_dirs[:limit]
    resolved: list[TaskContext] = []
    for task_dir in task_dirs:
        resolved.append(load_task_context((ROOT / task_dir).resolve() if not task_dir.is_absolute() else task_dir))
    return resolved


def load_existing_codex_records(path: Path) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") != "task_eval_record_v1":
            continue
        records[(payload["task_id"], payload["method"], payload.get("seed"))] = payload
    return records


def load_rainbow_baseline_rows(path: Path, *, task_ids: set[str], seed: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("kind") != "task_eval_record_v1":
            continue
        if payload.get("method") != "rainbow":
            continue
        if payload.get("task_id") not in task_ids:
            continue
        if seed is not None and payload.get("seed") != seed:
            continue
        tests_passed = _coerce_int(payload.get("tests_passed")) or 0
        tests_total = _coerce_int(payload.get("tests_total")) or 0
        fraction = (tests_passed / tests_total) if tests_total else 0.0
        rows.append(
            {
                "task_id": payload["task_id"],
                "method": "rainbow",
                "solved": (
                    _coerce_int(payload.get("solved"))
                    if payload.get("solved") is not None
                    else int(tests_total > 0 and tests_passed == tests_total)
                ),
                "tests_passed": tests_passed,
                "tests_total": tests_total,
                "fraction_tests_passed": _coerce_float(payload.get("fraction_tests_passed")) if payload.get("fraction_tests_passed") is not None else fraction,
                "elapsed_time_ms": _coerce_float(payload.get("elapsed_time_ms")),
                "llm_input_tokens": _coerce_int(payload.get("llm_input_tokens")),
                "llm_output_tokens": _coerce_int(payload.get("llm_output_tokens")),
                "llm_total_tokens": _coerce_int(payload.get("llm_total_tokens")),
                "llm_reasoning_tokens": None,
                "llm_execution_tokens": None,
                "branches_explored": _coerce_int(payload.get("branches_explored")),
                "steps_to_success": _coerce_int(payload.get("steps_to_success")),
                "seed": _coerce_int(payload.get("seed")),
                "codex_exit_code": None,
                "codex_last_message": None,
                "codex_json_events_summary": None,
                "llm_usage_payload": None,
            }
        )
    return rows


def write_raw_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_ready) + "\n")


def _format_metric_value(value: float | None, metric_name: str) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if metric_name in {"solve_rate", "mean_fraction_tests_passed", "median_fraction_tests_passed"}:
        return f"{value:.1%}"
    if metric_name.endswith("_ms") or metric_name == "time_per_solved_task":
        return f"{value:,.0f} ms"
    return f"{value:,.0f}"


def _bar_value_formatter(metric_key: str) -> FuncFormatter | PercentFormatter:
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        return PercentFormatter(xmax=1.0, decimals=0)
    return FuncFormatter(lambda value, _: f"{value:,.0f}")


def _format_bar_label(metric_key: str, value: float) -> str:
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        return f"{value:.1%}"
    if abs(value - round(value)) >= 0.05:
        return f"{value:,.1f}"
    return f"{value:,.0f}"


def _metric_values_for_method(rows: list[dict[str, Any]], method: str, key: str) -> np.ndarray:
    values = [
        float(value)
        for row in rows
        if row["method"] == method
        for value in [row.get(key)]
        if _coerce_float(value) is not None
    ]
    return np.asarray(values, dtype=float)


def bootstrap_mean_ci(values: np.ndarray, *, resamples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    sample_means = values[indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return float(low), float(high)


def exact_mcnemar_p_value(rainbow: np.ndarray, baseline: np.ndarray) -> float:
    better = int(np.sum((rainbow == 1) & (baseline == 0)))
    worse = int(np.sum((rainbow == 0) & (baseline == 1)))
    discordant = better + worse
    if discordant == 0:
        return 1.0
    smaller = min(better, worse)
    cumulative = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * cumulative))


def exact_sign_flip_p_value(deltas: np.ndarray) -> float:
    if deltas.size == 0 or np.allclose(deltas, 0.0):
        return 1.0
    observed = abs(float(np.mean(deltas)))
    if deltas.size <= 16:
        total = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=deltas.size):
            signed = deltas * np.asarray(signs, dtype=float)
            statistic = abs(float(np.mean(signed)))
            total += 1
            if statistic >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = np.random.default_rng(12345)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, deltas.size))
    statistics = np.abs((signs * deltas).mean(axis=1))
    return float(np.mean(statistics >= observed - 1e-12))


def compute_summary_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for row in rows:
        by_method.setdefault(row["method"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        items = by_method.get(method, [])
        solved_tasks = sum(int(row["solved"]) for row in items)
        elapsed_values = [_coerce_float(row.get("elapsed_time_ms")) for row in items]
        elapsed_values = [v for v in elapsed_values if v is not None]
        input_values = [_coerce_float(row.get("llm_input_tokens")) for row in items]
        input_values = [v for v in input_values if v is not None]
        fraction_values = [_coerce_float(row.get("fraction_tests_passed")) for row in items]
        fraction_values = [v for v in fraction_values if v is not None]
        token_values = [_coerce_float(row.get("llm_total_tokens")) for row in items]
        token_values = [v for v in token_values if v is not None]
        reasoning_values = [_coerce_float(row.get("llm_reasoning_tokens")) for row in items]
        reasoning_values = [v for v in reasoning_values if v is not None]
        execution_values = [_coerce_float(row.get("llm_execution_tokens")) for row in items]
        execution_values = [v for v in execution_values if v is not None]
        total_elapsed = sum(elapsed_values)
        total_tokens = sum(token_values)
        task_count = len(items)
        summaries.append(
            {
                "method": method,
                "task_count": task_count,
                "solved_tasks": solved_tasks,
                "solve_rate": (solved_tasks / task_count) if task_count else None,
                "mean_fraction_tests_passed": (sum(fraction_values) / len(fraction_values)) if fraction_values else None,
                "median_fraction_tests_passed": float(np.median(np.asarray(fraction_values))) if fraction_values else None,
                "mean_elapsed_time_ms": (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else None,
                "median_elapsed_time_ms": float(np.median(np.asarray(elapsed_values))) if elapsed_values else None,
                "mean_llm_input_tokens": (sum(input_values) / len(input_values)) if input_values else None,
                "mean_llm_total_tokens": (sum(token_values) / len(token_values)) if token_values else None,
                "median_llm_total_tokens": float(np.median(np.asarray(token_values))) if token_values else None,
                "mean_llm_reasoning_tokens": (sum(reasoning_values) / len(reasoning_values)) if reasoning_values else None,
                "mean_llm_execution_tokens": (sum(execution_values) / len(execution_values)) if execution_values else None,
                "tokens_per_solved_task": (total_tokens / solved_tasks) if solved_tasks else None,
                "time_per_solved_task": (total_elapsed / solved_tasks) if solved_tasks else None,
            }
        )
    return summaries


def compute_paired_deltas(rows: list[dict[str, Any]], *, resamples: int, seed: int) -> list[dict[str, Any]]:
    rows_by_task_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rows_by_task_method.setdefault(row["task_id"], {})[row["method"]] = row
    paired_rows: list[dict[str, Any]] = []
    for baseline in CODEX_METHODS:
        for metric_index, metric in enumerate(PAIRWISE_METRICS):
            rainbow_values: list[float] = []
            baseline_values: list[float] = []
            for task_id in sorted(rows_by_task_method):
                task_rows = rows_by_task_method[task_id]
                rainbow_row = task_rows.get("rainbow")
                baseline_row = task_rows.get(baseline)
                if rainbow_row is None or baseline_row is None:
                    continue
                rainbow_value = _coerce_float(rainbow_row.get(metric["name"]))
                baseline_value = _coerce_float(baseline_row.get(metric["name"]))
                if rainbow_value is None or baseline_value is None:
                    continue
                rainbow_values.append(rainbow_value)
                baseline_values.append(baseline_value)
            if not rainbow_values:
                continue
            rainbow_array = np.asarray(rainbow_values, dtype=float)
            baseline_array = np.asarray(baseline_values, dtype=float)
            deltas = rainbow_array - baseline_array
            ci_low, ci_high = bootstrap_mean_ci(
                deltas,
                resamples=resamples,
                seed=seed + (metric_index + 1) * 101 + METHOD_ORDER.index(baseline) * 1009,
            )
            if metric["name"] == "solved":
                p_value = exact_mcnemar_p_value(rainbow_array.astype(int), baseline_array.astype(int))
                test_name = "Exact McNemar"
            else:
                p_value = exact_sign_flip_p_value(deltas)
                test_name = "Exact paired sign-flip"
            rainbow_better_tasks = 0
            baseline_better_tasks = 0
            tied_tasks = 0
            for delta in deltas:
                if abs(float(delta)) <= 1e-12:
                    tied_tasks += 1
                elif (metric["name"] in {"solved"} and delta > 0) or (metric["name"] not in {"solved"} and delta < 0 and metric["better"] == "lower") or (metric["name"] not in {"solved"} and delta > 0 and metric["better"] == "higher"):
                    rainbow_better_tasks += 1
                else:
                    baseline_better_tasks += 1
            paired_rows.append(
                {
                    "baseline": baseline,
                    "metric": metric["name"],
                    "n_tasks": int(deltas.size),
                    "rainbow_mean": float(np.mean(rainbow_array)),
                    "baseline_mean": float(np.mean(baseline_array)),
                    "mean_delta": float(np.mean(deltas)),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_value,
                    "test_name": test_name,
                    "rainbow_better_tasks": rainbow_better_tasks,
                    "baseline_better_tasks": baseline_better_tasks,
                    "tied_tasks": tied_tasks,
                    "summary": "",
                }
            )
    return paired_rows


def export_figure(fig: Any, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=240)
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def plot_summary_bars(
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    output_stem: Path,
    resamples: int,
    seed: int,
) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    positions = np.arange(len(METHOD_ORDER))
    values: list[float] = []
    error_low: list[float] = []
    error_high: list[float] = []
    for method_index, method in enumerate(METHOD_ORDER):
        summary_row = next((row for row in summary_rows if row["method"] == method), None)
        value = _coerce_float(summary_row.get(metric_key) if summary_row else None) or 0.0
        values.append(value)
        method_values = _metric_values_for_method(
            rows,
            method,
            "solved" if metric_key == "solve_rate" else metric_key.replace("mean_", ""),
        )
        if method_values.size:
            ci_low, ci_high = bootstrap_mean_ci(
                method_values,
                resamples=resamples,
                seed=seed + method_index * 97 + len(metric_key),
            )
            error_low.append(max(0.0, value - ci_low))
            error_high.append(max(0.0, ci_high - value))
        else:
            error_low.append(0.0)
            error_high.append(0.0)
    bars = ax.bar(positions, values, color=[METHOD_COLORS[m] for m in METHOD_ORDER], width=0.66, zorder=3)
    ax.errorbar(
        positions,
        values,
        yerr=np.asarray([error_low, error_high]),
        fmt="none",
        ecolor="#263238",
        capsize=4,
        linewidth=1.2,
        zorder=4,
    )
    ax.set_xticks(positions, [METHOD_LABELS[m] for m in METHOD_ORDER])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(_bar_value_formatter(metric_key))
    ax.grid(axis="y", color="#E7ECF2", linewidth=1.0, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    upper = max(values[i] + error_high[i] for i in range(len(values))) if values else 1.0
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        ax.set_ylim(0.0, min(1.15, max(upper, 1.0) + 0.10))
    else:
        ax.set_ylim(0.0, upper * 1.18 if upper > 0 else 1.0)
    for index, (bar, value) in enumerate(zip(bars, values)):
        label = _format_bar_label(metric_key, value)
        cap_top = value + error_high[index] if error_high else value
        offset = ax.get_ylim()[1] * (0.015 if metric_key in {"solve_rate", "mean_fraction_tests_passed"} else 0.03)
        ax.text(bar.get_x() + bar.get_width() / 2, cap_top + offset, label, ha="center", va="bottom", fontsize=10, color="#1F2328")
    fig.tight_layout()
    export_figure(fig, output_stem)


def plot_pairwise_delta(baseline: str, paired_rows: list[dict[str, Any]], *, output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )
    baseline_rows = {row["metric"]: row for row in paired_rows if row["baseline"] == baseline}
    metric_order = [metric["name"] for metric in PAIRWISE_METRICS]
    fig, axes = plt.subplots(1, len(metric_order), figsize=(12.0, 4.2))
    for axis, metric_name in zip(axes.flat, metric_order):
        metric_meta = next(metric for metric in PAIRWISE_METRICS if metric["name"] == metric_name)
        row = baseline_rows.get(metric_name)
        axis.axvline(0.0, color="#4C566A", linewidth=1.0, zorder=1)
        axis.grid(axis="x", color="#E7ECF2", linewidth=1.0, zorder=0)
        axis.set_yticks([])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if row is None:
            axis.set_title(metric_meta["label"])
            axis.text(0.5, 0.5, "No paired data", ha="center", va="center", transform=axis.transAxes)
            continue
        mean_delta = float(row["mean_delta"])
        ci_low = float(row["ci_low"])
        ci_high = float(row["ci_high"])
        favorable = (metric_meta["better"] == "higher" and mean_delta >= 0) or (metric_meta["better"] == "lower" and mean_delta <= 0)
        color = METHOD_COLORS["rainbow"] if favorable else "#C05640"
        axis.barh([0], [mean_delta], color=color, alpha=0.18, height=0.34, zorder=2)
        axis.errorbar(
            [mean_delta],
            [0],
            xerr=np.asarray([[max(0.0, mean_delta - ci_low)], [max(0.0, ci_high - mean_delta)]]),
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.0,
            capsize=5,
            markersize=7,
            zorder=3,
        )
        span = max(abs(ci_low), abs(ci_high), abs(mean_delta))
        if span <= 1e-12:
            span = {"solved": 0.1, "elapsed_time_ms": 1000.0, "llm_total_tokens": 100.0}[metric_name]
        axis.set_xlim(-1.15 * span, 1.15 * span)
        axis.set_title(metric_meta["label"], fontsize=12)
        axis.text(0.02, 0.92, f"p={row['p_value']:.3f}", transform=axis.transAxes, ha="left", va="top", fontsize=10)
        if metric_name == "solved":
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        else:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.suptitle(f"Rainbow vs {METHOD_LABELS[baseline]}", fontsize=16, y=1.02)
    fig.tight_layout()
    export_figure(fig, output_stem)


def render_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    header = [
        "Method",
        "Solve rate",
        "Mean tests passed",
        "Mean time",
        "Mean total tokens",
        "Mean reasoning tokens",
        "Mean execution tokens",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    METHOD_LABELS[row["method"]],
                    _format_metric_value(_coerce_float(row["solve_rate"]), "solve_rate"),
                    _format_metric_value(_coerce_float(row["mean_fraction_tests_passed"]), "mean_fraction_tests_passed"),
                    _format_metric_value(_coerce_float(row["mean_elapsed_time_ms"]), "mean_elapsed_time_ms"),
                    _format_metric_value(_coerce_float(row["mean_llm_total_tokens"]), "mean_llm_total_tokens"),
                    _format_metric_value(_coerce_float(row["mean_llm_reasoning_tokens"]), "mean_llm_reasoning_tokens"),
                    _format_metric_value(_coerce_float(row["mean_llm_execution_tokens"]), "mean_llm_execution_tokens"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_results_report(
    *,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    token_breakdown_available: bool,
    task_count: int,
    rainbow_raw_input: Path,
) -> str:
    paired_lines = []
    for row in paired_rows:
        if row["metric"] == "solved":
            paired_lines.append(
                f"- Rainbow vs {METHOD_LABELS[row['baseline']]} solve-rate delta: {row['mean_delta']:+.2f} "
                f"(95% CI [{row['ci_low']:.2f}, {row['ci_high']:.2f}], p={row['p_value']:.3f})"
            )
        elif row["metric"] == "elapsed_time_ms":
            paired_lines.append(
                f"- Rainbow vs {METHOD_LABELS[row['baseline']]} time delta: {row['mean_delta']:+,.0f} ms "
                f"(95% CI [{row['ci_low']:.0f}, {row['ci_high']:.0f}], p={row['p_value']:.3f})"
            )
        elif row["metric"] == "llm_total_tokens":
            paired_lines.append(
                f"- Rainbow vs {METHOD_LABELS[row['baseline']]} token delta: {row['mean_delta']:+,.0f} "
                f"(95% CI [{row['ci_low']:.0f}, {row['ci_high']:.0f}], p={row['p_value']:.3f})"
            )
    report = [
        "# Codex Benchmark Results",
        "",
        f"Rainbow source: `{rainbow_raw_input}`",
        f"Paired units: `{task_count}` eval tasks",
        "",
        "## Summary",
        "",
        render_summary_table(summary_rows),
        "",
        "## Paired Comparisons",
        "",
        *paired_lines,
        "",
        "## Graphs",
        "",
        "- [Solve rate](solve_rate.png)",
        "- [Time per task](time_per_task.png)",
        "- [Tokens per task](tokens_per_task.png)",
    ]
    if token_breakdown_available:
        report.append("- [Token breakdown](token_breakdown.png)")
    report.extend(
        [
            f"- [Rainbow vs {METHOD_LABELS['codex_5_4_low']}](rainbow_vs_codex_5_4_low_delta.png)",
            f"- [Rainbow vs {METHOD_LABELS['codex_5_4_medium']}](rainbow_vs_codex_5_4_medium_delta.png)",
            f"- [Rainbow vs {METHOD_LABELS['codex_5_4_high']}](rainbow_vs_codex_5_4_high_delta.png)",
        ]
    )
    report.extend(
        [
            "",
            "## Assumptions",
            "",
            "- Rainbow rows are loaded from the existing raw eval file and re-scored using all-tests-gated success (`tests_passed == tests_total`).",
            "- Codex timing is measured around the full `codex exec` invocation.",
            "- Reasoning/execution token splits are included only when Codex CLI exposes them cleanly; otherwise those fields stay blank and total tokens remain the primary metric.",
        ]
    )
    return "\n".join(report) + "\n"


def plot_token_breakdown(summary_rows: list[dict[str, Any]], *, output_stem: Path) -> bool:
    codex_rows = [row for row in summary_rows if row["method"] in CODEX_METHODS and _coerce_float(row.get("mean_llm_reasoning_tokens")) is not None]
    if not codex_rows:
        return False
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D0D7DE",
            "axes.grid": False,
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    positions = np.arange(len(summary_rows))
    input_values = [(_coerce_float(row.get("mean_llm_input_tokens")) or 0.0) for row in summary_rows]
    reasoning_values = [(_coerce_float(row.get("mean_llm_reasoning_tokens")) or 0.0) for row in summary_rows]
    execution_values = [(_coerce_float(row.get("mean_llm_execution_tokens")) or 0.0) for row in summary_rows]
    ax.bar(positions, input_values, color="#B0B8C4", label="Input")
    ax.bar(positions, reasoning_values, bottom=input_values, color="#4F7BFF", label="Reasoning")
    ax.bar(positions, execution_values, bottom=np.asarray(input_values) + np.asarray(reasoning_values), color="#4C9F70", label="Execution")
    ax.set_xticks(positions, [METHOD_LABELS[row["method"]] for row in summary_rows])
    ax.set_title("Token composition by method")
    ax.set_ylabel("Mean tokens")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E7ECF2", linewidth=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    export_figure(fig, output_stem)
    return True

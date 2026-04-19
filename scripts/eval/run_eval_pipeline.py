#!/usr/bin/env python3
"""Standalone eval-set analysis, plotting, and reporting pipeline."""

from __future__ import annotations

import argparse
import ast
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
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter
except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing
    raise SystemExit(
        "matplotlib is required for graph generation. Install it in the "
        "'hyperreasoning' conda environment before running this pipeline."
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.verifier import CachedTaskVerifier
from llm.prompt_utils import TaskContext
from llm.proposal import plan_signature
from llm.compiler import apply_compiled_files


METHOD_ORDER = ["rainbow", "heuristic", "random", "one_shot"]
METHOD_TO_POLICY = {
    "rainbow": "rainbow",
    "heuristic": "heuristic",
    "random": "random",
    "one_shot": "oneshot",
}
METHOD_LABELS = {
    "rainbow": "Rainbow",
    "heuristic": "Heuristic",
    "random": "Random",
    "one_shot": "One-shot",
}
METHOD_COLORS = {
    "rainbow": "#1E5BFF",
    "heuristic": "#7B8794",
    "random": "#B48B63",
    "one_shot": "#6D9F9B",
}
PAIRWISE_METRICS = [
    {
        "name": "solved",
        "label": "Solve rate",
        "better": "higher",
        "summary_unit": "rate",
        "plot_stem": "solve_rate",
    },
    {
        "name": "fraction_tests_passed",
        "label": "Fraction tests passed",
        "better": "higher",
        "summary_unit": "fraction",
        "plot_stem": "fraction_tests_passed",
    },
    {
        "name": "elapsed_time_ms",
        "label": "Elapsed time (ms)",
        "better": "lower",
        "summary_unit": "ms",
        "plot_stem": "elapsed_time_ms",
    },
    {
        "name": "llm_total_tokens",
        "label": "Total tokens",
        "better": "lower",
        "summary_unit": "tokens",
        "plot_stem": "llm_total_tokens",
    },
]
COUNT_SENTINEL = "__HR_EVAL_COUNTS__"
NORMALIZED_HEADERS = [
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
    "branches_explored",
    "steps_to_success",
    "seed",
]
RERUN_FIELD_CHOICES = [
    "elapsed_time_ms",
    "llm_input_tokens",
    "llm_output_tokens",
    "llm_total_tokens",
    "branches_explored",
    "steps_to_success",
    "tests_passed",
    "tests_total",
    "fraction_tests_passed",
    "solved",
    "visible_test_passed",
    "hidden_test_passed",
    "visible_tests_passed",
    "visible_tests_total",
    "hidden_tests_passed",
    "hidden_tests_total",
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
    "mean_llm_total_tokens",
    "median_llm_total_tokens",
    "mean_branches_explored",
    "mean_steps_to_success",
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


class PipelineWarning(RuntimeError):
    """Internal warning carrier for malformed raw records."""


class VisibleTestCounts:
    def __init__(self, *, passed: int = 0, total: int = 0) -> None:
        self.passed = passed
        self.total = total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-manifest", type=Path, default=ROOT / "data/splits/eval_10.txt")
    parser.add_argument("--docs-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/models/rainbow_offline_v1_1776237078/best.pt")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument("--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-verified-plans-per-task", type=int, default=1)
    parser.add_argument("--proposal-source", choices=["heuristic", "llm", "hybrid"], default="heuristic")
    parser.add_argument("--allow-full-file-fallback", action="store_true")
    parser.add_argument("--compiler-temp", type=float, default=0.2)
    parser.add_argument("--timeout-s", type=float, default=12.0)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--raw-output", type=Path, default=None)
    parser.add_argument("--raw-input", nargs="*", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--replace-methods",
        nargs="+",
        choices=METHOD_ORDER,
        default=None,
        help="Delete existing raw records for these methods before rerunning them.",
    )
    parser.add_argument(
        "--preserve-fields",
        nargs="+",
        choices=RERUN_FIELD_CHOICES,
        default=None,
        help="When rerunning methods, carry over these fields from the existing raw results for matching task/method/seed rows.",
    )
    return parser.parse_args()


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
        "one_shot_llm": "one_shot",
        "1_shot": "one_shot",
    }
    return aliases.get(normalized)


def display_method_name(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " ").title())


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def format_metric_value(value: float | None, metric_name: str) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if metric_name in {"solve_rate", "mean_fraction_tests_passed", "median_fraction_tests_passed"}:
        return f"{value:.1%}"
    if metric_name in {"mean_llm_total_tokens", "median_llm_total_tokens", "tokens_per_solved_task"}:
        return f"{value:,.0f}"
    if metric_name.endswith("_ms") or metric_name == "time_per_solved_task":
        return f"{value:,.0f} ms"
    return f"{value:.3f}"


def format_delta(value: float, metric_name: str) -> str:
    if metric_name in {"solved", "fraction_tests_passed"}:
        return f"{value:+.2f}"
    if metric_name == "elapsed_time_ms":
        return f"{value:+,.0f} ms"
    if metric_name == "llm_total_tokens":
        return f"{value:+,.0f}"
    return f"{value:+.3f}"


def format_p_value(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


def _warn(message: str) -> None:
    print(f"[eval-pipeline] {message}", file=sys.stderr)


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fallback_visible_test_total(task: TaskContext) -> int:
    if task.visible_test_file is None:
        return 0
    source = task.files.get(task.visible_test_file)
    if not source:
        return 0
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    total = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            total += 1
    return total


def _count_script() -> str:
    return "\n".join(
        [
            "import json",
            "import sys",
            "import unittest",
            "",
            "module_target = sys.argv[1]",
            "execute_tests = sys.argv[2] == '1'",
            "suite = unittest.defaultTestLoader.loadTestsFromName(module_target)",
            "if not execute_tests:",
            "    print('__HR_EVAL_COUNTS__' + json.dumps({'tests_run': suite.countTestCases()}, sort_keys=True))",
            "    raise SystemExit(0)",
            "runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)",
            "result = runner.run(suite)",
            "payload = {",
            "    'tests_run': result.testsRun,",
            "    'failures': len(result.failures),",
            "    'errors': len(result.errors),",
            "    'skipped': len(getattr(result, 'skipped', [])),",
            "    'expected_failures': len(getattr(result, 'expectedFailures', [])),",
            "    'unexpected_successes': len(getattr(result, 'unexpectedSuccesses', [])),",
            "}",
            "print('__HR_EVAL_COUNTS__' + json.dumps(payload, sort_keys=True))",
            "raise SystemExit(0 if result.wasSuccessful() else 1)",
        ]
    )


def _module_target(test_file: str) -> str:
    if test_file.endswith(".py"):
        return test_file[:-3].replace("/", ".").replace("\\", ".")
    return test_file


def _strip_count_sentinel(text: str) -> tuple[str, dict[str, Any] | None]:
    payload = None
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith(COUNT_SENTINEL):
            try:
                payload = json.loads(line[len(COUNT_SENTINEL) :])
            except json.JSONDecodeError:
                payload = None
        else:
            cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    if cleaned and text.endswith("\n"):
        cleaned += "\n"
    return cleaned, payload


def _run_counting_command(
    *,
    cwd: Path,
    test_file: str,
    python_bin: str,
    timeout_s: float,
    execute_tests: bool,
) -> tuple[int, str, str, dict[str, Any] | None]:
    command = [python_bin, "-c", _count_script(), _module_target(test_file), "1" if execute_tests else "0"]
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = (exc.stderr or "") + f"\nTimed out after {timeout_s} seconds"
        return 124, stdout, stderr, None
    stdout, payload = _strip_count_sentinel(completed.stdout)
    stderr, stderr_payload = _strip_count_sentinel(completed.stderr)
    return completed.returncode, stdout, stderr, payload or stderr_payload


def discover_visible_test_total(task: TaskContext, *, python_bin: str, timeout_s: float) -> int:
    if task.visible_test_file is None:
        return 0
    returncode, _, stderr, payload = _run_counting_command(
        cwd=task.task_dir,
        test_file=task.visible_test_file,
        python_bin=python_bin,
        timeout_s=timeout_s,
        execute_tests=False,
    )
    if payload is not None:
        total = _coerce_int(payload.get("tests_run"))
        if total is not None and total > 0:
            return total
    fallback = _fallback_visible_test_total(task)
    if returncode not in {0, None}:
        _warn(
            f"Falling back to AST test counting for {task.task_id}; "
            f"loader stderr was: {(stderr or '').strip()[:200]}"
        )
    return fallback


def run_counted_visible_tests(
    *,
    workspace_dir: Path,
    test_file: str,
    python_bin: str,
    timeout_s: float,
) -> tuple[bool, int, str, str, VisibleTestCounts]:
    returncode, stdout, stderr, payload = _run_counting_command(
        cwd=workspace_dir,
        test_file=test_file,
        python_bin=python_bin,
        timeout_s=timeout_s,
        execute_tests=True,
    )
    if payload is None:
        return False, returncode, stdout, stderr, VisibleTestCounts()
    tests_run = _coerce_int(payload.get("tests_run")) or 0
    failures = _coerce_int(payload.get("failures")) or 0
    errors = _coerce_int(payload.get("errors")) or 0
    skipped = _coerce_int(payload.get("skipped")) or 0
    expected_failures = _coerce_int(payload.get("expected_failures")) or 0
    unexpected_successes = _coerce_int(payload.get("unexpected_successes")) or 0
    passed = max(0, tests_run - failures - errors - skipped - expected_failures - unexpected_successes)
    return (
        returncode == 0,
        returncode,
        stdout,
        stderr,
        VisibleTestCounts(passed=passed, total=tests_run),
    )


class EvalVisibleTestVerifier(CachedTaskVerifier):
    """Cached verifier that records exact visible and hidden test counts."""

    def __init__(
        self,
        *,
        task: TaskContext,
        client: Any,
        run_tests: bool,
        python_bin: str,
        timeout_s: float,
        compiler_temp: float,
        allow_full_file_fallback: bool,
        max_verified_plans: int,
        compile_fn: Callable[..., dict[str, str] | dict[str, object]] | None = None,
    ) -> None:
        self.task = task
        self.visible_test_total = discover_visible_test_total(task, python_bin=python_bin, timeout_s=timeout_s)
        self.hidden_test_total = (
            discover_visible_test_total(
                TaskContext.model_validate(
                    {
                        **task.model_dump(),
                        "visible_test_file": task.hidden_test_file,
                    }
                ),
                python_bin=python_bin,
                timeout_s=timeout_s,
            )
            if task.hidden_test_file
            else 0
        )
        self.visible_counts_by_signature: dict[str, VisibleTestCounts] = {}
        self.hidden_counts_by_signature: dict[str, VisibleTestCounts] = {}
        self._active_signature: str | None = None
        self.latest_result: Any | None = None
        super().__init__(
            client=client,
            run_tests=run_tests,
            run_hidden_tests=False,
            python_bin=python_bin,
            timeout_s=timeout_s,
            compiler_temp=compiler_temp,
            allow_full_file_fallback=allow_full_file_fallback,
            max_verified_plans=max_verified_plans,
            compile_fn=compile_fn,
            test_runner=self._counted_test_runner,
        )

    def _counted_test_runner(self, workspace_dir: Path, python_bin: str, timeout_s: float) -> tuple[bool, int, str, str]:
        if self.task.visible_test_file is None:
            return False, 1, "", "Missing visible test file"
        passed, returncode, stdout, stderr, counts = run_counted_visible_tests(
            workspace_dir=workspace_dir,
            test_file=self.task.visible_test_file,
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        if counts.total == 0 and self.visible_test_total:
            counts = VisibleTestCounts(
                passed=self.visible_test_total if passed else 0,
                total=self.visible_test_total,
            )
        if self._active_signature is not None:
            self.visible_counts_by_signature[self._active_signature] = counts
        return passed, returncode, stdout, stderr

    def verify(self, task: TaskContext, plan: Any) -> Any:
        signature = plan_signature(plan)
        self._active_signature = signature
        try:
            result = super().verify(task, plan)
        finally:
            self._active_signature = None
        if signature not in self.visible_counts_by_signature:
            self.visible_counts_by_signature[signature] = VisibleTestCounts(
                passed=self.visible_test_total if result.visible_test_passed is True else 0,
                total=self.visible_test_total,
            )
        if task.hidden_test_file and result.compile_success and result.compiled_files:
            hidden_passed, hidden_counts = run_hidden_counted_tests(
                task=task,
                compiled_files=result.compiled_files,
                python_bin=self.python_bin,
                timeout_s=self.timeout_s,
            )
            result.hidden_test_passed = hidden_passed
            self.hidden_counts_by_signature[signature] = hidden_counts
        elif task.hidden_test_file and signature not in self.hidden_counts_by_signature:
            self.hidden_counts_by_signature[signature] = VisibleTestCounts(
                passed=0,
                total=self.hidden_test_total,
            )
        self._cache[signature] = result.model_copy()
        self.latest_result = result.model_copy()
        return result

    def counts_for_signature(self, signature: str | None, *, solved: bool) -> VisibleTestCounts:
        if signature is not None and signature in self.visible_counts_by_signature:
            return self.visible_counts_by_signature[signature]
        return VisibleTestCounts(
            passed=self.visible_test_total if solved and self.visible_test_total else 0,
            total=self.visible_test_total,
        )

    def hidden_counts_for_signature(self, signature: str | None, *, solved: bool) -> VisibleTestCounts:
        if signature is not None and signature in self.hidden_counts_by_signature:
            return self.hidden_counts_by_signature[signature]
        return VisibleTestCounts(
            passed=self.hidden_test_total if solved and self.hidden_test_total else 0,
            total=self.hidden_test_total,
        )


def run_hidden_counted_tests(
    *,
    task: TaskContext,
    compiled_files: dict[str, str],
    python_bin: str,
    timeout_s: float,
) -> tuple[bool | None, VisibleTestCounts]:
    if not task.hidden_test_file:
        return None, VisibleTestCounts()
    temp_dir = Path(tempfile.mkdtemp(prefix=f"eval_hidden_{task.task_id}_"))
    try:
        workspace_dir = apply_compiled_files(task, compiled_files, workspace_dir=temp_dir)
        passed, _, _, _, counts = run_counted_visible_tests(
            workspace_dir=workspace_dir,
            test_file=task.hidden_test_file,
            python_bin=python_bin,
            timeout_s=timeout_s,
        )
        return passed, counts
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_search_config(method: str, args: argparse.Namespace) -> Any:
    from env.dsl_env import SearchControlConfig

    config_kwargs: dict[str, Any] = {
        "max_steps_per_episode": args.max_steps,
        "proposal_source": args.proposal_source,
        "run_tests": True,
        "python_bin": args.python_bin,
        "timeout_s": args.timeout_s,
        "compiler_temp": args.compiler_temp,
        "allow_full_file_fallback": args.allow_full_file_fallback,
        "max_verified_plans_per_task": args.max_verified_plans_per_task,
        "seed": args.seed,
    }
    if method == "one_shot":
        config_kwargs.update(
            {
                "max_steps_per_episode": 2,
                "max_bank_depth": 0,
                "root_candidate_batches": 1,
                "root_candidates_per_batch": 1,
                "max_root_plans": 1,
                "initial_root_reveal": 1,
                "request_batch_size": 1,
                "proposal_source": "llm",
                "llm_proposal_temperature": 0.0,
                "compiler_temp": 0.0,
                "max_verified_plans_per_task": 1,
            }
        )
    return SearchControlConfig(**config_kwargs)


def load_rainbow_artifacts(checkpoint: Path) -> tuple[Any, Any]:
    from env.state_encoder import StateEncoder
    from rl.rainbow import RainbowAgent

    agent, encoder_state = RainbowAgent.load(checkpoint)
    if encoder_state is None:
        raise RuntimeError(f"Checkpoint {checkpoint} does not contain encoder state")
    return agent, StateEncoder.from_dict(encoder_state)


def count_branches_explored(raw_result: Any) -> int:
    visited: set[str] = set()
    for node in raw_result.episode.nodes:
        bank_id = node.get("bank_id")
        if bank_id:
            visited.add(str(bank_id))
    for transition in raw_result.episode.transitions:
        selected_bank_id = transition.info.get("selected_bank_id")
        if selected_bank_id:
            visited.add(str(selected_bank_id))
        current_bank_id = transition.state.get("current_bank_id")
        if current_bank_id:
            visited.add(str(current_bank_id))
        if transition.next_state is not None:
            next_bank_id = transition.next_state.get("current_bank_id")
            if next_bank_id:
                visited.add(str(next_bank_id))
    return len(visited)


def steps_to_success(raw_result: Any) -> int | None:
    for index, transition in enumerate(raw_result.episode.transitions, start=1):
        if transition.info.get("visible_test_passed") is True:
            return index
    return None


def evaluate_task_method(
    task: TaskContext,
    *,
    method: str,
    args: argparse.Namespace,
    rainbow_agent: Any | None,
    rainbow_encoder: Any | None,
) -> dict[str, Any]:
    from env.dsl_env import run_single_task_search
    from llm.llm_client import LocalLLMClient

    config = build_search_config(method, args)
    client = LocalLLMClient(base_url=args.llm_base_url)
    verifier = EvalVisibleTestVerifier(
        task=task,
        client=client,
        run_tests=True,
        python_bin=config.python_bin,
        timeout_s=config.timeout_s,
        compiler_temp=config.compiler_temp,
        allow_full_file_fallback=config.allow_full_file_fallback,
        max_verified_plans=config.max_verified_plans_per_task,
    )
    agent = rainbow_agent if method == "rainbow" else None
    encoder = rainbow_encoder if method == "rainbow" else None
    started_at = time.perf_counter()
    result = run_single_task_search(
        task,
        config,
        policy=METHOD_TO_POLICY[method],
        client=client,
        verifier=verifier,
        agent=agent,
        encoder=encoder,
        seed=args.seed,
    )
    elapsed_time_ms = (time.perf_counter() - started_at) * 1000.0
    fallback_verification = None if verifier.latest_result is None else verifier.latest_result.model_dump()
    best_verification = result.best_verification or fallback_verification or {}
    hidden_status = best_verification.get("hidden_test_passed")
    visible_status = best_verification.get("visible_test_passed")
    solved = int((hidden_status is True) if task.hidden_test_file else (visible_status is True))
    visible_counts = verifier.counts_for_signature(best_verification.get("plan_signature"), solved=bool(visible_status))
    hidden_counts = verifier.hidden_counts_for_signature(best_verification.get("plan_signature"), solved=bool(hidden_status))
    tests_total = visible_counts.total + hidden_counts.total
    tests_passed = visible_counts.passed + hidden_counts.passed
    fraction_tests_passed = (tests_passed / tests_total) if tests_total else (1.0 if solved else 0.0)
    return {
        "kind": "task_eval_record_v1",
        "task_id": task.task_id,
        "family": task.family,
        "method": method,
        "policy": METHOD_TO_POLICY[method],
        "seed": args.seed,
        "solved": solved,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "fraction_tests_passed": fraction_tests_passed,
        "visible_test_passed": visible_status,
        "hidden_test_passed": hidden_status,
        "visible_tests_passed": visible_counts.passed,
        "visible_tests_total": visible_counts.total,
        "hidden_tests_passed": hidden_counts.passed,
        "hidden_tests_total": hidden_counts.total,
        "elapsed_time_ms": elapsed_time_ms,
        "llm_input_tokens": client.total_prompt_tokens,
        "llm_output_tokens": client.total_completion_tokens,
        "llm_total_tokens": client.total_tokens,
        "branches_explored": count_branches_explored(result),
        "steps_to_success": steps_to_success(result),
        "compile_success": best_verification.get("compile_success"),
        "compile_error": best_verification.get("compile_error"),
        "visible_test_stdout": best_verification.get("visible_test_stdout"),
        "visible_test_stderr": best_verification.get("visible_test_stderr"),
        "hidden_test_stdout": best_verification.get("hidden_test_stdout"),
        "hidden_test_stderr": best_verification.get("hidden_test_stderr"),
        "best_bank_id": result.best_bank_id,
        "visible_test_file": task.visible_test_file,
        "hidden_test_file": task.hidden_test_file,
        "plan_bank_summary": result.plan_bank.get("summary", {}),
        "verifier_summary": result.verifier_summary,
    }


def run_evaluations(args: argparse.Namespace, raw_output_path: Path) -> list[Path]:
    from data.task_store import TaskStore

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    replace_methods = set(args.replace_methods or [])
    existing_records = load_existing_records(raw_output_path)
    preserve_fields = set(args.preserve_fields or [])
    if replace_methods:
        remove_method_records(raw_output_path, replace_methods)
    if args.task_manifest is None:
        raise SystemExit("An eval task manifest is required; this pipeline only runs on the held-out eval split.")
    task_store = TaskStore.from_manifest(args.task_manifest, limit=args.num_tasks)
    rainbow_agent = None
    rainbow_encoder = None
    if "rainbow" in args.methods:
        rainbow_agent, rainbow_encoder = load_rainbow_artifacts(args.checkpoint)
    completed_keys = load_completed_run_keys(raw_output_path) if args.resume else set()
    run_plan: list[tuple[Any, str]] = []
    for task in task_store.iter_contexts():
        for method in args.methods:
            key = (task.task_id, method, args.seed)
            if key in completed_keys:
                continue
            run_plan.append((task, method))
    total_runs = len(run_plan)
    progress = None
    if not args.no_progress:
        progress = tqdm(total=total_runs, desc="Eval", unit="run", dynamic_ncols=True)
    file_mode = choose_raw_output_file_mode(
        raw_output_exists=raw_output_path.exists(),
        resume=args.resume,
        replace_methods=replace_methods,
        preserve_fields=preserve_fields,
    )
    with raw_output_path.open(file_mode, encoding="utf-8") as handle:
        try:
            for task, method in run_plan:
                if progress is not None:
                    progress.set_postfix(task=task.task_id, method=method, refresh=False)
                started_at = time.perf_counter()
                try:
                    record = evaluate_task_method(
                        task,
                        method=method,
                        args=args,
                        rainbow_agent=rainbow_agent,
                        rainbow_encoder=rainbow_encoder,
                    )
                except Exception as exc:  # pragma: no cover - exercised only on live evaluation failures
                    record = {
                        "kind": "task_eval_error_v1",
                        "task_id": task.task_id,
                        "family": task.family,
                        "method": method,
                        "policy": METHOD_TO_POLICY[method],
                        "seed": args.seed,
                        "error": str(exc),
                    }
                    _warn(f"Evaluation failed for task={task.task_id} method={method}: {exc}")
                record = merge_preserved_fields(
                    record,
                    existing_records.get((task.task_id, method, args.seed)),
                    preserve_fields,
                )
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        task=task.task_id,
                        method=method,
                        solved=record.get("solved", "-"),
                        elapsed_s=f"{time.perf_counter() - started_at:.1f}",
                        refresh=False,
                    )
        finally:
            if progress is not None:
                progress.close()
    return [raw_output_path]


def iter_raw_objects(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".jsonl":
        objects: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise PipelineWarning(f"{path}:{line_number} is not valid JSON: {exc}") from exc
                if isinstance(payload, dict):
                    objects.append(payload)
                else:
                    raise PipelineWarning(f"{path}:{line_number} did not contain a JSON object")
        return objects
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("strategies"), list):
            return [item for item in payload["strategies"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            return [payload]
        raise PipelineWarning(f"{path} did not contain a JSON object or list of objects")
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise PipelineWarning(f"Unsupported raw input type: {path}")


def sanitize_canonical_row(raw: dict[str, Any]) -> dict[str, Any] | None:
    task_id = raw.get("task_id")
    method = canonical_method_name(raw.get("method") or raw.get("policy"))
    solved = _coerce_int(raw.get("solved"))
    tests_passed = _coerce_int(raw.get("tests_passed"))
    tests_total = _coerce_int(raw.get("tests_total"))
    llm_input_tokens = _coerce_int(raw.get("llm_input_tokens"))
    llm_output_tokens = _coerce_int(raw.get("llm_output_tokens"))
    llm_total_tokens = _coerce_int(raw.get("llm_total_tokens"))
    fraction_tests_passed = _coerce_float(raw.get("fraction_tests_passed"))
    elapsed_time_ms = _coerce_float(raw.get("elapsed_time_ms"))
    branches_explored = _coerce_int(raw.get("branches_explored"))
    steps_to_success_value = _coerce_int(raw.get("steps_to_success"))
    seed = _coerce_int(raw.get("seed"))

    if not isinstance(task_id, str) or not task_id:
        raise PipelineWarning("Missing task_id")
    if method is None:
        raise PipelineWarning(f"Unsupported or missing method for task {task_id}")
    if solved is None:
        raise PipelineWarning(f"Missing solved flag for task {task_id} method {method}")
    solved = 1 if solved else 0
    if llm_total_tokens is None and llm_input_tokens is not None and llm_output_tokens is not None:
        llm_total_tokens = llm_input_tokens + llm_output_tokens
    if tests_total is not None and tests_passed is None:
        tests_passed = tests_total if solved else 0
    if fraction_tests_passed is None:
        if tests_total and tests_passed is not None:
            fraction_tests_passed = tests_passed / tests_total
        elif solved:
            fraction_tests_passed = 1.0
        elif tests_total == 0:
            fraction_tests_passed = 0.0
    if tests_passed is not None and tests_total is not None and tests_passed > tests_total:
        raise PipelineWarning(f"tests_passed exceeded tests_total for task {task_id} method {method}")
    return {
        "task_id": task_id,
        "method": method,
        "solved": solved,
        "tests_passed": tests_passed,
        "tests_total": tests_total,
        "fraction_tests_passed": fraction_tests_passed,
        "elapsed_time_ms": elapsed_time_ms,
        "llm_input_tokens": llm_input_tokens,
        "llm_output_tokens": llm_output_tokens,
        "llm_total_tokens": llm_total_tokens,
        "branches_explored": branches_explored,
        "steps_to_success": steps_to_success_value,
        "seed": seed,
    }


def normalize_raw_object(raw: dict[str, Any]) -> dict[str, Any] | None:
    if raw.get("kind") == "task_eval_error_v1":
        raise PipelineWarning(
            f"Dropping failed evaluation record for task {raw.get('task_id')} "
            f"method {raw.get('method')}: {raw.get('error')}"
        )
    if raw.get("kind") == "task_eval_record_v1":
        return sanitize_canonical_row(raw)
    if "strategy" in raw and isinstance(raw["strategy"], dict):
        strategy = raw["strategy"]
        merged = {
            "task_id": strategy.get("task_id"),
            "method": strategy.get("policy"),
            "solved": int(strategy.get("visible_test_passed") is True),
            "tests_passed": raw.get("tests_passed"),
            "tests_total": raw.get("tests_total"),
            "fraction_tests_passed": raw.get("fraction_tests_passed"),
            "elapsed_time_ms": (
                (_coerce_float(strategy.get("elapsed_s")) or 0.0) * 1000.0
                if strategy.get("elapsed_s") is not None
                else raw.get("elapsed_time_ms")
            ),
            "llm_input_tokens": strategy.get("prompt_tokens"),
            "llm_output_tokens": strategy.get("completion_tokens"),
            "llm_total_tokens": strategy.get("total_tokens"),
            "branches_explored": raw.get("branches_explored"),
            "steps_to_success": raw.get("steps_to_success"),
            "seed": raw.get("seed"),
        }
        return sanitize_canonical_row(merged)
    if raw.get("policy") and raw.get("task_id") and "solved" not in raw:
        raise PipelineWarning(
            f"Raw record for task {raw.get('task_id')} method {raw.get('policy')} "
            "did not include task-level solved/test fields"
        )
    return sanitize_canonical_row(raw)


def load_normalized_rows(raw_paths: list[Path]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    for path in raw_paths:
        try:
            raw_objects = iter_raw_objects(path)
        except PipelineWarning as exc:
            _warn(str(exc))
            continue
        for raw_object in raw_objects:
            try:
                row = normalize_raw_object(raw_object)
            except PipelineWarning as exc:
                _warn(str(exc))
                continue
            if row is None:
                continue
            key = (row["task_id"], row["method"], row["seed"])
            if key in deduped:
                _warn(
                    f"Duplicate normalized row for task={row['task_id']} method={row['method']} "
                    f"seed={row['seed']}; keeping the last occurrence"
                )
            deduped[key] = row
    rows = sorted(
        deduped.values(),
        key=lambda item: (METHOD_ORDER.index(item["method"]), item["task_id"], item["seed"] or -1),
    )
    return rows


def _csv_ready(row: dict[str, Any], headers: list[str]) -> dict[str, Any]:
    ready: dict[str, Any] = {}
    for header in headers:
        value = row.get(header)
        if value is None:
            ready[header] = ""
        elif isinstance(value, float):
            ready[header] = format_float(value, digits=6)
        else:
            ready[header] = value
    return ready


def write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_ready(row, headers))


def load_completed_run_keys(path: Path) -> set[tuple[str, str, int | None]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int | None]] = set()
    try:
        raw_objects = iter_raw_objects(path)
    except (FileNotFoundError, PipelineWarning):
        return set()
    for raw_object in raw_objects:
        if raw_object.get("kind") != "task_eval_record_v1":
            continue
        task_id = raw_object.get("task_id")
        method = canonical_method_name(raw_object.get("method") or raw_object.get("policy"))
        seed = _coerce_int(raw_object.get("seed"))
        if isinstance(task_id, str) and method is not None:
            completed.add((task_id, method, seed))
    return completed


def remove_method_records(path: Path, methods: set[str]) -> None:
    if not path.exists():
        return
    kept_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                kept_lines.append(line)
                continue
            method = canonical_method_name(payload.get("method") or payload.get("policy"))
            if payload.get("kind") == "task_eval_record_v1" and method in methods:
                continue
            kept_lines.append(line)
    with path.open("w", encoding="utf-8") as handle:
        handle.writelines(kept_lines)


def load_existing_records(path: Path) -> dict[tuple[str, str, int | None], dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[tuple[str, str, int | None], dict[str, Any]] = {}
    try:
        raw_objects = iter_raw_objects(path)
    except (FileNotFoundError, PipelineWarning):
        return {}
    for raw_object in raw_objects:
        if raw_object.get("kind") != "task_eval_record_v1":
            continue
        task_id = raw_object.get("task_id")
        method = canonical_method_name(raw_object.get("method") or raw_object.get("policy"))
        seed = _coerce_int(raw_object.get("seed"))
        if isinstance(task_id, str) and method is not None:
            existing[(task_id, method, seed)] = raw_object
    return existing


def merge_preserved_fields(
    record: dict[str, Any],
    existing_record: dict[str, Any] | None,
    preserve_fields: set[str],
) -> dict[str, Any]:
    if existing_record is None or not preserve_fields:
        return record
    merged = dict(record)
    for field in preserve_fields:
        if field in existing_record:
            merged[field] = existing_record[field]
    return merged


def choose_raw_output_file_mode(
    *,
    raw_output_exists: bool,
    resume: bool,
    replace_methods: set[str],
    preserve_fields: set[str],
) -> str:
    append_mode = raw_output_exists and (resume or bool(replace_methods) or bool(preserve_fields))
    return "a" if append_mode else "w"


def _numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        coerced = _coerce_float(value)
        if coerced is not None:
            values.append(coerced)
    return values


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def compute_summary_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHOD_ORDER}
    for row in rows:
        rows_by_method.setdefault(row["method"], []).append(row)
    summaries: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        method_rows = rows_by_method.get(method, [])
        solved_tasks = sum(int(row["solved"]) for row in method_rows)
        elapsed_values = _numeric_values(method_rows, "elapsed_time_ms")
        fraction_values = _numeric_values(method_rows, "fraction_tests_passed")
        token_values = _numeric_values(method_rows, "llm_total_tokens")
        branch_values = _numeric_values(method_rows, "branches_explored")
        solved_step_values = [
            float(row["steps_to_success"])
            for row in method_rows
            if row.get("steps_to_success") is not None and int(row["solved"]) == 1
        ]
        total_elapsed = sum(elapsed_values)
        total_tokens = sum(token_values)
        task_count = len(method_rows)
        summaries.append(
            {
                "method": method,
                "task_count": task_count,
                "solved_tasks": solved_tasks,
                "solve_rate": (solved_tasks / task_count) if task_count else None,
                "mean_fraction_tests_passed": mean_or_none(fraction_values),
                "median_fraction_tests_passed": median_or_none(fraction_values),
                "mean_elapsed_time_ms": mean_or_none(elapsed_values),
                "median_elapsed_time_ms": median_or_none(elapsed_values),
                "mean_llm_total_tokens": mean_or_none(token_values),
                "median_llm_total_tokens": median_or_none(token_values),
                "mean_branches_explored": mean_or_none(branch_values),
                "mean_steps_to_success": mean_or_none(solved_step_values),
                "tokens_per_solved_task": (total_tokens / solved_tasks) if solved_tasks else None,
                "time_per_solved_task": (total_elapsed / solved_tasks) if solved_tasks else None,
            }
        )
    return summaries


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    if values.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    sample_means = values[indices].mean(axis=1)
    low, high = np.quantile(sample_means, [0.025, 0.975])
    return (float(low), float(high))


def exact_mcnemar_p_value(rainbow: Any, baseline: Any) -> float:
    rainbow_array = np.asarray(rainbow, dtype=int)
    baseline_array = np.asarray(baseline, dtype=int)
    better = int(np.sum((rainbow_array == 1) & (baseline_array == 0)))
    worse = int(np.sum((rainbow_array == 0) & (baseline_array == 1)))
    discordant = better + worse
    if discordant == 0:
        return 1.0
    smaller = min(better, worse)
    cumulative = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * cumulative))


def exact_sign_flip_p_value(deltas: Any) -> float:
    deltas_array = np.asarray(deltas, dtype=float)
    if deltas_array.size == 0 or np.allclose(deltas_array, 0.0):
        return 1.0
    observed = abs(float(np.mean(deltas_array)))
    if deltas_array.size <= 16:
        total = 0
        extreme = 0
        for signs in product((-1.0, 1.0), repeat=deltas_array.size):
            signed = deltas_array * np.asarray(signs, dtype=float)
            statistic = abs(float(np.mean(signed)))
            total += 1
            if statistic >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = np.random.default_rng(12345)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(200000, deltas_array.size))
    statistics = np.abs((signs * deltas_array).mean(axis=1))
    return float(np.mean(statistics >= observed - 1e-12))


def task_metric_direction(metric_name: str, delta: float) -> bool:
    if metric_name in {"solved", "fraction_tests_passed"}:
        return delta > 0
    if metric_name in {"elapsed_time_ms", "llm_total_tokens"}:
        return delta < 0
    return delta > 0


def build_pairwise_summary(record: dict[str, Any]) -> str:
    baseline_name = record["baseline"].replace("_", " ")
    metric = record["metric"]
    mean_delta = float(record["mean_delta"])
    ci_low = float(record["ci_low"])
    ci_high = float(record["ci_high"])
    p_text = format_p_value(record.get("p_value"))
    if metric == "solved":
        direction = "matches" if abs(mean_delta) <= 1e-12 else ("improves" if mean_delta >= 0 else "reduces")
        return (
            f"Rainbow {direction} solve rate by {mean_delta:+.2f} "
            f"(95% CI [{ci_low:.2f}, {ci_high:.2f}], {p_text}) vs {baseline_name}."
        )
    if metric == "fraction_tests_passed":
        direction = "matches" if abs(mean_delta) <= 1e-12 else ("improves" if mean_delta >= 0 else "reduces")
        return (
            f"Rainbow {direction} mean fraction of tests passed by {mean_delta:+.2f} "
            f"(95% CI [{ci_low:.2f}, {ci_high:.2f}], {p_text}) vs {baseline_name}."
        )
    if metric == "elapsed_time_ms":
        direction = "reduces" if mean_delta <= 0 else "increases"
        magnitude = abs(mean_delta)
        low = abs(ci_high) if mean_delta <= 0 else abs(ci_low)
        high = abs(ci_low) if mean_delta <= 0 else abs(ci_high)
        return (
            f"Rainbow {direction} elapsed time by {magnitude:,.0f} ms "
            f"(95% CI [{low:,.0f}, {high:,.0f}] ms, {p_text}) vs {baseline_name}."
        )
    direction = "reduces" if mean_delta <= 0 else "increases"
    magnitude = abs(mean_delta)
    low = abs(ci_high) if mean_delta <= 0 else abs(ci_low)
    high = abs(ci_low) if mean_delta <= 0 else abs(ci_high)
    return (
        f"Rainbow {direction} total tokens by {magnitude:,.0f} "
        f"(95% CI [{low:,.0f}, {high:,.0f}], {p_text}) vs {baseline_name}."
    )


def compute_paired_deltas(
    rows: list[dict[str, Any]],
    *,
    resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows_by_task_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rows_by_task_method.setdefault(row["task_id"], {})[row["method"]] = row
    paired_rows: list[dict[str, Any]] = []
    for baseline in ["heuristic", "random", "one_shot"]:
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
                elif task_metric_direction(metric["name"], float(delta)):
                    rainbow_better_tasks += 1
                else:
                    baseline_better_tasks += 1
            record = {
                "baseline": baseline,
                "metric": metric["name"],
                "n_tasks": int(deltas.size),
                "rainbow_mean": float(np.mean(rainbow_array)),
                "baseline_mean": float(np.mean(baseline_array)),
                "mean_delta": float(np.mean(deltas)),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_value": float(p_value),
                "test_name": test_name,
                "rainbow_better_tasks": rainbow_better_tasks,
                "baseline_better_tasks": baseline_better_tasks,
                "tied_tasks": tied_tasks,
            }
            record["summary"] = build_pairwise_summary(record)
            paired_rows.append(record)
    return paired_rows


def configure_plot_style() -> None:
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


def _bar_value_formatter(metric_key: str) -> FuncFormatter | PercentFormatter:
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        return PercentFormatter(xmax=1.0, decimals=0)
    if metric_key == "mean_branches_explored":
        return FuncFormatter(lambda value, _: f"{value:.1f}")
    return FuncFormatter(lambda value, _: f"{value:,.0f}")


def _format_bar_label(metric_key: str, value: float) -> str:
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        return f"{value:.1%}"
    if metric_key == "mean_branches_explored":
        return f"{value:.1f}"
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
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
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
    bars = ax.bar(
        positions,
        values,
        color=[METHOD_COLORS[method] for method in METHOD_ORDER],
        width=0.66,
        zorder=3,
    )
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
    ax.set_xticks(positions, [display_method_name(method) for method in METHOD_ORDER])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(_bar_value_formatter(metric_key))
    ax.grid(axis="y", color="#E7ECF2", linewidth=1.0, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if metric_key in {"solve_rate", "mean_fraction_tests_passed"}:
        upper = max(max(values) + max(error_high, default=0.0), 1.0)
        ax.set_ylim(0.0, min(1.05, upper + 0.05))
    else:
        upper = max(values[i] + error_high[i] for i in range(len(values))) if values else 1.0
        ax.set_ylim(0.0, upper * 1.18 if upper > 0 else 1.0)
    for index, (bar, value) in enumerate(zip(bars, values)):
        label = _format_bar_label(metric_key, value)
        cap_top = value + error_high[index] if error_high else value
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            cap_top + ax.get_ylim()[1] * 0.03,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2328",
        )
    fig.tight_layout()
    export_figure(fig, output_stem)


def plot_pairwise_delta(baseline: str, paired_rows: list[dict[str, Any]], *, output_stem: Path) -> None:
    configure_plot_style()
    baseline_rows = {row["metric"]: row for row in paired_rows if row["baseline"] == baseline}
    metric_order = [metric["name"] for metric in PAIRWISE_METRICS]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8))
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
        favorable = task_metric_direction(metric_name, mean_delta)
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
            span = {
                "solved": 0.1,
                "fraction_tests_passed": 0.1,
                "elapsed_time_ms": 1000.0,
                "llm_total_tokens": 100.0,
            }[metric_name]
        axis.set_xlim(-1.15 * span, 1.15 * span)
        axis.set_title(
            f"{metric_meta['label']} ({'higher' if metric_meta['better'] == 'higher' else 'lower'} better)",
            fontsize=12,
        )
        axis.text(
            0.02,
            0.92,
            format_p_value(row.get("p_value")),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#1F2328",
        )
        axis.text(
            0.02,
            0.10,
            f"Δ={format_delta(mean_delta, metric_name)}\n95% CI [{format_delta(ci_low, metric_name)}, {format_delta(ci_high, metric_name)}]",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="#1F2328",
        )
        if metric_name in {"solved", "fraction_tests_passed"}:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        else:
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    fig.suptitle(f"Rainbow vs {display_method_name(baseline)}", fontsize=16, y=1.02)
    fig.tight_layout()
    export_figure(fig, output_stem)


def render_summary_table(summary_rows: list[dict[str, Any]]) -> str:
    header = [
        "Method",
        "Solve rate",
        "Mean tests passed",
        "Mean time",
        "Mean tokens",
        "Tokens / solved",
        "Time / solved",
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
                    display_method_name(row["method"]),
                    format_metric_value(_coerce_float(row["solve_rate"]), "solve_rate"),
                    format_metric_value(_coerce_float(row["mean_fraction_tests_passed"]), "mean_fraction_tests_passed"),
                    format_metric_value(_coerce_float(row["mean_elapsed_time_ms"]), "mean_elapsed_time_ms"),
                    format_metric_value(_coerce_float(row["mean_llm_total_tokens"]), "mean_llm_total_tokens"),
                    format_metric_value(_coerce_float(row["tokens_per_solved_task"]), "tokens_per_solved_task"),
                    format_metric_value(_coerce_float(row["time_per_solved_task"]), "time_per_solved_task"),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def build_results_report(
    *,
    docs_dir: Path,
    markdown_dir: Path,
    png_dir: Path,
    summary_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    normalized_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    solve_rate_rows = {row["method"]: row for row in summary_rows}
    method_ranking = sorted(
        summary_rows,
        key=lambda row: (_coerce_float(row.get("solve_rate")) or 0.0, _coerce_float(row.get("mean_fraction_tests_passed")) or 0.0),
        reverse=True,
    )
    best_method = display_method_name(method_ranking[0]["method"]) if method_ranking else "n/a"
    solve_summaries = [
        row["summary"]
        for row in paired_rows
        if row["metric"] == "solved"
    ]
    missing_pairs = []
    expected_methods = set(METHOD_ORDER)
    tasks = sorted({row["task_id"] for row in normalized_rows})
    task_methods: dict[str, set[str]] = {task_id: set() for task_id in tasks}
    for row in normalized_rows:
        task_methods[row["task_id"]].add(row["method"])
    for task_id, seen_methods in task_methods.items():
        if seen_methods != expected_methods:
            missing_pairs.append(f"- `{task_id}`: missing {', '.join(sorted(expected_methods - seen_methods))}")
    report_lines = [
        "# Eval Results",
        "",
        f"Source eval manifest: `{args.task_manifest.relative_to(ROOT) if args.task_manifest.is_relative_to(ROOT) else args.task_manifest}`",
        f"Rainbow checkpoint: `{args.checkpoint.relative_to(ROOT) if args.checkpoint.is_relative_to(ROOT) else args.checkpoint}`",
        f"Paired units: `{len(tasks)}` eval tasks",
        "",
        "## Summary",
        "",
        render_summary_table(summary_rows),
        "",
        "## Interpretation",
        "",
        f"`{best_method}` is the top method on the held-out eval split by solve rate, with the paired comparisons below computed task-by-task against Rainbow.",
        "",
        "## Paired Comparisons",
        "",
    ]
    report_lines.extend(f"- {summary}" for summary in solve_summaries)
    extra_pairwise = [
        row["summary"]
        for row in paired_rows
        if row["metric"] in {"fraction_tests_passed", "elapsed_time_ms", "llm_total_tokens"}
    ]
    report_lines.extend(f"- {summary}" for summary in extra_pairwise)
    report_lines.extend(
        [
            "",
            "## Graphs",
            "",
            "- [Solve rate](../png/solve_rate.png)",
            "- [Fraction tests passed](../png/fraction_tests_passed.png)",
            "- [Time per task](../png/time_per_task.png)",
            "- [Tokens per task](../png/tokens_per_task.png)",
            "- [Branches explored per task](../png/branches_per_task.png)",
            "- [Rainbow vs heuristic](../png/rainbow_vs_heuristic_delta.png)",
            "- [Rainbow vs random](../png/rainbow_vs_random_delta.png)",
            "- [Rainbow vs one-shot](../png/rainbow_vs_one_shot_delta.png)",
        ]
    )
    if missing_pairs:
        report_lines.extend(
            [
                "",
                "## Coverage Notes",
                "",
                "Some task/method combinations were missing from the normalized table:",
                *missing_pairs,
            ]
        )
    report_lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- `one_shot` is the exported name for the existing internal `oneshot` policy.",
            "- `tests_passed` and `tests_total` are measured from each task's visible eval tests.",
            "- Paired confidence intervals use 10,000 bootstrap resamples.",
            "- P-values use exact paired tests: McNemar for `solved`, and exact sign-flip randomization tests for the continuous paired deltas.",
        ]
    )
    return "\n".join(report_lines) + "\n"


def write_results_report(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> int:
    args = parse_args()
    docs_dir = args.docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir = docs_dir / "markdown"
    png_dir = docs_dir / "png"
    svg_dir = docs_dir / "svg"
    markdown_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    svg_dir.mkdir(parents=True, exist_ok=True)
    raw_output = args.raw_output or (docs_dir / "raw_eval_results.jsonl")
    raw_paths = list(args.raw_input or [])
    if not raw_paths:
        raw_paths = run_evaluations(args, raw_output)

    normalized_rows = load_normalized_rows(raw_paths)
    if not normalized_rows:
        raise SystemExit("No normalized evaluation rows were produced.")

    normalized_path = docs_dir / "normalized_results.csv"
    summary_path = docs_dir / "summary_metrics.csv"
    paired_path = docs_dir / "paired_deltas.csv"

    write_csv(normalized_path, NORMALIZED_HEADERS, normalized_rows)

    summary_rows = compute_summary_metrics(normalized_rows)
    write_csv(summary_path, SUMMARY_HEADERS, summary_rows)

    paired_rows = compute_paired_deltas(
        normalized_rows,
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    write_csv(paired_path, PAIRED_HEADERS, paired_rows)

    plot_summary_bars(
        normalized_rows,
        summary_rows,
        metric_key="solve_rate",
        title="Solve rate by method",
        ylabel="Solve rate",
        output_stem=svg_dir / "solve_rate",
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    plot_summary_bars(
        normalized_rows,
        summary_rows,
        metric_key="mean_fraction_tests_passed",
        title="Fraction tests passed by method",
        ylabel="Mean fraction tests passed",
        output_stem=svg_dir / "fraction_tests_passed",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 100,
    )
    plot_summary_bars(
        normalized_rows,
        summary_rows,
        metric_key="mean_elapsed_time_ms",
        title="Time per task by method",
        ylabel="Mean elapsed time (ms)",
        output_stem=svg_dir / "time_per_task",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 200,
    )
    plot_summary_bars(
        normalized_rows,
        summary_rows,
        metric_key="mean_llm_total_tokens",
        title="Tokens per task by method",
        ylabel="Mean total tokens",
        output_stem=svg_dir / "tokens_per_task",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 300,
    )
    plot_summary_bars(
        normalized_rows,
        summary_rows,
        metric_key="mean_branches_explored",
        title="Branches explored per task by method",
        ylabel="Mean branches explored",
        output_stem=svg_dir / "branches_per_task",
        resamples=args.bootstrap_resamples,
        seed=args.seed + 350,
    )
    for baseline in ["heuristic", "random", "one_shot"]:
        plot_pairwise_delta(
            baseline,
            paired_rows,
            output_stem=svg_dir / f"rainbow_vs_{baseline}_delta",
        )

    for generated_png in svg_dir.glob("*.png"):
        generated_png.replace(png_dir / generated_png.name)

    report_path = markdown_dir / "results.md"
    report = build_results_report(
        docs_dir=docs_dir,
        markdown_dir=markdown_dir,
        png_dir=png_dir,
        summary_rows=summary_rows,
        paired_rows=paired_rows,
        normalized_rows=normalized_rows,
        args=args,
    )
    write_results_report(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

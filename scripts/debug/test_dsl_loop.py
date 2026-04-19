#!/usr/bin/env python3
"""Debug the DSL loop against generated tasks and a local llama.cpp server.

Usage:
    conda run -n hyperreasoning python scripts/debug/test_dsl_loop.py \
        --tasks-root data/generated_tasks/hard --num-tasks 2 --k 3 --run-tests

    conda run -n hyperreasoning python scripts/debug/test_dsl_loop.py \
        --task-dir data/generated_tasks/hard/streaming_parser_reentrancy_2000 --k 3 --run-tests

The script expects task folders with:
    task.json
    source files
    test_visible.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifacts.io import make_attempt_id, make_run_id, save_attempt_record, save_raw_text
from artifacts.records import AttemptRecord, PlanExecutionRecord, PlanRecord
from llm.compiler import CompilePlanError, compile_plan_to_code
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import TaskContext, load_task_context as _load_task_context
from llm.proposal import plan_signature, propose_dsl_candidates


SEPARATOR = "-" * 40


@dataclass(slots=True)
class TestRunResult:
    passed: bool
    stdout: str
    stderr: str
    return_code: int
    command: list[str]


@dataclass(slots=True)
class PlanResult:
    index: int
    plan_id: str
    strategy: str
    modified_files: list[str]
    compile_error: str | None
    test_result: TestRunResult | None
    compiled_files: dict[str, str]
    raw_compiler_prompt: str | None = None
    raw_compiler_response: str | None = None
    attempted_compile: bool = True
    compile_latency_s: float | None = None
    test_latency_s: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", type=Path, default=None, help="Single task directory to run")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=ROOT / "data/generated_tasks/hard",
        help="Directory containing many task folders",
    )
    parser.add_argument("--num-tasks", type=int, default=3, help="Number of tasks to run from --tasks-root")
    parser.add_argument("--k", type=int, default=3, help="Number of DSL plans to request")
    parser.add_argument("--run-tests", action="store_true", help="Run visible tests for each compiled candidate")
    parser.add_argument("--proposal-temp", type=float, default=0.7, help="Temperature for proposal generation")
    parser.add_argument("--compile-temp", type=float, default=0.2, help="Temperature for compilation")
    parser.add_argument(
        "--proposal-source",
        choices=["heuristic", "llm", "hybrid"],
        default="heuristic",
        help="Source of DSL proposals; heuristic is cheapest for bulk data generation",
    )
    parser.add_argument(
        "--compile-top-n",
        type=int,
        default=1,
        help="Compile only the top N proposals while still saving all proposed plans",
    )
    parser.add_argument(
        "--allow-full-file-fallback",
        action="store_true",
        help="Allow expensive full-file compiler fallback after edit-mode failure",
    )
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8080", help="Local llama.cpp server URL")
    parser.add_argument("--python-bin", default=sys.executable, help="Python interpreter used for tests")
    parser.add_argument("--timeout", type=float, default=12.0, help="Timeout for visible test execution")
    parser.add_argument("--save-artifacts", action="store_true", help="Persist one attempt artifact per task")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts/smoke/debug_runs",
        help="Directory root for saved attempt artifacts",
    )
    parser.add_argument("--save-raw-llm", action="store_true", help="Save raw prompts/responses under artifacts")
    parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    parser.add_argument(
        "--print-raw-llm",
        action="store_true",
        help="Print full raw LLM proposal/compiler responses immediately after each call",
    )
    return parser.parse_args()


def load_task_context(task_dir: Path) -> TaskContext:
    """Load one task directory into a prompt/test-friendly context."""

    return _load_task_context(task_dir)


def to_repo_relative(path: Path) -> str:
    """Convert a path to a repo-relative POSIX string when possible."""

    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_task_dirs(task_dir: Path | None, tasks_root: Path, num_tasks: int) -> list[Path]:
    """Resolve one or more task directories from CLI args."""

    if task_dir is not None:
        return [task_dir]
    candidates = [
        path
        for path in sorted(tasks_root.iterdir())
        if path.is_dir() and (path / "task.json").exists()
    ]
    return candidates[:num_tasks]


def pretty_print_plans(plans: list[Any]) -> str:
    """Render detailed DSL plan summaries for debugging."""

    lines: list[str] = []
    for index, plan in enumerate(plans, start=1):
        lines.append(SEPARATOR)
        lines.append(f"Plan {index}")
        lines.append(SEPARATOR)
        lines.append(f"Strategy: {plan.strategy}")
        lines.append(f"Target files: {', '.join(plan.target_files)}")
        lines.append(f"Suspected bug types: {', '.join(plan.suspected_bug_types) or '-'}")
        lines.append(f"Invariants: {', '.join(plan.invariants)}")
        lines.append(f"Validation checks: {', '.join(plan.validation_checks)}")
        lines.append(f"Risks: {', '.join(plan.risks) or '-'}")
        lines.append(f"Touched symbols: {', '.join(plan.touched_symbols) or '-'}")
        lines.append(f"Edit style: {plan.edit_style}")
        lines.append(f"Signature: {plan_signature(plan)}")
    return "\n".join(lines)


def create_temp_workspace(task_dir: Path) -> Path:
    """Create a temp workspace containing a full copy of the task directory."""

    temp_root = Path(tempfile.mkdtemp(prefix=f"dsl_loop_{task_dir.name}_"))
    workspace_dir = temp_root / task_dir.name
    shutil.copytree(task_dir, workspace_dir)
    return workspace_dir


def apply_compiled_files(workspace_dir: Path, compiled_files: dict[str, str]) -> Path:
    """Overwrite compiled files inside a copied task workspace."""

    for relative_path, content in compiled_files.items():
        destination = workspace_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    return workspace_dir


def preview_compiled_files(compiled_files: dict[str, str], *, max_lines: int = 20) -> str:
    """Render a short file preview for debugging."""

    chunks: list[str] = []
    for path, content in compiled_files.items():
        preview_lines = content.splitlines()[:max_lines]
        chunks.append(SEPARATOR)
        chunks.append(f"Preview: {path}")
        chunks.append(SEPARATOR)
        chunks.extend(preview_lines)
    return "\n".join(chunks)


def run_visible_tests(workspace_dir: Path, *, python_bin: str, timeout: float = 12.0) -> TestRunResult:
    """Run visible tests with a simple subprocess strategy."""

    visible_test = workspace_dir / "test_visible.py"
    if not visible_test.exists():
        return TestRunResult(
            passed=False,
            stdout="",
            stderr="Missing test_visible.py",
            return_code=1,
            command=[],
        )

    commands = [
        [python_bin, "-m", "unittest", "test_visible.py"],
        [python_bin, "test_visible.py"],
    ]
    last_result: TestRunResult | None = None
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            result = TestRunResult(
                passed=completed.returncode == 0,
                stdout=completed.stdout,
                stderr=completed.stderr,
                return_code=completed.returncode,
                command=command,
            )
            if result.passed:
                return result
            last_result = result
        except subprocess.TimeoutExpired as exc:
            last_result = TestRunResult(
                passed=False,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\nTimed out after {timeout} seconds",
                return_code=124,
                command=command,
            )
        except Exception as exc:  # pragma: no cover - debugging safety net
            last_result = TestRunResult(
                passed=False,
                stdout="",
                stderr=f"Test runner crashed: {exc}",
                return_code=1,
                command=command,
            )
    assert last_result is not None
    return last_result


def summarize_task_results(task: TaskContext, results: list[PlanResult]) -> str:
    """Build a task-level summary."""

    passing = [result for result in results if result.test_result is not None and result.test_result.passed]
    compile_failures = sum(1 for result in results if result.compile_error is not None)
    best_plan = choose_best_plan(results)
    best_plan_index = None if best_plan is None else best_plan.index
    return "\n".join(
        [
            "Task summary:",
            f"- task_id: {task.task_id}",
            f"- best plan index: {best_plan_index if best_plan_index is not None else 'none'}",
            f"- number of passing plans: {len(passing)}",
            f"- number of compile failures: {compile_failures}",
            f"- number of compiled plans: {sum(1 for result in results if result.attempted_compile)}",
            f"- total plans attempted: {len(results)}",
        ]
    )


def compute_score(result: PlanResult) -> float:
    """Temporary ranking score for one plan execution."""

    score = 0.0
    if result.compile_error is None:
        score += 0.25
    if result.test_result is not None and result.test_result.passed:
        score += 1.0
    return score


def choose_best_plan(results: list[PlanResult]) -> PlanResult | None:
    """Pick the highest-scoring plan, preferring lower plan index on ties."""

    if not results:
        return None
    return max(results, key=lambda item: (compute_score(item), -item.index))


def difficulty_from_task(task: TaskContext) -> str | None:
    """Extract the task difficulty from task.json if present."""

    raw = task.task_json.get("difficulty")
    return str(raw) if raw is not None else None


def to_plan_record(plan: Any) -> PlanRecord:
    """Convert a DSL plan into a persisted plan record."""

    return PlanRecord.model_validate(
        {
            "plan_id": plan.plan_id,
            "strategy": plan.strategy,
            "target_files": list(plan.target_files),
            "suspected_bug_types": list(plan.suspected_bug_types),
            "invariants": list(plan.invariants),
            "subgoals": list(plan.subgoals),
            "validation_checks": list(plan.validation_checks),
            "risks": list(plan.risks),
            "touched_symbols": list(plan.touched_symbols),
            "edit_style": plan.edit_style,
            "confidence": plan.confidence,
            "notes": plan.notes,
        }
    )


def to_execution_record(result: PlanResult) -> PlanExecutionRecord:
    """Convert one plan result into a persisted execution record."""

    test_result = result.test_result
    return PlanExecutionRecord.model_validate(
        {
            "plan_id": result.plan_id,
            "compile_success": result.attempted_compile and result.compile_error is None,
            "attempted_compile": result.attempted_compile,
            "compile_error": result.compile_error,
            "compiled_files": result.compiled_files,
            "files_changed": result.modified_files,
            "compile_latency_s": result.compile_latency_s,
            "visible_test_passed": None if test_result is None else test_result.passed,
            "visible_test_returncode": None if test_result is None else test_result.return_code,
            "visible_test_stdout": None if test_result is None else test_result.stdout,
            "visible_test_stderr": None if test_result is None else test_result.stderr,
            "test_latency_s": result.test_latency_s,
            "hidden_summary": None,
            "score": None if not result.attempted_compile else compute_score(result),
        }
    )


def save_attempt_artifact(
    *,
    task: TaskContext,
    run_id: str,
    attempt_index: int,
    plans: list[Any],
    results: list[PlanResult],
    artifacts_dir: Path,
    save_raw_llm: bool,
    raw_proposal_prompt: str | None,
    raw_proposal_response: str | None,
    proposal_source: str,
    proposal_latency_s: float | None,
) -> Path:
    """Build and save one persistent attempt artifact."""

    best = choose_best_plan(results)
    attempt_id = make_attempt_id(task.task_id, attempt_index)
    summary = {
        "num_plans": len(plans),
        "num_compile_success": sum(
            1 for result in results if result.attempted_compile and result.compile_error is None
        ),
        "num_visible_test_pass": sum(
            1
            for result in results
            if result.test_result is not None and result.test_result.passed
        ),
        "num_compiled_plans": sum(1 for result in results if result.attempted_compile),
        "best_plan_index": None if best is None else best.index,
        "best_plan_id": None if best is None else best.plan_id,
        "best_score": None if best is None else compute_score(best),
        "had_any_valid_plan": bool(plans),
        "had_any_compiled_output": any(bool(result.compiled_files) for result in results),
        "proposal_source": proposal_source,
        "proposal_latency_s": proposal_latency_s,
    }

    record = AttemptRecord.model_validate(
        {
            "run_id": run_id,
            "attempt_id": attempt_id,
            "task_id": task.task_id,
            "family": task.family,
            "difficulty": difficulty_from_task(task),
            "task_dir": to_repo_relative(task.task_dir),
            "task_prompt": task.prompt,
            "language": task.language,
            "metadata": task.metadata,
            "dsl_candidates": [to_plan_record(plan) for plan in plans],
            "selected_plan_id": None if best is None else best.plan_id,
            "selected_plan_index": None if best is None else best.index,
            "plan_executions": [to_execution_record(result) for result in results],
            "summary": summary,
        }
    )
    attempt_dir = save_attempt_record(record, artifacts_dir)

    if save_raw_llm:
        raw_dir = attempt_dir / "raw"
        if raw_proposal_prompt:
            save_raw_text(raw_dir / "proposal_prompt.txt", raw_proposal_prompt)
        if raw_proposal_response:
            save_raw_text(raw_dir / "proposal_response.txt", raw_proposal_response)
        for result in results:
            if result.raw_compiler_prompt:
                save_raw_text(raw_dir / f"compiler_prompt_plan_{result.index - 1}.txt", result.raw_compiler_prompt)
            if result.raw_compiler_response:
                save_raw_text(raw_dir / f"compiler_response_plan_{result.index - 1}.txt", result.raw_compiler_response)

    return attempt_dir


def print_plan_result(result: PlanResult) -> None:
    """Print one plan result block."""

    print(SEPARATOR)
    print(f"PLAN {result.index}")
    print(SEPARATOR)
    print(f"Strategy: {result.strategy}")
    print(f"Plan ID: {result.plan_id}")
    print(f"Files changed: {', '.join(result.modified_files) if result.modified_files else '-'}")
    if result.compile_latency_s is not None:
        print(f"Compile latency: {result.compile_latency_s:.2f}s")
    if result.test_latency_s is not None:
        print(f"Test latency: {result.test_latency_s:.2f}s")
    if not result.attempted_compile:
        print("Test result: NOT COMPILED (ranked out)")
        print(SEPARATOR)
        return
    if result.compile_error is not None:
        print("Test result: COMPILE FAILURE")
        print(result.compile_error)
        print(SEPARATOR)
        return

    if result.test_result is None:
        print("Test result: NOT RUN")
    else:
        print(f"Test result: {'PASS' if result.test_result.passed else 'FAIL'}")
        if not result.test_result.passed:
            stderr_lines = result.test_result.stderr.strip().splitlines()[:20]
            if stderr_lines:
                print("stderr preview:")
                for line in stderr_lines:
                    print(f"  {line}")
    print(SEPARATOR)


def main() -> int:
    args = parse_args()
    run_id = args.run_id or make_run_id()
    task_dirs = resolve_task_dirs(args.task_dir, args.tasks_root, args.num_tasks)
    if not task_dirs:
        print("No task directories found.")
        return 1

    client = LocalLLMClient(base_url=args.llm_base_url)
    all_task_results: dict[str, list[PlanResult]] = {}

    for task_dir in task_dirs:
        print(f"\n{'=' * 72}")
        print(f"Loading task: {task_dir}")
        print(f"{'=' * 72}")
        try:
            task = load_task_context(task_dir)
        except Exception as exc:
            print(f"Failed to load task: {exc}")
            continue

        print(f"Task loaded: {task.task_id} ({task.family})")
        print(f"Target files: {', '.join(task.target_files)}")
        print(f"Prompt: {task.prompt}")

        raw_proposal_prompt: str | None = None
        raw_proposal_response: str | None = None
        proposal_source = args.proposal_source
        proposal_latency_s: float | None = None
        plans: list[Any] = []
        proposal_started = time.perf_counter()
        try:
            proposal_result = propose_dsl_candidates(
                task,
                k=args.k,
                temperature=args.proposal_temp,
                client=client,
                return_debug=True,
                source=args.proposal_source,
            )
            assert isinstance(proposal_result, dict)
            plans = list(proposal_result.get("plans", []))
            raw_proposal_prompt = proposal_result.get("raw_prompt") if isinstance(proposal_result.get("raw_prompt"), str) else None
            raw_proposal_response = proposal_result.get("raw_response") if isinstance(proposal_result.get("raw_response"), str) else None
            proposal_source = proposal_result.get("proposal_source", args.proposal_source)
            proposal_latency_s = time.perf_counter() - proposal_started
            print(f"Proposal source: {proposal_source}")
            print(f"Proposal latency: {proposal_latency_s:.2f}s")
            if args.print_raw_llm and raw_proposal_response is not None:
                print()
                print("=" * 72)
                print("RAW PROPOSAL RESPONSE")
                print("=" * 72)
                print(raw_proposal_response)
                print("=" * 72)
        except Exception as exc:
            print(f"DSL proposal generation failed: {exc}")
            if args.save_artifacts:
                attempt_dir = save_attempt_artifact(
                    task=task,
                    run_id=run_id,
                    attempt_index=1,
                    plans=[],
                    results=[],
                    artifacts_dir=args.artifacts_dir,
                    save_raw_llm=args.save_raw_llm,
                    raw_proposal_prompt=raw_proposal_prompt,
                    raw_proposal_response=raw_proposal_response,
                    proposal_source=proposal_source,
                    proposal_latency_s=proposal_latency_s,
                )
                print(f"Saved attempt artifact to {attempt_dir}")
            all_task_results[task.task_id] = []
            continue

        if not plans:
            print("No valid DSL plans returned.")
            if args.save_artifacts:
                attempt_dir = save_attempt_artifact(
                    task=task,
                    run_id=run_id,
                    attempt_index=1,
                    plans=[],
                    results=[],
                    artifacts_dir=args.artifacts_dir,
                    save_raw_llm=args.save_raw_llm,
                    raw_proposal_prompt=raw_proposal_prompt,
                    raw_proposal_response=raw_proposal_response,
                    proposal_source=proposal_source,
                    proposal_latency_s=proposal_latency_s,
                )
                print(f"Saved attempt artifact to {attempt_dir}")
            all_task_results[task.task_id] = []
            continue

        print()
        print(pretty_print_plans(plans))

        task_results: list[PlanResult] = []
        for index, plan in enumerate(plans, start=1):
            attempted_compile = index <= args.compile_top_n
            compiled_files: dict[str, str] | None = None
            compile_error: str | None = None
            test_result: TestRunResult | None = None
            raw_compiler_prompt: str | None = None
            raw_compiler_response: str | None = None
            compile_latency_s: float | None = None
            test_latency_s: float | None = None

            if attempted_compile:
                compile_started = time.perf_counter()
                try:
                    compiler_result = compile_plan_to_code(
                        task,
                        plan,
                        temperature=args.compile_temp,
                        client=client,
                        return_debug=True,
                        allow_full_file_fallback=args.allow_full_file_fallback,
                    )
                    compile_latency_s = time.perf_counter() - compile_started
                    assert isinstance(compiler_result, dict)
                    compiled_files = dict(compiler_result.get("compiled_files", {}))
                    raw_compiler_prompt = compiler_result.get("raw_prompt") if isinstance(compiler_result.get("raw_prompt"), str) else None
                    raw_compiler_response = compiler_result.get("raw_response") if isinstance(compiler_result.get("raw_response"), str) else None
                    if args.print_raw_llm and raw_compiler_response is not None:
                        print()
                        print("=" * 72)
                        print(f"RAW COMPILER RESPONSE FOR PLAN {index}")
                        print("=" * 72)
                        print(raw_compiler_response)
                        print("=" * 72)
                    print()
                    print(SEPARATOR)
                    print(f"Compiled output for Plan {index}")
                    print(SEPARATOR)
                    print(f"Modified files: {', '.join(compiled_files)}")
                    print(preview_compiled_files(compiled_files))
                except Exception as exc:
                    compile_latency_s = time.perf_counter() - compile_started
                    compile_error = str(exc)
                    if isinstance(exc, CompilePlanError):
                        raw_compiler_prompt = exc.raw_prompt
                        raw_compiler_response = exc.raw_response
                        if args.print_raw_llm and raw_compiler_response is not None:
                            print()
                            print("=" * 72)
                            print(f"RAW COMPILER RESPONSE FOR PLAN {index} (FAILED)")
                            print("=" * 72)
                            print(raw_compiler_response)
                            print("=" * 72)

            if attempted_compile and compiled_files is not None and args.run_tests:
                workspace_dir: Path | None = None
                test_started = time.perf_counter()
                try:
                    workspace_dir = create_temp_workspace(task.task_dir)
                    apply_compiled_files(workspace_dir, compiled_files)
                    test_result = run_visible_tests(
                        workspace_dir,
                        python_bin=args.python_bin,
                        timeout=args.timeout,
                    )
                    test_latency_s = time.perf_counter() - test_started
                except Exception as exc:
                    test_latency_s = time.perf_counter() - test_started
                    test_result = TestRunResult(
                        passed=False,
                        stdout="",
                        stderr=f"Visible test execution crashed: {exc}",
                        return_code=1,
                        command=[],
                    )
                finally:
                    if workspace_dir is not None:
                        shutil.rmtree(workspace_dir.parent, ignore_errors=True)

            result = PlanResult(
                index=index,
                plan_id=getattr(plan, "plan_id", f"plan_{index - 1}"),
                strategy=getattr(plan, "strategy", "unknown"),
                modified_files=sorted(compiled_files) if compiled_files else [],
                compile_error=compile_error,
                test_result=test_result,
                compiled_files={} if compiled_files is None else compiled_files,
                raw_compiler_prompt=raw_compiler_prompt,
                raw_compiler_response=raw_compiler_response,
                attempted_compile=attempted_compile,
                compile_latency_s=compile_latency_s,
                test_latency_s=test_latency_s,
            )
            task_results.append(result)
            print()
            print_plan_result(result)

        print()
        print(summarize_task_results(task, task_results))
        if args.save_artifacts:
            attempt_dir = save_attempt_artifact(
                task=task,
                run_id=run_id,
                attempt_index=1,
                plans=plans,
                results=task_results,
                artifacts_dir=args.artifacts_dir,
                save_raw_llm=args.save_raw_llm,
                raw_proposal_prompt=raw_proposal_prompt,
                raw_proposal_response=raw_proposal_response,
                proposal_source=proposal_source,
                proposal_latency_s=proposal_latency_s,
            )
            print(f"Saved attempt artifact to {attempt_dir}")
        all_task_results[task.task_id] = task_results

    total_plans = sum(len(results) for results in all_task_results.values())
    total_passing = sum(
        1
        for results in all_task_results.values()
        for result in results
        if result.test_result is not None and result.test_result.passed
    )
    total_compile_failures = sum(
        1
        for results in all_task_results.values()
        for result in results
        if result.compile_error is not None
    )
    print(f"\n{'=' * 72}")
    print("Aggregate summary")
    print(f"{'=' * 72}")
    print(f"Tasks processed: {len(all_task_results)}")
    print(f"Total plans: {total_plans}")
    print(f"Passing plans: {total_passing}")
    print(f"Compile failures: {total_compile_failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

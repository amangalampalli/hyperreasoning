"""Cached verification helpers for search-control collection and runtime."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from env.dsl_schema import PlanDSL
from llm.compiler import CompilePlanError, apply_compiled_files, compile_plan_to_code
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import TaskContext
from llm.proposal import plan_signature

LOGGER = logging.getLogger("env.verifier")


class VerificationResult(BaseModel):
    """Observed outcome for one plan verification attempt."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    plan_signature: str
    cached: bool = False
    attempted_compile: bool = False
    compile_success: bool | None = None
    compile_error: str | None = None
    compiled_files: dict[str, str] = Field(default_factory=dict)
    files_changed: list[str] = Field(default_factory=list)
    compile_latency_s: float | None = None
    visible_test_passed: bool | None = None
    visible_test_returncode: int | None = None
    visible_test_stdout: str | None = None
    visible_test_stderr: str | None = None
    test_latency_s: float | None = None
    hidden_test_passed: bool | None = None
    hidden_test_returncode: int | None = None
    hidden_test_stdout: str | None = None
    hidden_test_stderr: str | None = None
    hidden_test_latency_s: float | None = None
    label_tier: str = "compile"


def _syntax_check(compiled_files: dict[str, str]) -> str | None:
    for relative_path, content in compiled_files.items():
        if not relative_path.endswith(".py"):
            continue
        try:
            compile(content, relative_path, "exec")
        except SyntaxError as exc:
            return f"{relative_path}:{exc.lineno}: {exc.msg}"
    return None


def _run_test_file(
    workspace_dir: Path,
    *,
    test_file: str,
    python_bin: str,
    timeout_s: float,
) -> tuple[bool, int, str, str]:
    target = workspace_dir / test_file
    if not target.exists():
        return False, 1, "", f"Missing {test_file}"

    module_target = test_file[:-3].replace("/", ".").replace("\\", ".") if test_file.endswith(".py") else test_file
    commands = [
        [python_bin, "-m", "unittest", module_target],
        [python_bin, "-m", module_target],
    ]
    last_stdout = ""
    last_stderr = ""
    last_code = 1
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                cwd=workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            last_stdout = exc.stdout or ""
            last_stderr = (exc.stderr or "") + f"\nTimed out after {timeout_s} seconds"
            last_code = 124
            continue
        if completed.returncode == 0:
            return True, 0, completed.stdout, completed.stderr
        last_stdout = completed.stdout
        last_stderr = completed.stderr
        last_code = completed.returncode
    return False, last_code, last_stdout, last_stderr


class CachedTaskVerifier:
    """Verify plans once per task and reuse the result across episodes."""

    def __init__(
        self,
        *,
        client: LocalLLMClient | None = None,
        run_tests: bool = True,
        run_hidden_tests: bool = False,
        python_bin: str = "python",
        timeout_s: float = 12.0,
        compiler_temp: float = 0.2,
        allow_full_file_fallback: bool = False,
        max_verified_plans: int = 8,
        compile_fn: Callable[..., dict[str, str] | dict[str, object]] | None = None,
        test_runner: Callable[[Path, str, float], tuple[bool, int, str, str]] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.client = client or LocalLLMClient()
        self.run_tests = run_tests
        self.run_hidden_tests = run_hidden_tests
        self.python_bin = python_bin
        self.timeout_s = timeout_s
        self.compiler_temp = compiler_temp
        self.allow_full_file_fallback = allow_full_file_fallback
        self.max_verified_plans = max_verified_plans
        self._compile_fn = compile_fn or compile_plan_to_code
        self._test_runner = test_runner or (lambda workspace_dir, python_bin, timeout_s: _run_test_file(
            workspace_dir,
            test_file="test_visible.py",
            python_bin=python_bin,
            timeout_s=timeout_s,
        ))
        self._cache: dict[str, VerificationResult] = {}
        self.progress_callback = progress_callback

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    def can_verify(self, signature: str) -> bool:
        return signature in self._cache or len(self._cache) < self.max_verified_plans

    def cached_result(self, signature: str) -> VerificationResult | None:
        result = self._cache.get(signature)
        if result is None:
            return None
        return result.model_copy(update={"cached": True})

    def verify(self, task: TaskContext, plan: PlanDSL) -> VerificationResult:
        signature = plan_signature(plan)
        cached = self.cached_result(signature)
        if cached is not None:
            self._emit_progress(
                {
                    "event": "verification_cached",
                    "task_id": task.task_id,
                    "plan_id": plan.plan_id,
                    "plan_signature": signature,
                }
            )
            return cached
        if not self.can_verify(signature):
            raise RuntimeError(f"Verification budget exhausted for task {task.task_id}")

        self._emit_progress(
            {
                "event": "verification_started",
                "task_id": task.task_id,
                "plan_id": plan.plan_id,
                "plan_signature": signature,
            }
        )
        compile_started = time.perf_counter()
        try:
            compiler_result = self._compile_fn(
                task,
                plan,
                temperature=self.compiler_temp,
                client=self.client,
                return_debug=False,
                allow_full_file_fallback=self.allow_full_file_fallback,
            )
            if isinstance(compiler_result, dict) and "compiled_files" in compiler_result:
                compiled_files = dict(compiler_result["compiled_files"])
            else:
                compiled_files = dict(compiler_result)
            compile_success = True
            compile_error = _syntax_check(compiled_files)
            if compile_error is not None:
                compile_success = False
                compiled_files = {}
        except (CompilePlanError, RuntimeError) as exc:
            compiled_files = {}
            compile_success = False
            compile_error = str(exc)

        result = VerificationResult(
            plan_id=plan.plan_id,
            plan_signature=signature,
            attempted_compile=True,
            compile_success=compile_success,
            compile_error=compile_error,
            compiled_files=compiled_files,
            files_changed=sorted(compiled_files),
            compile_latency_s=time.perf_counter() - compile_started,
            label_tier="compile",
        )
        self._emit_progress(
            {
                "event": "compile_completed",
                "task_id": task.task_id,
                "plan_id": plan.plan_id,
                "plan_signature": signature,
                "compile_success": compile_success,
                "compile_error": compile_error,
                "elapsed_s": result.compile_latency_s,
            }
        )

        if compile_success and compiled_files and self.run_tests:
            test_started = time.perf_counter()
            temp_dir = Path(tempfile.mkdtemp(prefix=f"verify_{task.task_id}_"))
            try:
                workspace_dir = apply_compiled_files(task, compiled_files, workspace_dir=temp_dir)
                passed, returncode, stdout, stderr = self._test_runner(
                    workspace_dir,
                    self.python_bin,
                    self.timeout_s,
                )
                result.visible_test_passed = passed
                result.visible_test_returncode = returncode
                result.visible_test_stdout = stdout
                result.visible_test_stderr = stderr
                result.test_latency_s = time.perf_counter() - test_started
                result.label_tier = "visible_test"
                self._emit_progress(
                    {
                        "event": "tests_completed",
                        "task_id": task.task_id,
                        "plan_id": plan.plan_id,
                        "plan_signature": signature,
                        "visible_test_passed": passed,
                        "returncode": returncode,
                        "elapsed_s": result.test_latency_s,
                    }
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if compile_success and compiled_files and self.run_hidden_tests and task.hidden_test_file:
            hidden_started = time.perf_counter()
            temp_dir = Path(tempfile.mkdtemp(prefix=f"verify_hidden_{task.task_id}_"))
            try:
                workspace_dir = apply_compiled_files(task, compiled_files, workspace_dir=temp_dir)
                LOGGER.info(
                    "Running hidden tests for task=%s plan=%s file=%s",
                    task.task_id,
                    plan.plan_id,
                    task.hidden_test_file,
                )
                passed, returncode, stdout, stderr = _run_test_file(
                    workspace_dir,
                    test_file=task.hidden_test_file,
                    python_bin=self.python_bin,
                    timeout_s=self.timeout_s,
                )
                result.hidden_test_passed = passed
                result.hidden_test_returncode = returncode
                result.hidden_test_stdout = stdout
                result.hidden_test_stderr = stderr
                result.hidden_test_latency_s = time.perf_counter() - hidden_started
                if not passed:
                    summary = (stderr or stdout or "").strip()
                    if not summary:
                        summary = f"returncode={returncode}"
                    LOGGER.warning(
                        "Hidden tests failed for task=%s plan=%s returncode=%s summary=%s",
                        task.task_id,
                        plan.plan_id,
                        returncode,
                        summary.replace("\n", " | "),
                    )
                else:
                    LOGGER.info(
                        "Hidden tests passed for task=%s plan=%s file=%s",
                        task.task_id,
                        plan.plan_id,
                        task.hidden_test_file,
                    )
                self._emit_progress(
                    {
                        "event": "hidden_tests_completed",
                        "task_id": task.task_id,
                        "plan_id": plan.plan_id,
                        "plan_signature": signature,
                        "hidden_test_passed": passed,
                        "returncode": returncode,
                        "elapsed_s": result.hidden_test_latency_s,
                    }
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        self._cache[signature] = result
        return result.model_copy()

    def _emit_progress(self, event: dict[str, Any]) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event)

    def summary(self) -> dict[str, Any]:
        results = list(self._cache.values())
        return {
            "verified_plans": len(results),
            "compile_successes": sum(1 for item in results if item.compile_success),
            "visible_test_passes": sum(1 for item in results if item.visible_test_passed is True),
            "hidden_test_passes": sum(1 for item in results if item.hidden_test_passed is True),
            "avg_compile_latency_s": (
                sum(item.compile_latency_s or 0.0 for item in results) / len(results) if results else 0.0
            ),
            "avg_test_latency_s": (
                sum(item.test_latency_s or 0.0 for item in results if item.test_latency_s is not None)
                / max(1, sum(1 for item in results if item.test_latency_s is not None))
            ),
            "avg_hidden_test_latency_s": (
                sum(item.hidden_test_latency_s or 0.0 for item in results if item.hidden_test_latency_s is not None)
                / max(1, sum(1 for item in results if item.hidden_test_latency_s is not None))
            ),
        }

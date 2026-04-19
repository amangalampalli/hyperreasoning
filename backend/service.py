"""Backend orchestration helpers for plugin-facing task runs."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from itertools import count
from pathlib import Path
import threading
from typing import Any, Callable

import httpx
import orjson
from tqdm.auto import tqdm

from backend.config import BackendConfig, load_backend_config
from backend.models import (
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    HealthResponse,
    RunDiagnostic,
    StrategySummary,
    TaskRunRequest,
    TaskRunResponse,
)
from backend.run_cache import cache_for_context, compute_comparison_cache_key, compute_run_cache_key
from backend.search_graph import TaskSearchGraphCollector
from env.dsl_env import (
    SearchControlConfig,
    TaskSearchResult,
    run_single_task_search,
)
from env.state_encoder import StateEncoder
from env.verifier import CachedTaskVerifier
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import load_task_context
from rl.rainbow import RainbowAgent


_REQUEST_COUNTER = count(1)
_ACTIVE_PROGRESS_POSITIONS: set[int] = set()
_PROGRESS_POSITION_LOCK = threading.Lock()
LOGGER = logging.getLogger("backend.service")


def llm_health(base_url: str) -> HealthResponse:
    try:
        with httpx.Client(timeout=3.0) as client:
            response = client.get(base_url.rstrip("/"))
            response.raise_for_status()
        reachable = True
    except Exception:
        reachable = False
    return HealthResponse(status="ok", llm_base_url=base_url, llm_reachable=reachable)


def _materialize_task(request: TaskRunRequest) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="hyperreasoning_backend_task_"))
    for relative_path, content in request.files.items():
        file_path = temp_root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    target_files = list(request.target_files) or [
        path for path in request.files if not path.startswith("test_")
    ]
    payload = {
        "task_id": "plugin_task",
        "family": request.family,
        "difficulty": "custom",
        "language": request.language,
        "prompt": request.prompt,
        "target_files": target_files,
        "visible_test_file": request.visible_test_file,
        "hidden_test_file": request.hidden_test_file,
        "metadata": {"target_files": target_files},
    }
    (temp_root / "task.json").write_text(orjson.dumps(payload).decode("utf-8"), encoding="utf-8")
    return temp_root


def _build_summary(result: TaskSearchResult, *, elapsed_s: float, client: LocalLLMClient) -> StrategySummary:
    best_verification = result.best_verification or {}
    visible_counts = _test_counts_from_verification(
        passed=best_verification.get("visible_test_passed"),
        stdout=best_verification.get("visible_test_stdout"),
        stderr=best_verification.get("visible_test_stderr"),
    )
    hidden_counts = _test_counts_from_verification(
        passed=best_verification.get("hidden_test_passed"),
        stdout=best_verification.get("hidden_test_stdout"),
        stderr=best_verification.get("hidden_test_stderr"),
    )
    tests_passed = visible_counts[0] + hidden_counts[0]
    tests_total = visible_counts[1] + hidden_counts[1]
    return StrategySummary(
        policy=result.policy,
        task_id=result.task_id,
        family=result.family,
        total_reward=result.total_reward,
        steps=result.steps,
        compile_successes=result.compile_successes,
        visible_passes=result.visible_passes,
        hidden_passes=int(result.verifier_summary.get("hidden_test_passes", 0)),
        visible_tests_passed=visible_counts[0],
        visible_tests_total=visible_counts[1],
        hidden_tests_passed=hidden_counts[0],
        hidden_tests_total=hidden_counts[1],
        tests_passed=tests_passed,
        tests_total=tests_total,
        fraction_tests_passed=(tests_passed / tests_total) if tests_total else 0.0,
        best_bank_id=result.best_bank_id,
        compile_success=best_verification.get("compile_success"),
        visible_test_passed=best_verification.get("visible_test_passed"),
        hidden_test_passed=best_verification.get("hidden_test_passed"),
        compile_error=best_verification.get("compile_error"),
        hidden_test_stdout=best_verification.get("hidden_test_stdout"),
        hidden_test_stderr=best_verification.get("hidden_test_stderr"),
        hidden_test_returncode=best_verification.get("hidden_test_returncode"),
        elapsed_s=elapsed_s,
        llm_requests=client.total_requests,
        prompt_tokens=client.total_prompt_tokens,
        completion_tokens=client.total_completion_tokens,
        total_tokens=client.total_tokens,
    )


def _test_counts_from_verification(*, passed: Any, stdout: Any, stderr: Any) -> tuple[int, int]:
    output = "\n".join(str(item) for item in (stdout, stderr) if isinstance(item, str) and item.strip())
    total = _unittest_total(output)
    if total is None:
        return (0, 0)
    if passed is True:
        return (total, total)
    return (_unittest_ok_count(output), total)


def _unittest_total(output: str) -> int | None:
    matches = re.findall(r"Ran\s+(\d+)\s+tests?", output)
    if not matches:
        return None
    return int(matches[-1])


def _unittest_ok_count(output: str) -> int:
    verbose_ok = len(re.findall(r"(?m)^.+\s\.\.\.\sok\s*$", output))
    if verbose_ok:
        return verbose_ok
    dots_line = next((line.strip() for line in output.splitlines() if set(line.strip()) <= {".", "F", "E", "s", "x"} and line.strip()), "")
    return dots_line.count(".")


def _build_diagnostics(result: TaskSearchResult) -> list[RunDiagnostic]:
    diagnostics_by_bank: dict[str, RunDiagnostic] = {}
    plan_entries = result.plan_bank.get("entries") if isinstance(result.plan_bank.get("entries"), dict) else {}
    for transition in result.episode.transitions:
        if transition.action != "COMPILE_TO_CODE":
            continue
        state = transition.state if isinstance(transition.state, dict) else {}
        info = transition.info if isinstance(transition.info, dict) else {}
        bank_id = _string_or_none(state.get("current_bank_id"))
        if bank_id is None:
            continue
        diagnostics_by_bank[bank_id] = _diagnostic_from_payload(
            policy=result.policy,
            bank_id=bank_id,
            info=info,
            plan_entry=plan_entries.get(bank_id),
            is_best=bank_id == result.best_bank_id,
        )

    if result.best_bank_id and result.best_verification and result.best_bank_id not in diagnostics_by_bank:
        diagnostics_by_bank[result.best_bank_id] = _diagnostic_from_payload(
            policy=result.policy,
            bank_id=result.best_bank_id,
            info=result.best_verification,
            plan_entry=plan_entries.get(result.best_bank_id),
            is_best=True,
        )

    return list(diagnostics_by_bank.values())


def _diagnostic_from_payload(
    *,
    policy: str,
    bank_id: str,
    info: dict[str, Any],
    plan_entry: Any,
    is_best: bool,
) -> RunDiagnostic:
    plan = plan_entry.get("plan") if isinstance(plan_entry, dict) and isinstance(plan_entry.get("plan"), dict) else {}
    compile_success = _bool_or_none(info.get("compile_success"))
    visible_passed = _bool_or_none(info.get("visible_test_passed"))
    hidden_passed = _bool_or_none(info.get("hidden_test_passed"))
    return RunDiagnostic(
        policy=policy,
        bank_id=bank_id,
        plan_id=_string_or_none(plan.get("plan_id") or info.get("plan_id")),
        plan_signature=_string_or_none(
            (plan_entry.get("plan_signature") if isinstance(plan_entry, dict) else None)
            or info.get("plan_signature")
        ),
        strategy=_string_or_none(plan.get("strategy")),
        target_files=[str(item) for item in plan.get("target_files") or []],
        status=_diagnostic_status(
            compile_success=compile_success,
            visible_test_passed=visible_passed,
            hidden_test_passed=hidden_passed,
        ),
        is_best=is_best,
        compile_success=compile_success,
        compile_error=_string_or_none(info.get("compile_error")),
        visible_test_passed=visible_passed,
        visible_test_returncode=_int_or_none(info.get("visible_test_returncode")),
        visible_test_stdout=_string_or_none(info.get("visible_test_stdout")),
        visible_test_stderr=_string_or_none(info.get("visible_test_stderr")),
        hidden_test_passed=hidden_passed,
        hidden_test_returncode=_int_or_none(info.get("hidden_test_returncode")),
        hidden_test_stdout=_string_or_none(info.get("hidden_test_stdout")),
        hidden_test_stderr=_string_or_none(info.get("hidden_test_stderr")),
    )


def _diagnostic_status(
    *,
    compile_success: bool | None,
    visible_test_passed: bool | None,
    hidden_test_passed: bool | None,
) -> str:
    if compile_success is False:
        return "compile_failed"
    if hidden_test_passed is False:
        return "hidden_failed"
    if visible_test_passed is False:
        return "visible_failed"
    if compile_success is True or visible_test_passed is True or hidden_test_passed is True:
        return "success"
    return "unknown"


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _load_rainbow_artifacts(checkpoint_path: str | None) -> tuple[RainbowAgent | None, StateEncoder | None]:
    if checkpoint_path is None:
        return None, None
    agent, encoder_state = RainbowAgent.load(Path(checkpoint_path))
    if encoder_state is None:
        raise RuntimeError("Checkpoint does not include encoder state")
    return agent, StateEncoder.from_dict(encoder_state)


def _build_search_config(request: TaskRunRequest) -> SearchControlConfig:
    config_kwargs: dict[str, Any] = {
        "max_steps_per_episode": request.max_steps,
        "proposal_source": request.proposal_source,
        "run_tests": request.run_tests,
        "allow_full_file_fallback": request.allow_full_file_fallback,
        "max_verified_plans_per_task": request.max_verified_plans_per_task,
        "seed": request.seed,
    }
    if request.policy == "oneshot":
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
                "max_verified_plans_per_task": 1,
            }
        )
    return SearchControlConfig(**config_kwargs)


def _request_progress_label(request: TaskRunRequest) -> str:
    request_index = next(_REQUEST_COUNTER)
    return f"api:{request_index:04d}:{request.policy}:{request.family}"


def _acquire_progress_position() -> int:
    with _PROGRESS_POSITION_LOCK:
        position = 0
        while position in _ACTIVE_PROGRESS_POSITIONS:
            position += 1
        _ACTIVE_PROGRESS_POSITIONS.add(position)
        return position


def _release_progress_position(position: int) -> None:
    with _PROGRESS_POSITION_LOCK:
        _ACTIVE_PROGRESS_POSITIONS.discard(position)


def _format_duration(value: Any) -> str:
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "-"


def _create_request_progress_bar(request: TaskRunRequest, *, position: int) -> tqdm:
    max_steps = _build_search_config(request).max_steps_per_episode
    return tqdm(
        total=max_steps,
        desc=_request_progress_label(request),
        unit="step",
        leave=True,
        position=position,
        dynamic_ncols=True,
    )


class _RequestProgressReporter:
    def __init__(self, request: TaskRunRequest) -> None:
        self.request = request
        self.max_steps = _build_search_config(request).max_steps_per_episode
        self.position = _acquire_progress_position()
        self.progress_bar = _create_request_progress_bar(request, position=self.position)
        self.started_at = time.perf_counter()
        self.llm_requests = 0
        self.last_llm_label = "-"
        self.last_llm_duration = "-"
        self.last_compile_duration = "-"
        self.last_test_duration = "-"
        self.phase = "queued"

    def on_progress(self, event: dict[str, Any]) -> None:
        event_name = event.get("event")
        if event_name == "search_started":
            self.phase = "planning"
            self._set_postfix(max_steps=event.get("max_steps", self.max_steps))
            return
        if event_name == "plan_bank_built":
            self.phase = "search"
            self._set_postfix(
                roots=event.get("root_candidates", 0),
                plans=event.get("total_plans", 0),
            )
            return
        if event_name == "llm_request_started":
            self.phase = str(event.get("request_label") or "llm")
            self._set_postfix(mode=event.get("mode", "-"), attempt=event.get("attempt", 1))
            return
        if event_name == "llm_request_completed":
            self.phase = "llm_done"
            self.llm_requests += 1
            self.last_llm_label = str(event.get("request_label") or event.get("mode") or "llm")
            self.last_llm_duration = _format_duration(event.get("elapsed_s"))
            self._set_postfix(mode=event.get("mode", "-"), llm=self.last_llm_label)
            return
        if event_name == "llm_request_failed":
            self.phase = "llm_error"
            self.last_llm_label = str(event.get("request_label") or event.get("mode") or "llm")
            self.last_llm_duration = _format_duration(event.get("elapsed_s"))
            self._set_postfix(mode=event.get("mode", "-"), llm_error=self.last_llm_label)
            return
        if event_name == "verification_started":
            self.phase = "verify"
            self._set_postfix(plan=event.get("plan_id", "-"))
            return
        if event_name == "verification_cached":
            self.phase = "verify_cached"
            self._set_postfix(plan=event.get("plan_id", "-"))
            return
        if event_name == "compile_completed":
            self.phase = "compile"
            self.last_compile_duration = _format_duration(event.get("elapsed_s"))
            self._set_postfix(compiled=event.get("compile_success"))
            return
        if event_name == "tests_completed":
            self.phase = "tests"
            self.last_test_duration = _format_duration(event.get("elapsed_s"))
            self._set_postfix(tests=event.get("visible_test_passed"))
            return
        if event_name == "step_completed":
            completed = min(int(event.get("step", 0)), self.max_steps)
            self.progress_bar.update(max(0, completed - self.progress_bar.n))
            self.phase = "search"
            self._set_postfix(
                action=event.get("action", "-"),
                compile=event.get("compile_success"),
                visible=event.get("visible_test_passed"),
                tier=event.get("label_tier", "-"),
            )
            return
        if event_name == "search_completed":
            self.progress_bar.update(max(0, self.max_steps - self.progress_bar.n))
            self.phase = "done"
            self._set_postfix(
                compile=event.get("compile_successes", 0),
                visible=event.get("visible_passes", 0),
                elapsed=_format_duration(time.perf_counter() - self.started_at),
            )

    def _set_postfix(self, **extra: Any) -> None:
        self.progress_bar.set_postfix(
            phase=self.phase,
            llm_n=self.llm_requests,
            llm_t=self.last_llm_duration,
            compile_t=self.last_compile_duration,
            test_t=self.last_test_duration,
            **extra,
            refresh=False,
        )

    def close(self, *, status: str) -> None:
        LOGGER.info(
            "%s %s in %.1fs after %d LLM calls",
            status.capitalize(),
            self.progress_bar.desc,
            time.perf_counter() - self.started_at,
            self.llm_requests,
        )
        self.progress_bar.close()
        _release_progress_position(self.position)


def run_task_request(
    request: TaskRunRequest,
    *,
    llm_base_url: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    backend_config: BackendConfig | None = None,
    store_result: bool = True,
) -> TaskRunResponse:
    app_config = backend_config or load_backend_config()
    cache_key = compute_run_cache_key(request)
    run_cache = cache_for_context(request.client_context)
    if run_cache is not None:
        cached = run_cache.find_task_by_cache_key(cache_key)
        if cached is not None:
            if cached.result is None:
                raise RuntimeError("Cached task run package did not include a task result")
            response = cached.result
            response.cache_hit = True
            response.cache_key = cache_key
            response.run_id = cached.item.run_id
            response.cloud_status = cached.item.cloud_status
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "cache_hit",
                        "run_id": cached.item.run_id,
                        "cache_key": cache_key,
                        "policy": request.policy,
                        "max_steps": request.max_steps,
                    }
                )
            return response

    temp_dir = _materialize_task(request)
    reporter = _RequestProgressReporter(request)
    search_graph = TaskSearchGraphCollector(policy=request.policy)
    status = "failed"

    LOGGER.info(
        "Starting task run policy=%s family=%s run_tests=%s run_hidden_tests=%s hidden_test_file=%s",
        request.policy,
        request.family,
        request.run_tests,
        request.run_hidden_tests,
        request.hidden_test_file,
    )

    def emit_progress(event: dict[str, Any]) -> None:
        reporter.on_progress(event)
        search_graph.on_progress(event)
        if progress_callback is not None:
            progress_callback(event)

    try:
        task = load_task_context(temp_dir)
        search_config = _build_search_config(request)
        client = LocalLLMClient(base_url=llm_base_url, progress_callback=emit_progress)
        verifier = CachedTaskVerifier(
            client=client,
            run_tests=request.run_tests,
            run_hidden_tests=request.run_hidden_tests,
            python_bin=search_config.python_bin,
            timeout_s=search_config.timeout_s,
            compiler_temp=search_config.compiler_temp,
            allow_full_file_fallback=search_config.allow_full_file_fallback,
            max_verified_plans=search_config.max_verified_plans_per_task,
            progress_callback=emit_progress,
        )
        if request.policy == "rainbow":
            agent, encoder = _load_rainbow_artifacts(request.checkpoint_path)
        else:
            agent, encoder = None, None
        result = run_single_task_search(
            task,
            search_config,
            policy=request.policy,
            client=client,
            verifier=verifier,
            agent=agent,
            encoder=encoder,
            seed=request.seed,
            progress_callback=emit_progress,
        )
        status = "completed"
        elapsed_s = time.perf_counter() - reporter.started_at
        response = TaskRunResponse(
            strategy=_build_summary(result, elapsed_s=elapsed_s, client=client),
            root_candidates=result.root_candidates,
            plan_bank=result.plan_bank,
            nodes=result.episode.nodes,
            edges=result.episode.edges,
            transitions=[transition.model_dump() for transition in result.episode.transitions],
            best_plan=result.best_plan,
            best_compiled_files=result.best_compiled_files,
            verifier_summary=result.verifier_summary,
            diagnostics=_build_diagnostics(result),
        )
        response.search_graph_events = search_graph.finalize(response, policy=request.policy)
        response.cache_key = cache_key
        if run_cache is not None and store_result:
            item = run_cache.store_task_run(request=request, response=response, backend_config=app_config)
            response.run_id = item.run_id
            response.cache_key = item.cache_key
            response.cloud_status = item.cloud_status
        return response
    finally:
        reporter.close(status=status)
        shutil.rmtree(temp_dir, ignore_errors=True)


def compare_strategies(
    request: CompareStrategiesRequest,
    *,
    llm_base_url: str,
    backend_config: BackendConfig | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> CompareStrategiesResponse:
    config = backend_config or load_backend_config()
    cache_key = compute_comparison_cache_key(request)
    run_cache = cache_for_context(request.client_context)
    if run_cache is not None:
        cached = run_cache.find_by_cache_key(cache_key, policy="comparison")
        if cached is not None:
            if cached.compare_result is None:
                raise RuntimeError("Cached comparison package did not include a comparison result")
            response = cached.compare_result
            response.cache_hit = True
            response.cache_key = cache_key
            response.run_id = cached.item.run_id
            response.cloud_status = cached.item.cloud_status
            if progress_callback is not None:
                progress_callback(
                    {
                        "event": "comparison_cache_hit",
                        "run_id": cached.item.run_id,
                        "cache_key": cache_key,
                        "total_policies": len(request.policies),
                        "max_steps": request.max_steps,
                    }
                )
            return response

    responses: list[TaskRunResponse] = []
    for index, policy in enumerate(request.policies, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "comparison_policy_started",
                    "policy": policy,
                    "current_policy_index": index,
                    "total_policies": len(request.policies),
                    "max_steps": request.max_steps,
                }
            )
        single = TaskRunRequest(**{**request.model_dump(exclude={"policies"}), "policy": policy})
        responses.append(
            run_task_request(
                single,
                llm_base_url=llm_base_url,
                progress_callback=progress_callback,
                backend_config=config,
                store_result=False,
            )
        )
    response = CompareStrategiesResponse(strategies=responses, cache_key=cache_key)
    if run_cache is not None:
        item = run_cache.store_comparison_run(request=request, response=response, backend_config=config)
        response.run_id = item.run_id
        response.cache_key = item.cache_key
        response.cloud_status = item.cloud_status
    return response

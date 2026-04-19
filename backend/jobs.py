"""In-memory async job runner for plugin-visible task progress."""

from __future__ import annotations

import threading
import time
from typing import Any
from uuid import uuid4

from backend.config import BackendConfig, load_backend_config
from backend.models import (
    AsyncJobAcceptedResponse,
    CompareJobStatusResponse,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    JobProgressSnapshot,
    TaskRunJobStatusResponse,
    TaskRunRequest,
    TaskRunResponse,
)
from backend.service import compare_strategies, run_task_request


class _BaseJob:
    def __init__(self, *, job_id: str, kind: str) -> None:
        self.job_id = job_id
        self.kind = kind
        self.status = "queued"
        self.created_at = time.perf_counter()
        self.progress = JobProgressSnapshot()
        self.error: str | None = None
        self._lock = threading.Lock()

    def update_progress(self, event: dict[str, Any]) -> None:
        with self._lock:
            self.progress.elapsed_s = time.perf_counter() - self.created_at
            event_name = str(event.get("event") or "")
            policy = event.get("policy")
            if isinstance(policy, str):
                self.progress.policy = policy
            if event_name == "cache_hit":
                self.progress.phase = "cache_hit"
                self.progress.max_steps = int(event.get("max_steps") or self.progress.max_steps or 0)
                self.progress.current_step = self.progress.max_steps
            elif event_name == "comparison_cache_hit":
                self.progress.phase = "cache_hit"
                self.progress.total_policies = int(event.get("total_policies") or self.progress.total_policies or 1)
                self.progress.max_steps = int(event.get("max_steps") or self.progress.max_steps or 0)
                self.progress.current_policy_index = self.progress.total_policies
                self.progress.current_step = self.progress.max_steps
            elif event_name == "comparison_policy_started":
                self.progress.phase = "queued"
                self.progress.policy = str(event.get("policy") or self.progress.policy or "")
                self.progress.current_policy_index = int(event.get("current_policy_index") or self.progress.current_policy_index or 0)
                self.progress.total_policies = int(event.get("total_policies") or self.progress.total_policies or 1)
                self.progress.current_step = 0
                self.progress.max_steps = int(event.get("max_steps") or self.progress.max_steps or 0)
            elif event_name == "search_started":
                self.progress.phase = "planning"
                self.progress.max_steps = int(event.get("max_steps") or self.progress.max_steps or 0)
            elif event_name == "plan_bank_built":
                self.progress.phase = "search"
                self.progress.roots = int(event.get("root_candidates") or 0)
                self.progress.plans = int(event.get("total_plans") or 0)
            elif event_name == "llm_request_started":
                self.progress.phase = str(event.get("request_label") or "llm")
            elif event_name == "llm_request_completed":
                self.progress.phase = "llm_done"
                self.progress.llm_requests += 1
                self.progress.last_llm_label = str(event.get("request_label") or event.get("mode") or "llm")
                elapsed = event.get("elapsed_s")
                self.progress.last_llm_duration_s = float(elapsed) if elapsed is not None else None
            elif event_name == "llm_request_failed":
                self.progress.phase = "llm_error"
                self.progress.last_llm_label = str(event.get("request_label") or event.get("mode") or "llm")
                elapsed = event.get("elapsed_s")
                self.progress.last_llm_duration_s = float(elapsed) if elapsed is not None else None
                self.progress.error = str(event.get("error") or "LLM request failed")
            elif event_name == "verification_started":
                self.progress.phase = "verify"
            elif event_name == "verification_cached":
                self.progress.phase = "verify_cached"
            elif event_name == "compile_completed":
                self.progress.phase = "compile"
                elapsed = event.get("elapsed_s")
                self.progress.last_compile_duration_s = float(elapsed) if elapsed is not None else None
            elif event_name == "tests_completed":
                self.progress.phase = "tests"
                elapsed = event.get("elapsed_s")
                self.progress.last_test_duration_s = float(elapsed) if elapsed is not None else None
            elif event_name == "step_completed":
                self.progress.phase = "search"
                self.progress.current_step = int(event.get("step") or self.progress.current_step or 0)
                self.progress.max_steps = int(event.get("max_steps") or self.progress.max_steps or 0)
                self.progress.action = event.get("action")
                self.progress.label_tier = event.get("label_tier")
            elif event_name == "search_completed":
                self.progress.phase = "done"
                self.progress.compile_successes = int(event.get("compile_successes") or 0)
                self.progress.visible_passes = int(event.get("visible_passes") or 0)
                if self.progress.max_steps > 0:
                    self.progress.current_step = self.progress.max_steps

    def _progress_snapshot(self) -> JobProgressSnapshot:
        progress = self.progress.model_copy()
        if self.error is not None:
            progress.error = self.error
        return progress


class _TaskRunJob(_BaseJob):
    def __init__(self, request: TaskRunRequest, *, llm_base_url: str, backend_config: BackendConfig) -> None:
        super().__init__(job_id=uuid4().hex, kind="task_run")
        self.request = request
        self.llm_base_url = llm_base_url
        self.backend_config = backend_config
        self.result: TaskRunResponse | None = None

    def status_response(self) -> TaskRunJobStatusResponse:
        with self._lock:
            return TaskRunJobStatusResponse(
                job_id=self.job_id,
                status=self.status,  # type: ignore[arg-type]
                progress=self._progress_snapshot(),
                result=self.result,
            )

    def run(self) -> None:
        with self._lock:
            self.status = "running"
            self.progress.phase = "queued"
            self.progress.policy = self.request.policy
            self.progress.max_steps = self.request.max_steps
        try:
            result = run_task_request(
                self.request,
                llm_base_url=self.llm_base_url,
                progress_callback=self.update_progress,
                backend_config=self.backend_config,
            )
            with self._lock:
                self.result = result
                self.status = "completed"
                self.progress.phase = "done"
                self.progress.elapsed_s = time.perf_counter() - self.created_at
        except Exception as exc:
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
                self.progress.phase = "failed"
                self.progress.error = self.error
                self.progress.elapsed_s = time.perf_counter() - self.created_at


class _CompareJob(_BaseJob):
    def __init__(self, request: CompareStrategiesRequest, *, llm_base_url: str, backend_config: BackendConfig) -> None:
        super().__init__(job_id=uuid4().hex, kind="compare")
        self.request = request
        self.llm_base_url = llm_base_url
        self.backend_config = backend_config
        self.result: CompareStrategiesResponse | None = None

    def status_response(self) -> CompareJobStatusResponse:
        with self._lock:
            return CompareJobStatusResponse(
                job_id=self.job_id,
                status=self.status,  # type: ignore[arg-type]
                progress=self._progress_snapshot(),
                result=self.result,
            )

    def run(self) -> None:
        with self._lock:
            self.status = "running"
            self.progress.phase = "queued"
            self.progress.total_policies = len(self.request.policies)
            self.progress.max_steps = self.request.max_steps
        try:
            result = compare_strategies(
                self.request,
                llm_base_url=self.llm_base_url,
                backend_config=self.backend_config,
                progress_callback=self.update_progress,
            )
            with self._lock:
                self.result = result
                self.status = "completed"
                self.progress.phase = "done"
                self.progress.elapsed_s = time.perf_counter() - self.created_at
        except Exception as exc:
            with self._lock:
                self.status = "failed"
                self.error = str(exc)
                self.progress.phase = "failed"
                self.progress.error = self.error
                self.progress.elapsed_s = time.perf_counter() - self.created_at


class AsyncJobStore:
    def __init__(self, backend_config: BackendConfig | None = None) -> None:
        self.backend_config = backend_config or load_backend_config()
        self._jobs: dict[str, _TaskRunJob | _CompareJob] = {}
        self._lock = threading.Lock()

    def start_task_run(self, request: TaskRunRequest, *, llm_base_url: str) -> AsyncJobAcceptedResponse:
        job = _TaskRunJob(request, llm_base_url=llm_base_url, backend_config=self.backend_config)
        self._start(job)
        return AsyncJobAcceptedResponse(job_id=job.job_id, kind="task_run")

    def start_compare(self, request: CompareStrategiesRequest, *, llm_base_url: str) -> AsyncJobAcceptedResponse:
        job = _CompareJob(request, llm_base_url=llm_base_url, backend_config=self.backend_config)
        self._start(job)
        return AsyncJobAcceptedResponse(job_id=job.job_id, kind="compare")

    def get_task_run(self, job_id: str) -> TaskRunJobStatusResponse:
        job = self._get(job_id, expected_kind="task_run")
        return job.status_response()

    def get_compare(self, job_id: str) -> CompareJobStatusResponse:
        job = self._get(job_id, expected_kind="compare")
        return job.status_response()

    def _start(self, job: _TaskRunJob | _CompareJob) -> None:
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=job.run, name=f"hyperreasoning-{job.kind}-{job.job_id[:8]}", daemon=True)
        thread.start()

    def _get(self, job_id: str, *, expected_kind: str) -> _TaskRunJob | _CompareJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.kind != expected_kind:
            raise KeyError(job_id)
        return job

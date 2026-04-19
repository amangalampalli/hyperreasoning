"""Pydantic request/response models for the local plugin backend."""

from __future__ import annotations

from typing import Any
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SearchPolicy = Literal["random", "heuristic", "rainbow", "oneshot"]


class ClientContext(BaseModel):
    """IDE/project context used for local cache placement and cloud grouping."""

    model_config = ConfigDict(extra="ignore")

    project_id: str | None = None
    project_name: str | None = None
    project_root: str | None = None
    task_root: str | None = None
    active_file: str | None = None


class TaskRunRequest(BaseModel):
    """One task-time search request from the IDE/plugin."""

    model_config = ConfigDict(extra="ignore")

    client_context: ClientContext | None = None
    prompt: str
    files: dict[str, str]
    target_files: list[str] = Field(default_factory=list)
    visible_test_file: str | None = None
    hidden_test_file: str | None = None
    language: str = "python"
    family: str = "custom_single_file"
    policy: SearchPolicy = "rainbow"
    proposal_source: Literal["heuristic", "llm", "hybrid"] = "heuristic"
    max_steps: int = 8
    max_verified_plans_per_task: int = 1
    allow_full_file_fallback: bool = True
    run_tests: bool = True
    run_hidden_tests: bool = False
    checkpoint_path: str | None = None
    seed: int = 123


class CompareStrategiesRequest(TaskRunRequest):
    """Request for running multiple search policies on one task."""

    policies: list[SearchPolicy] = Field(
        default_factory=lambda: ["heuristic", "rainbow", "oneshot"]
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str
    llm_base_url: str
    llm_reachable: bool


class StrategySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str
    task_id: str
    family: str
    total_reward: float
    steps: int
    compile_successes: int
    visible_passes: int
    hidden_passes: int = 0
    visible_tests_passed: int = 0
    visible_tests_total: int = 0
    hidden_tests_passed: int = 0
    hidden_tests_total: int = 0
    tests_passed: int = 0
    tests_total: int = 0
    fraction_tests_passed: float = 0.0
    best_bank_id: str | None = None
    compile_success: bool | None = None
    visible_test_passed: bool | None = None
    hidden_test_passed: bool | None = None
    compile_error: str | None = None
    hidden_test_stdout: str | None = None
    hidden_test_stderr: str | None = None
    hidden_test_returncode: int | None = None
    elapsed_s: float | None = None
    llm_requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunDiagnostic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy: str
    bank_id: str | None = None
    plan_id: str | None = None
    plan_signature: str | None = None
    strategy: str | None = None
    target_files: list[str] = Field(default_factory=list)
    status: Literal["success", "compile_failed", "visible_failed", "hidden_failed", "unknown"] = "unknown"
    is_best: bool = False
    compile_success: bool | None = None
    compile_error: str | None = None
    visible_test_passed: bool | None = None
    visible_test_returncode: int | None = None
    visible_test_stdout: str | None = None
    visible_test_stderr: str | None = None
    hidden_test_passed: bool | None = None
    hidden_test_returncode: int | None = None
    hidden_test_stdout: str | None = None
    hidden_test_stderr: str | None = None


class TaskRunResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: StrategySummary
    root_candidates: list[dict]
    plan_bank: dict = Field(default_factory=dict)
    nodes: list[dict]
    edges: list[tuple[str, str]]
    transitions: list[dict]
    best_plan: dict | None = None
    best_compiled_files: dict[str, str] = Field(default_factory=dict)
    verifier_summary: dict = Field(default_factory=dict)
    search_graph_events: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[RunDiagnostic] = Field(default_factory=list)
    run_id: str | None = None
    cache_key: str | None = None
    cache_hit: bool = False
    cloud_status: Literal["disabled", "pending", "synced", "failed", "unknown"] = "unknown"


class CompareStrategiesResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategies: list[TaskRunResponse]
    run_id: str | None = None
    cache_key: str | None = None
    cache_hit: bool = False
    cloud_status: Literal["disabled", "pending", "synced", "failed", "unknown"] = "unknown"


class JobProgressSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str = "queued"
    policy: str | None = None
    current_step: int = 0
    max_steps: int = 0
    llm_requests: int = 0
    last_llm_label: str | None = None
    last_llm_duration_s: float | None = None
    last_compile_duration_s: float | None = None
    last_test_duration_s: float | None = None
    compile_successes: int = 0
    visible_passes: int = 0
    roots: int | None = None
    plans: int | None = None
    action: str | None = None
    label_tier: str | None = None
    current_policy_index: int = 0
    total_policies: int = 1
    elapsed_s: float = 0.0
    error: str | None = None


class AsyncJobAcceptedResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["task_run", "compare"]


class TaskRunJobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["task_run"] = "task_run"
    status: Literal["queued", "running", "completed", "failed"]
    progress: JobProgressSnapshot
    result: TaskRunResponse | None = None


class CompareJobStatusResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: Literal["compare"] = "compare"
    status: Literal["queued", "running", "completed", "failed"]
    progress: JobProgressSnapshot
    result: CompareStrategiesResponse | None = None


class RunHistoryItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    cache_key: str
    project_id: str
    project_name: str | None = None
    project_root: str | None = None
    task_root: str | None = None
    active_file: str | None = None
    prompt_preview: str
    policy: str
    family: str
    created_at: str
    updated_at: str
    visible_passes: int = 0
    hidden_passes: int = 0
    compile_successes: int = 0
    total_reward: float = 0.0
    elapsed_s: float | None = None
    llm_requests: int = 0
    local_status: Literal["available", "missing"] = "available"
    cloud_status: Literal["disabled", "pending", "synced", "failed", "unknown"] = "unknown"
    cloud_object_path: str | None = None
    package_sha256: str | None = None


class RunHistoryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[RunHistoryItem]
    cloud_enabled: bool = False
    errors: list[str] = Field(default_factory=list)


class RunLoadResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: RunHistoryItem
    kind: Literal["task_run", "comparison"] = "task_run"
    request: TaskRunRequest | None = None
    result: TaskRunResponse | None = None
    compare_request: CompareStrategiesRequest | None = None
    compare_result: CompareStrategiesResponse | None = None


class RunSyncRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    client_context: ClientContext
    limit: int = 100


class RunSyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloud_enabled: bool
    uploaded: int = 0
    downloaded: int = 0
    failed: int = 0
    errors: list[str] = Field(default_factory=list)

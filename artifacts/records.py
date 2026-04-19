"""Structured persistent records for DSL attempts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PlanRecord(BaseModel):
    """Stored DSL candidate plan."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    strategy: str
    target_files: list[str]
    suspected_bug_types: list[str]
    invariants: list[str]
    subgoals: list[str]
    validation_checks: list[str]
    risks: list[str]
    touched_symbols: list[str]
    edit_style: str
    confidence: float | None = None
    notes: str


class PlanExecutionRecord(BaseModel):
    """Stored result of compiling and executing one plan."""

    model_config = ConfigDict(extra="forbid")

    plan_id: str
    compile_success: bool
    attempted_compile: bool = True
    compile_error: str | None = None
    compiled_files: dict[str, str] = Field(default_factory=dict)
    files_changed: list[str] = Field(default_factory=list)
    compile_latency_s: float | None = None
    visible_test_passed: bool | None = None
    visible_test_returncode: int | None = None
    visible_test_stdout: str | None = None
    visible_test_stderr: str | None = None
    test_latency_s: float | None = None
    hidden_summary: dict[str, Any] | None = None
    score: float | None = None


class AttemptRecord(BaseModel):
    """Stored full task attempt."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    attempt_id: str
    task_id: str
    family: str
    difficulty: str | None = None
    task_dir: str
    task_prompt: str
    language: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    dsl_candidates: list[PlanRecord] = Field(default_factory=list)
    selected_plan_id: str | None = None
    selected_plan_index: int | None = None
    plan_executions: list[PlanExecutionRecord] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

"""Deterministic repair stub for failed DSL branches."""

from __future__ import annotations

from dataclasses import dataclass

from env.dsl_schema import PlanDSL
from llm.prompt_utils import TaskContext


@dataclass(slots=True)
class RepairSuggestion:
    plan: PlanDSL
    reason: str


def propose_repair(task: TaskContext, plan: PlanDSL, *, feedback: str) -> RepairSuggestion:
    """Return a conservative local repair candidate without extra LLM calls."""

    payload = plan.to_dict()
    payload["strategy"] = "minimal_patch" if plan.strategy != "minimal_patch" else plan.strategy
    payload["subgoals"] = ["repair from feedback", *list(plan.subgoals)[:2]]
    payload["validation_checks"] = ["visible_tests", *list(plan.validation_checks)[:2]]
    payload["notes"] = f"repair:{feedback[:24].strip()}"[:24] or "repair attempt"
    payload["confidence"] = max(0.1, (plan.confidence or 0.5) - 0.15)
    repaired = PlanDSL.from_dict(
        payload,
        task_id=task.task_id,
        family=task.family,
        language=task.language,
        task_target_files=task.target_files,
    )
    return RepairSuggestion(plan=repaired, reason="deterministic_repair_stub")

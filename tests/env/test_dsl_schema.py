from __future__ import annotations

import pytest

from env.dsl_schema import PlanDSL


def test_plan_dsl_validates_expected_shape() -> None:
    plan = PlanDSL.from_dict(
        {
            "strategy": "minimal_patch",
            "target_files": ["retry.py"],
            "suspected_bug_types": ["retry_bug"],
            "invariants": ["keep public api"],
            "subgoals": ["fix retry loop"],
            "validation_checks": ["visible_tests"],
            "risks": ["none"],
            "touched_symbols": ["run_with_retry"],
            "edit_style": "surgical_patch",
            "confidence": 0.6,
            "notes": "repair retry logic",
        },
        task_id="task-1",
        family="async_retry_contract",
        language="python",
        task_target_files=["retry.py"],
    )
    assert plan.task_id == "task-1"
    assert plan.plan_id


def test_plan_dsl_rejects_invalid_strategy() -> None:
    with pytest.raises(ValueError):
        PlanDSL.from_dict(
            {
                "strategy": "totally_invalid",
                "target_files": ["retry.py"],
                "invariants": ["x"],
                "subgoals": ["y"],
                "validation_checks": ["z"],
                "edit_style": "surgical_patch",
                "notes": "bad",
            },
            task_id="task-1",
            family="async_retry_contract",
            language="python",
            task_target_files=["retry.py"],
        )

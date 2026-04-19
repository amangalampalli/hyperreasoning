"""Heuristic scoring and reward shaping for offline rollouts."""

from __future__ import annotations

from artifacts.records import PlanRecord


def score_plan_heuristic(plan: PlanRecord) -> float:
    """Cheap pre-compile heuristic rank score."""

    confidence = 0.0 if plan.confidence is None else 0.10 * plan.confidence
    small_scope_bonus = 0.05 / max(1, len(plan.target_files))
    symbol_bonus = 0.02 * min(len(plan.touched_symbols), 3)
    risk_penalty = 0.01 * max(0, len(plan.risks) - 1)
    return confidence + small_scope_bonus + symbol_bonus - risk_penalty


def score_branch_execution(
    *,
    plan: PlanRecord,
    compile_success: bool,
    compile_error: str | None,
    visible_test_passed: bool | None,
    files_changed: list[str],
) -> float:
    """Score used to rank branches for expansion."""

    score = score_plan_heuristic(plan)
    if compile_success:
        score += 0.25
    else:
        score -= 0.10
    if visible_test_passed is True:
        score += 1.0
    elif visible_test_passed is False:
        score -= 0.05
    if compile_error is not None:
        score -= 0.05
    if files_changed:
        score += 0.05 / max(1, len(files_changed))
    return score


def compute_reward(
    *,
    compile_success: bool,
    visible_test_passed: bool | None,
    valid_plan: bool,
    depth: int,
    attempted_compile: bool,
    compile_error: str | None,
) -> float:
    """Dense reward for offline RL transitions."""

    reward = 0.0
    if valid_plan:
        reward += 0.10
    if compile_success:
        reward += 0.35
    if visible_test_passed is True:
        reward += 1.00
    elif visible_test_passed is False:
        reward += 0.0
    if compile_error is not None:
        reward -= 0.08
    if not attempted_compile:
        reward -= 0.02
    reward -= 0.03 * depth
    return reward

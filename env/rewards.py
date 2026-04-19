"""Centralized reward shaping for DSL search control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RewardWeights:
    """Configurable dense reward weights."""

    all_tests_pass: float = 1.00
    compile_success: float = 0.35
    lint_success: float = 0.20
    test_pass_fraction: float = 0.30
    valid_dsl: float = 0.10
    per_search_expansion_penalty: float = 0.03
    normalized_token_cost_penalty: float = 0.03
    invalid_dsl_penalty: float = 0.05
    compile_runtime_crash_penalty: float = 0.08


@dataclass(slots=True)
class RewardBreakdown:
    """Named reward components for logging/debugging."""

    valid_dsl: float = 0.0
    compile_success: float = 0.0
    lint_success: float = 0.0
    test_fraction: float = 0.0
    all_tests_pass: float = 0.0
    expansion_penalty: float = 0.0
    token_cost_penalty: float = 0.0
    invalid_dsl_penalty: float = 0.0
    compile_runtime_crash_penalty: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.valid_dsl
            + self.compile_success
            + self.lint_success
            + self.test_fraction
            + self.all_tests_pass
            - self.expansion_penalty
            - self.token_cost_penalty
            - self.invalid_dsl_penalty
            - self.compile_runtime_crash_penalty
        )


def compute_search_reward(
    *,
    weights: RewardWeights | None = None,
    valid_dsl: bool,
    compile_success: bool,
    lint_success: bool = False,
    tests_pass_fraction: float | None = None,
    all_tests_passed: bool = False,
    search_expansions: int = 0,
    normalized_token_cost: float = 0.0,
    compile_or_runtime_crash: bool = False,
) -> RewardBreakdown:
    """Return a configurable reward breakdown."""

    w = weights or RewardWeights()
    breakdown = RewardBreakdown()
    if valid_dsl:
        breakdown.valid_dsl = w.valid_dsl
    else:
        breakdown.invalid_dsl_penalty = w.invalid_dsl_penalty
    if compile_success:
        breakdown.compile_success = w.compile_success
    if lint_success:
        breakdown.lint_success = w.lint_success
    if tests_pass_fraction is not None and tests_pass_fraction > 0.0:
        breakdown.test_fraction = w.test_pass_fraction * tests_pass_fraction
    if all_tests_passed:
        breakdown.all_tests_pass = w.all_tests_pass
    if search_expansions > 0:
        breakdown.expansion_penalty = w.per_search_expansion_penalty * float(search_expansions)
    if normalized_token_cost > 0.0:
        breakdown.token_cost_penalty = w.normalized_token_cost_penalty * normalized_token_cost
    if compile_or_runtime_crash:
        breakdown.compile_runtime_crash_penalty = w.compile_runtime_crash_penalty
    return breakdown

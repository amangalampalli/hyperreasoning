"""Canonical DSL search-control environment and synthetic collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from artifacts.records import PlanRecord
from data.task_store import TaskStore
from env.dsl_schema import PlanDSL
from env.heuristics import compute_reward, score_plan_heuristic
from env.verifier import CachedTaskVerifier, VerificationResult
from llm.llm_client import LocalLLMClient
from llm.prompt_utils import TaskContext
from llm.proposal import generate_candidate_plans, heuristic_candidate_plans, plan_signature, propose_dsl_candidates, select_diverse_plans


ROOT_BANK_ID = "__ROOT__"
BASE_ACTION_SPACE: tuple[str, ...] = (
    "SELECT_CHILD_0",
    "SELECT_CHILD_1",
    "SELECT_CHILD_2",
    "SELECT_CHILD_3",
    "REQUEST_MORE_CANDIDATES",
    "REFINE_CURRENT_PLAN",
    "COMPILE_TO_CODE",
    "BACKTRACK",
    "TERMINATE",
)
ACTION_SPACE: tuple[str, ...] = BASE_ACTION_SPACE + ("REPAIR_FROM_FEEDBACK",)
ACTION_TO_ID = {action: index for index, action in enumerate(ACTION_SPACE)}


class SearchControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episodes_per_task: int = 20
    max_steps_per_episode: int = 8
    max_bank_depth: int = 2
    root_candidate_batches: int = 1
    root_candidates_per_batch: int = 4
    max_root_plans: int = 6
    refinement_branching: int = 3
    select_child_slots: int = 4
    initial_root_reveal: int = 2
    initial_refine_reveal: int = 2
    request_batch_size: int = 2
    proposal_source: str = "heuristic"
    llm_proposal_temperature: float | None = None
    selection_epsilon: float = 0.28
    compiler_temp: float = 0.2
    run_tests: bool = True
    python_bin: str = "python"
    timeout_s: float = 12.0
    allow_full_file_fallback: bool = False
    max_verified_plans_per_task: int = 8
    seed: int = 123


class SearchControlTransition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: dict[str, Any]
    action: str
    reward: float
    next_state: dict[str, Any] | None
    done: bool
    valid_actions: list[str]
    info: dict[str, Any] = Field(default_factory=dict)


class PlanBankEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bank_id: str
    parent_bank_id: str | None = None
    depth: int
    plan_signature: str
    heuristic_score: float
    plan: dict[str, Any]
    child_bank_ids: list[str] = Field(default_factory=list)


class TaskPlanBank(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    proposal_source: str
    root_bank_ids: list[str] = Field(default_factory=list)
    entries: dict[str, PlanBankEntry] = Field(default_factory=dict)
    summary: dict[str, Any] = Field(default_factory=dict)


class SearchEpisode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_id: str
    family: str
    seed: int
    transitions: list[SearchControlTransition]
    nodes: list[dict[str, Any]]
    edges: list[tuple[str, str]]
    summary: dict[str, Any]


class TaskDatasetBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_bank: TaskPlanBank
    episodes: list[SearchEpisode]
    transitions: list[SearchControlTransition]
    summary: dict[str, Any]


class TaskSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    family: str
    policy: str
    root_candidates: list[dict[str, Any]]
    episode: SearchEpisode
    total_reward: float
    steps: int
    compile_successes: int
    visible_passes: int
    best_bank_id: str | None = None
    best_plan: dict[str, Any] | None = None
    best_verification: dict[str, Any] | None = None
    best_compiled_files: dict[str, str] = Field(default_factory=dict)
    verifier_summary: dict[str, Any] = Field(default_factory=dict)
    plan_bank: dict[str, Any] = Field(default_factory=dict)


@dataclass(slots=True)
class _RuntimeState:
    path: list[str]
    revealed_counts: dict[str, int]
    known_results: dict[str, VerificationResult]
    best_verified_bank_id: str | None
    steps_taken: int


@dataclass(slots=True)
class EpisodeStats:
    task_id: str
    family: str
    steps: int
    total_reward: float
    compile_successes: int
    visible_test_passes: int


def action_name_to_id(action: str) -> int:
    return ACTION_TO_ID[action]


def action_id_to_name(action_id: int) -> str:
    return ACTION_SPACE[action_id]


def encode_action_mask(valid_actions: list[str]) -> np.ndarray:
    mask = np.zeros(len(ACTION_SPACE), dtype=np.bool_)
    for action in valid_actions:
        if action in ACTION_TO_ID:
            mask[ACTION_TO_ID[action]] = True
    return mask


def _plan_to_record(plan: PlanDSL) -> PlanRecord:
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


def _confidence_key(plan: PlanDSL) -> float:
    return -1.0 if plan.confidence is None else plan.confidence


def _refine_root_candidates(
    task_context: TaskContext,
    *,
    config: SearchControlConfig,
    client: LocalLLMClient | None,
) -> list[PlanDSL]:
    candidates: list[PlanDSL] = []
    llm_client = client or LocalLLMClient()
    if config.proposal_source in {"llm", "hybrid"}:
        temps = [0.35, 0.55, 0.75, 0.9]
        for batch_index in range(config.root_candidate_batches):
            temperature = (
                config.llm_proposal_temperature
                if config.llm_proposal_temperature is not None
                else temps[min(batch_index, len(temps) - 1)]
            )
            result = propose_dsl_candidates(
                task_context,
                k=config.root_candidates_per_batch,
                temperature=temperature,
                client=llm_client,
                source=config.proposal_source,
            )
            assert isinstance(result, list)
            candidates.extend(result)
    if config.proposal_source in {"heuristic", "hybrid"}:
        candidates.extend(heuristic_candidate_plans(task_context, config.root_candidates_per_batch + 2))

    deduped: dict[str, PlanDSL] = {}
    for plan in candidates:
        signature = plan_signature(plan)
        existing = deduped.get(signature)
        if existing is None or _confidence_key(plan) > _confidence_key(existing):
            deduped[signature] = plan
    return select_diverse_plans(list(deduped.values()), config.max_root_plans)


def build_task_plan_bank(
    task_context: TaskContext,
    config: SearchControlConfig,
    *,
    client: LocalLLMClient | None = None,
) -> TaskPlanBank:
    entries: dict[str, PlanBankEntry] = {}
    bank_counter = 0

    def next_bank_id() -> str:
        nonlocal bank_counter
        bank_counter += 1
        return f"bank_{bank_counter:05d}"

    def add_entry(plan: PlanDSL, parent_bank_id: str | None, depth: int) -> str:
        bank_id = next_bank_id()
        entries[bank_id] = PlanBankEntry(
            bank_id=bank_id,
            parent_bank_id=parent_bank_id,
            depth=depth,
            plan_signature=plan_signature(plan),
            heuristic_score=score_plan_heuristic(_plan_to_record(plan)),
            plan=plan.model_dump(),
        )
        if depth < config.max_bank_depth:
            raw_children = generate_candidate_plans(
                task_context,
                parent_plan=plan,
                depth=depth + 1,
                k=config.refinement_branching,
                proposal_source="heuristic",
                client=client,
            )
            deduped_children: dict[str, PlanDSL] = {}
            for child in raw_children:
                signature = plan_signature(child)
                existing = deduped_children.get(signature)
                if existing is None or _confidence_key(child) > _confidence_key(existing):
                    deduped_children[signature] = child
            for child in select_diverse_plans(list(deduped_children.values()), config.refinement_branching):
                child_bank_id = add_entry(child, bank_id, depth + 1)
                entries[bank_id].child_bank_ids.append(child_bank_id)
        return bank_id

    root_candidates = _refine_root_candidates(task_context, config=config, client=client)
    root_bank_ids = [add_entry(plan, None, 0) for plan in root_candidates]

    return TaskPlanBank(
        task_id=task_context.task_id,
        family=task_context.family,
        proposal_source=config.proposal_source,
        root_bank_ids=root_bank_ids,
        entries=entries,
        summary={
            "task_id": task_context.task_id,
            "root_plans": len(root_bank_ids),
            "total_plans": len(entries),
            "max_bank_depth": max((entry.depth for entry in entries.values()), default=0),
            "avg_branching": (
                sum(len(entry.child_bank_ids) for entry in entries.values() if entry.child_bank_ids)
                / max(1, sum(1 for entry in entries.values() if entry.child_bank_ids))
            ),
        },
    )


class _SearchControlRuntime:
    def __init__(
        self,
        *,
        task_context: TaskContext,
        plan_bank: TaskPlanBank,
        verifier: CachedTaskVerifier,
        config: SearchControlConfig,
        rng: random.Random,
    ) -> None:
        self.task_context = task_context
        self.plan_bank = plan_bank
        self.verifier = verifier
        self.config = config
        self.rng = rng
        self._state: _RuntimeState | None = None
        self._episode_node_counter = 0

    def reset(self) -> dict[str, Any]:
        revealed_counts = {ROOT_BANK_ID: min(self.config.initial_root_reveal, len(self.plan_bank.root_bank_ids))}
        self._state = _RuntimeState(
            path=[],
            revealed_counts=revealed_counts,
            known_results={},
            best_verified_bank_id=None,
            steps_taken=0,
        )
        self._episode_node_counter = 0
        return self.build_state()

    def _require_state(self) -> _RuntimeState:
        if self._state is None:
            raise RuntimeError("Runtime must be reset before use")
        return self._state

    def _current_bank_id(self) -> str | None:
        state = self._require_state()
        return state.path[-1] if state.path else None

    def _children_for(self, bank_id: str | None) -> list[str]:
        if bank_id is None:
            return list(self.plan_bank.root_bank_ids)
        return list(self.plan_bank.entries[bank_id].child_bank_ids)

    def _revealed_children(self, bank_id: str | None) -> list[str]:
        state = self._require_state()
        key = ROOT_BANK_ID if bank_id is None else bank_id
        limit = state.revealed_counts.get(key, 0)
        return self._children_for(bank_id)[:limit]

    def _hidden_child_count(self, bank_id: str | None) -> int:
        state = self._require_state()
        key = ROOT_BANK_ID if bank_id is None else bank_id
        total = len(self._children_for(bank_id))
        return max(0, total - state.revealed_counts.get(key, 0))

    def valid_actions(self) -> list[str]:
        current_bank_id = self._current_bank_id()
        valid: list[str] = []
        visible_children = self._revealed_children(current_bank_id)
        for index in range(self.config.select_child_slots):
            if index < len(visible_children):
                valid.append(f"SELECT_CHILD_{index}")
        if self._hidden_child_count(current_bank_id) > 0:
            valid.append("REQUEST_MORE_CANDIDATES")
        if current_bank_id is not None and not visible_children and self._children_for(current_bank_id):
            valid.append("REFINE_CURRENT_PLAN")
        if current_bank_id is not None:
            known_result = self._require_state().known_results.get(current_bank_id)
            signature = self.plan_bank.entries[current_bank_id].plan_signature
            if known_result is None or known_result.cached:
                if self.verifier.can_verify(signature):
                    valid.append("COMPILE_TO_CODE")
            else:
                valid.append("COMPILE_TO_CODE")
        if current_bank_id is not None:
            valid.append("BACKTRACK")
        valid.append("TERMINATE")
        return valid

    def build_state(self) -> dict[str, Any]:
        state = self._require_state()
        current_bank_id = self._current_bank_id()
        current_entry = None if current_bank_id is None else self.plan_bank.entries[current_bank_id]
        current_result = None if current_bank_id is None else state.known_results.get(current_bank_id)
        child_slots: list[dict[str, Any]] = []
        for child_bank_id in self._revealed_children(current_bank_id)[: self.config.select_child_slots]:
            child_entry = self.plan_bank.entries[child_bank_id]
            child_plan = PlanDSL.model_validate(child_entry.plan)
            child_result = state.known_results.get(child_bank_id)
            child_slots.append(
                {
                    "bank_id": child_bank_id,
                    "strategy": child_plan.strategy,
                    "heuristic_score": child_entry.heuristic_score,
                    "compile_known": child_result is not None,
                    "compile_success": None if child_result is None else child_result.compile_success,
                    "visible_test_passed": None if child_result is None else child_result.visible_test_passed,
                }
            )
        best_result = None if state.best_verified_bank_id is None else state.known_results.get(state.best_verified_bank_id)
        return {
            "task_id": self.task_context.task_id,
            "family": self.task_context.family,
            "current_bank_id": current_bank_id,
            "current_depth": 0 if current_entry is None else current_entry.depth,
            "current_strategy": "ROOT" if current_entry is None else PlanDSL.model_validate(current_entry.plan).strategy,
            "current_heuristic_score": 0.0 if current_entry is None else current_entry.heuristic_score,
            "current_compile_success": None if current_result is None else current_result.compile_success,
            "current_visible_test_passed": None if current_result is None else current_result.visible_test_passed,
            "path_length": len(state.path),
            "remaining_steps": max(0, self.config.max_steps_per_episode - state.steps_taken),
            "visible_child_count": len(child_slots),
            "hidden_child_count": self._hidden_child_count(current_bank_id),
            "known_result_count": len(state.known_results),
            "best_compile_success": None if best_result is None else best_result.compile_success,
            "best_visible_test_passed": None if best_result is None else best_result.visible_test_passed,
            "compile_budget_remaining": max(0, self.verifier.max_verified_plans - self.verifier.cache_size),
            "child_slots": child_slots,
            "valid_actions": self.valid_actions(),
        }

    def _next_node_id(self) -> str:
        self._episode_node_counter += 1
        return f"state_{self._episode_node_counter:04d}"

    def step(self, action: str) -> tuple[dict[str, Any] | None, float, bool, dict[str, Any]]:
        valid_actions = self.valid_actions()
        state = self._require_state()
        info: dict[str, Any] = {"label_tier": "dsl_only"}
        if action not in valid_actions:
            reward = -0.08
            state.steps_taken += 1
            done = state.steps_taken >= self.config.max_steps_per_episode
            next_state = None if done else self.build_state()
            return next_state, reward, done, {**info, "invalid_action": True}

        current_bank_id = self._current_bank_id()
        reward = -0.01
        done = False
        if action.startswith("SELECT_CHILD_"):
            child_index = int(action.rsplit("_", 1)[-1])
            visible_children = self._revealed_children(current_bank_id)
            chosen_bank_id = visible_children[child_index]
            state.path.append(chosen_bank_id)
            chosen_entry = self.plan_bank.entries[chosen_bank_id]
            reward = 0.02 + min(0.18, chosen_entry.heuristic_score)
            info.update({"selected_bank_id": chosen_bank_id, "selection_score": chosen_entry.heuristic_score})
        elif action == "REQUEST_MORE_CANDIDATES":
            key = ROOT_BANK_ID if current_bank_id is None else current_bank_id
            hidden_before = self._hidden_child_count(current_bank_id)
            newly_revealed = min(self.config.request_batch_size, hidden_before)
            state.revealed_counts[key] = state.revealed_counts.get(key, 0) + newly_revealed
            reward = 0.01 + 0.02 * newly_revealed if newly_revealed else -0.04
            info.update({"new_candidates": newly_revealed})
        elif action == "REFINE_CURRENT_PLAN":
            assert current_bank_id is not None
            hidden_before = self._hidden_child_count(current_bank_id)
            newly_revealed = min(self.config.initial_refine_reveal, hidden_before)
            state.revealed_counts[current_bank_id] = state.revealed_counts.get(current_bank_id, 0) + newly_revealed
            reward = 0.03 + 0.02 * newly_revealed if newly_revealed else -0.04
            info.update({"new_refinements": newly_revealed})
        elif action == "COMPILE_TO_CODE":
            assert current_bank_id is not None
            if current_bank_id in state.known_results:
                known = state.known_results[current_bank_id]
                reward = -0.03 if known.cached else -0.02
                info.update({"redundant_compile": True, "label_tier": known.label_tier})
            else:
                entry = self.plan_bank.entries[current_bank_id]
                result = self.verifier.verify(self.task_context, PlanDSL.model_validate(entry.plan))
                state.known_results[current_bank_id] = result
                if result.compile_success or result.visible_test_passed:
                    best_bank_id = state.best_verified_bank_id
                    best_result = None if best_bank_id is None else state.known_results[best_bank_id]
                    if best_result is None or (result.visible_test_passed is True and best_result.visible_test_passed is not True):
                        state.best_verified_bank_id = current_bank_id
                reward = compute_reward(
                    compile_success=bool(result.compile_success),
                    visible_test_passed=result.visible_test_passed,
                    valid_plan=True,
                    depth=entry.depth,
                    attempted_compile=True,
                    compile_error=result.compile_error,
                )
                info.update(
                    {
                        "compile_success": result.compile_success,
                        "visible_test_passed": result.visible_test_passed,
                        "hidden_test_passed": result.hidden_test_passed,
                        "visible_test_returncode": result.visible_test_returncode,
                        "visible_test_stdout": result.visible_test_stdout,
                        "visible_test_stderr": result.visible_test_stderr,
                        "hidden_test_returncode": result.hidden_test_returncode,
                        "hidden_test_stdout": result.hidden_test_stdout,
                        "hidden_test_stderr": result.hidden_test_stderr,
                        "compile_error": result.compile_error,
                        "label_tier": result.label_tier,
                    }
                )
        elif action == "BACKTRACK":
            assert current_bank_id is not None
            current_result = state.known_results.get(current_bank_id)
            state.path.pop()
            reward = 0.02 if current_result and not current_result.compile_success else -0.01
        elif action == "TERMINATE":
            best_result = None if state.best_verified_bank_id is None else state.known_results.get(state.best_verified_bank_id)
            if best_result is not None and best_result.visible_test_passed is True:
                reward = 1.20
            elif best_result is not None and best_result.compile_success:
                reward = 0.20
            else:
                reward = -0.05 if state.steps_taken else -0.08
            done = True

        state.steps_taken += 1
        if state.steps_taken >= self.config.max_steps_per_episode and not done:
            done = True
            reward += 0.05 if state.best_verified_bank_id is not None else -0.02
        next_state = None if done else self.build_state()
        info.update({"steps_taken": state.steps_taken})
        return next_state, reward, done, info


def choose_policy_action(runtime: _SearchControlRuntime, *, rng: random.Random) -> str:
    valid_actions = runtime.valid_actions()
    if not valid_actions:
        return "TERMINATE"
    if rng.random() < runtime.config.selection_epsilon:
        return rng.choice(valid_actions)

    current_bank_id = runtime._current_bank_id()
    state = runtime._require_state()
    weights: dict[str, float] = {action: 0.05 for action in valid_actions}
    if "TERMINATE" in weights:
        if state.best_verified_bank_id is not None:
            best_result = state.known_results.get(state.best_verified_bank_id)
            if best_result is not None and best_result.visible_test_passed is True:
                weights["TERMINATE"] = 0.85
        elif state.steps_taken >= runtime.config.max_steps_per_episode - 1:
            weights["TERMINATE"] = 0.5
    if current_bank_id is not None:
        current_entry = runtime.plan_bank.entries[current_bank_id]
        current_known = state.known_results.get(current_bank_id)
        if "COMPILE_TO_CODE" in weights and current_known is None:
            weights["COMPILE_TO_CODE"] = 0.30 + max(0.0, min(0.4, current_entry.heuristic_score))
        if "BACKTRACK" in weights and current_known is not None and current_known.compile_success is False:
            weights["BACKTRACK"] = 0.55
        if "REFINE_CURRENT_PLAN" in weights and not runtime._revealed_children(current_bank_id):
            weights["REFINE_CURRENT_PLAN"] = 0.60
    visible_children = runtime._revealed_children(current_bank_id)
    for index, child_bank_id in enumerate(visible_children[: runtime.config.select_child_slots]):
        action = f"SELECT_CHILD_{index}"
        if action not in weights:
            continue
        child_score = runtime.plan_bank.entries[child_bank_id].heuristic_score
        weights[action] = 0.25 + max(0.0, min(0.45, child_score))
    if "REQUEST_MORE_CANDIDATES" in weights:
        weights["REQUEST_MORE_CANDIDATES"] = 0.28 if runtime._hidden_child_count(current_bank_id) > 0 else 0.02

    actions = list(weights)
    total = sum(max(0.0, weights[action]) for action in actions)
    if total <= 0:
        return rng.choice(valid_actions)
    threshold = rng.random() * total
    running = 0.0
    for action in actions:
        running += max(0.0, weights[action])
        if running >= threshold:
            return action
    return actions[-1]


def collect_task_dataset(
    task_context: TaskContext,
    config: SearchControlConfig,
    *,
    client: LocalLLMClient | None = None,
    verifier: CachedTaskVerifier | None = None,
) -> TaskDatasetBundle:
    llm_client = client or LocalLLMClient()
    plan_bank = build_task_plan_bank(task_context, config, client=llm_client)
    shared_verifier = verifier or CachedTaskVerifier(
        client=llm_client,
        run_tests=config.run_tests,
        python_bin=config.python_bin,
        timeout_s=config.timeout_s,
        compiler_temp=config.compiler_temp,
        allow_full_file_fallback=config.allow_full_file_fallback,
        max_verified_plans=config.max_verified_plans_per_task,
    )
    all_transitions: list[SearchControlTransition] = []
    episodes: list[SearchEpisode] = []
    base_seed = config.seed
    for episode_index in range(config.episodes_per_task):
        rng = random.Random(base_seed + episode_index)
        runtime = _SearchControlRuntime(
            task_context=task_context,
            plan_bank=plan_bank,
            verifier=shared_verifier,
            config=config,
            rng=rng,
        )
        initial_state = runtime.reset()
        nodes: list[dict[str, Any]] = [{"node_id": "state_0000", "parent_id": None, "state": initial_state, "incoming_action": None}]
        edges: list[tuple[str, str]] = []
        transitions: list[SearchControlTransition] = []
        parent_node_id = "state_0000"
        done = False
        while not done:
            state_before = runtime.build_state()
            valid_actions = runtime.valid_actions()
            action = choose_policy_action(runtime, rng=rng)
            next_state, reward, done, info = runtime.step(action)
            transition = SearchControlTransition(
                state=state_before,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                valid_actions=valid_actions,
                info=info,
            )
            transitions.append(transition)
            all_transitions.append(transition)
            if next_state is not None:
                node_id = runtime._next_node_id()
                nodes.append(
                    {
                        "node_id": node_id,
                        "parent_id": parent_node_id,
                        "state": next_state,
                        "incoming_action": action,
                        "reward": reward,
                        "label_tier": info.get("label_tier"),
                    }
                )
                edges.append((parent_node_id, node_id))
                parent_node_id = node_id
        episodes.append(
            SearchEpisode(
                episode_id=f"{task_context.task_id}_ep_{episode_index:04d}",
                task_id=task_context.task_id,
                family=task_context.family,
                seed=base_seed + episode_index,
                transitions=transitions,
                nodes=nodes,
                edges=edges,
                summary={
                    "steps": len(transitions),
                    "terminated_with_success": any(t.info.get("visible_test_passed") is True for t in transitions),
                    "compile_actions": sum(1 for t in transitions if t.action == "COMPILE_TO_CODE"),
                    "verified_actions": sum(1 for t in transitions if t.info.get("label_tier") in {"compile", "visible_test"}),
                },
            )
        )
    summary = {
        "task_id": task_context.task_id,
        "family": task_context.family,
        "episodes": len(episodes),
        "transitions": len(all_transitions),
        "bank_root_plans": len(plan_bank.root_bank_ids),
        "bank_total_plans": len(plan_bank.entries),
        "verifier": shared_verifier.summary(),
        "avg_steps_per_episode": len(all_transitions) / max(1, len(episodes)),
        "completed": True,
    }
    return TaskDatasetBundle(plan_bank=plan_bank, episodes=episodes, transitions=all_transitions, summary=summary)


def _choose_runtime_action(
    runtime: _SearchControlRuntime,
    *,
    policy: str,
    rng: random.Random,
    agent: Any | None = None,
    encoder: Any | None = None,
) -> str:
    valid_actions = runtime.valid_actions()
    if not valid_actions:
        return "TERMINATE"
    if policy == "random":
        return rng.choice(valid_actions)
    if policy == "heuristic":
        return choose_policy_action(runtime, rng=rng)
    if policy == "rainbow":
        if agent is None or encoder is None:
            raise ValueError("Rainbow policy requires agent and encoder")
        action_id = agent.act(encoder.encode_state(runtime.build_state()), encode_action_mask(valid_actions), epsilon=0.0)
        return action_id_to_name(action_id)
    if policy == "oneshot":
        select_action = next((action for action in valid_actions if action.startswith("SELECT_CHILD_")), None)
        if select_action is not None:
            return select_action
        if "COMPILE_TO_CODE" in valid_actions:
            return "COMPILE_TO_CODE"
        return "TERMINATE"
    raise ValueError(f"Unsupported policy {policy}")


def run_single_task_search(
    task_context: TaskContext,
    config: SearchControlConfig,
    *,
    policy: str = "heuristic",
    client: LocalLLMClient | None = None,
    verifier: CachedTaskVerifier | None = None,
    agent: Any | None = None,
    encoder: Any | None = None,
    seed: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> TaskSearchResult:
    if progress_callback is not None:
        progress_callback(
            {
                "event": "search_started",
                "task_id": task_context.task_id,
                "family": task_context.family,
                "policy": policy,
                "max_steps": config.max_steps_per_episode,
            }
        )
    llm_client = client or LocalLLMClient()
    plan_bank = build_task_plan_bank(task_context, config, client=llm_client)
    if progress_callback is not None:
        progress_callback(
            {
                "event": "plan_bank_built",
                "task_id": task_context.task_id,
                "family": task_context.family,
                "policy": policy,
                "root_candidates": len(plan_bank.root_bank_ids),
                "total_plans": len(plan_bank.entries),
                "plan_bank": plan_bank.model_dump(),
            }
        )
    shared_verifier = verifier or CachedTaskVerifier(
        client=llm_client,
        run_tests=config.run_tests,
        python_bin=config.python_bin,
        timeout_s=config.timeout_s,
        compiler_temp=config.compiler_temp,
        allow_full_file_fallback=config.allow_full_file_fallback,
        max_verified_plans=config.max_verified_plans_per_task,
    )
    rng = random.Random(config.seed if seed is None else seed)
    runtime = _SearchControlRuntime(
        task_context=task_context,
        plan_bank=plan_bank,
        verifier=shared_verifier,
        config=config,
        rng=rng,
    )
    initial_state = runtime.reset()
    if progress_callback is not None:
        progress_callback(
            {
                "event": "state_initialized",
                "task_id": task_context.task_id,
                "family": task_context.family,
                "policy": policy,
                "current_bank_id": initial_state.get("current_bank_id"),
                "visible_child_bank_ids": [
                    slot.get("bank_id")
                    for slot in initial_state.get("child_slots", [])
                    if isinstance(slot, dict) and isinstance(slot.get("bank_id"), str)
                ],
            }
        )

    def build_episode_node(
        *,
        node_id: str,
        parent_id: str | None,
        state_snapshot: dict[str, Any],
        incoming_action: str | None,
        reward: float | None = None,
        label_tier: str | None = None,
    ) -> dict[str, Any]:
        bank_id = state_snapshot.get("current_bank_id")
        entry = None if bank_id is None else runtime.plan_bank.entries.get(str(bank_id))
        plan_payload = None if entry is None else dict(entry.plan)
        plan_text = None
        if plan_payload is not None:
            try:
                plan_text = PlanDSL.model_validate(plan_payload).to_compact_text()
            except Exception:
                plan_text = None
        known_result = None if bank_id is None else runtime._require_state().known_results.get(str(bank_id))
        return {
            "node_id": node_id,
            "parent_id": parent_id,
            "state": state_snapshot,
            "incoming_action": incoming_action,
            "reward": reward,
            "label_tier": label_tier,
            "bank_id": bank_id,
            "heuristic_score": None if entry is None else entry.heuristic_score,
            "plan": plan_payload,
            "plan_text": plan_text,
            "compile_success": None if known_result is None else known_result.compile_success,
            "visible_test_passed": None if known_result is None else known_result.visible_test_passed,
            "hidden_test_passed": None if known_result is None else known_result.hidden_test_passed,
        }

    nodes: list[dict[str, Any]] = [
        build_episode_node(
            node_id="state_0000",
            parent_id=None,
            state_snapshot=initial_state,
            incoming_action=None,
        )
    ]
    edges: list[tuple[str, str]] = []
    transitions: list[SearchControlTransition] = []
    parent_node_id = "state_0000"
    done = False
    total_reward = 0.0
    while not done:
        state_before = runtime.build_state()
        valid_actions = runtime.valid_actions()
        action = _choose_runtime_action(runtime, policy=policy, rng=rng, agent=agent, encoder=encoder)
        next_state, reward, done, info = runtime.step(action)
        total_reward += reward
        transition = SearchControlTransition(
            state=state_before,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            valid_actions=valid_actions,
            info=info,
        )
        transitions.append(transition)
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "step_completed",
                    "task_id": task_context.task_id,
                    "family": task_context.family,
                    "policy": policy,
                    "step": runtime._require_state().steps_taken,
                    "max_steps": config.max_steps_per_episode,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "current_bank_id_before": state_before.get("current_bank_id"),
                    "selected_bank_id": info.get("selected_bank_id"),
                    "parent_bank_id": state_before.get("current_bank_id"),
                    "current_bank_id": None if next_state is None else next_state.get("current_bank_id"),
                    "best_bank_id": runtime._require_state().best_verified_bank_id,
                    "path_bank_ids": list(runtime._require_state().path),
                    "visible_child_bank_ids_before": [
                        slot.get("bank_id")
                        for slot in state_before.get("child_slots", [])
                        if isinstance(slot, dict) and isinstance(slot.get("bank_id"), str)
                    ],
                    "visible_child_bank_ids_after": [
                        slot.get("bank_id")
                        for slot in (next_state or {}).get("child_slots", [])
                        if isinstance(slot, dict) and isinstance(slot.get("bank_id"), str)
                    ],
                    "compile_success": info.get("compile_success"),
                    "visible_test_passed": info.get("visible_test_passed"),
                    "hidden_test_passed": info.get("hidden_test_passed"),
                    "label_tier": info.get("label_tier"),
                }
            )
        if next_state is not None:
            node_id = runtime._next_node_id()
            nodes.append(
                build_episode_node(
                    node_id=node_id,
                    parent_id=parent_node_id,
                    state_snapshot=next_state,
                    incoming_action=action,
                    reward=reward,
                    label_tier=info.get("label_tier"),
                )
            )
            edges.append((parent_node_id, node_id))
            parent_node_id = node_id
    stats = EpisodeStats(
        task_id=task_context.task_id,
        family=task_context.family,
        steps=runtime._require_state().steps_taken,
        total_reward=total_reward,
        compile_successes=sum(1 for result in runtime._require_state().known_results.values() if result.compile_success),
        visible_test_passes=sum(1 for result in runtime._require_state().known_results.values() if result.visible_test_passed is True),
    )
    best_bank_id = runtime._require_state().best_verified_bank_id
    best_plan = None
    best_verification = None
    best_compiled_files: dict[str, str] = {}
    if best_bank_id is not None:
        best_plan = dict(plan_bank.entries[best_bank_id].plan)
        verification = runtime._require_state().known_results.get(best_bank_id)
        if verification is not None:
            best_verification = verification.model_dump()
            best_compiled_files = dict(verification.compiled_files)
    root_candidates = [
        {
            "bank_id": bank_id,
            "heuristic_score": plan_bank.entries[bank_id].heuristic_score,
            "plan": plan_bank.entries[bank_id].plan,
        }
        for bank_id in plan_bank.root_bank_ids
    ]
    episode = SearchEpisode(
        episode_id=f"{task_context.task_id}_{policy}_single",
        task_id=task_context.task_id,
        family=task_context.family,
        seed=config.seed if seed is None else seed,
        transitions=transitions,
        nodes=nodes,
        edges=edges,
        summary={
            "steps": len(transitions),
            "terminated_with_success": any(t.info.get("visible_test_passed") is True for t in transitions),
            "compile_actions": sum(1 for t in transitions if t.action == "COMPILE_TO_CODE"),
            "verified_actions": sum(1 for t in transitions if t.info.get("label_tier") in {"compile", "visible_test"}),
        },
    )
    if progress_callback is not None:
        progress_callback(
            {
                "event": "search_completed",
                "task_id": task_context.task_id,
                "family": task_context.family,
                "policy": policy,
                "steps": stats.steps,
                "compile_successes": stats.compile_successes,
                "visible_passes": stats.visible_test_passes,
                "best_bank_id": best_bank_id,
            }
        )
    return TaskSearchResult(
        task_id=task_context.task_id,
        family=task_context.family,
        policy=policy,
        root_candidates=root_candidates,
        episode=episode,
        total_reward=total_reward,
        steps=stats.steps,
        compile_successes=stats.compile_successes,
        visible_passes=stats.visible_test_passes,
        best_bank_id=best_bank_id,
        best_plan=best_plan,
        best_verification=best_verification,
        best_compiled_files=best_compiled_files,
        verifier_summary=shared_verifier.summary(),
        plan_bank=plan_bank.model_dump(),
    )


class DSLSearchEnv:
    """Canonical reset/step API for DSL search control."""

    def __init__(
        self,
        *,
        tasks_root: Path | None = None,
        task_store: TaskStore | None = None,
        config: SearchControlConfig | None = None,
        llm_base_url: str = "http://127.0.0.1:8080",
        seed: int = 123,
    ) -> None:
        self.config = config or SearchControlConfig()
        self.llm_base_url = llm_base_url
        self._rng = random.Random(seed)
        self.task_store = task_store or TaskStore.from_tasks_root(tasks_root or Path("data/generated_tasks"))
        self._task: TaskContext | None = None
        self._runtime: _SearchControlRuntime | None = None
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._compile_successes = 0
        self._visible_test_passes = 0

    @property
    def num_actions(self) -> int:
        return len(ACTION_SPACE)

    def _resolve_task(self, task: str | Path | TaskContext | None) -> TaskContext:
        if task is None:
            return self.task_store.sample(self._rng)
        return self.task_store.get(task)

    def reset(self, task: str | Path | TaskContext | None = None, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng.seed(seed)
        self._task = self._resolve_task(task)
        client = LocalLLMClient(base_url=self.llm_base_url)
        verifier = CachedTaskVerifier(
            client=client,
            run_tests=self.config.run_tests,
            python_bin=self.config.python_bin,
            timeout_s=self.config.timeout_s,
            compiler_temp=self.config.compiler_temp,
            allow_full_file_fallback=self.config.allow_full_file_fallback,
            max_verified_plans=self.config.max_verified_plans_per_task,
        )
        plan_bank = build_task_plan_bank(self._task, self.config, client=client)
        self._runtime = _SearchControlRuntime(
            task_context=self._task,
            plan_bank=plan_bank,
            verifier=verifier,
            config=self.config,
            rng=self._rng,
        )
        obs = self._runtime.reset()
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._compile_successes = 0
        self._visible_test_passes = 0
        info = {"task_id": self._task.task_id, "family": self._task.family, "valid_actions": self.valid_actions(), "action_mask": encode_action_mask(self.valid_actions())}
        return obs, info

    def valid_actions(self) -> list[str]:
        if self._runtime is None:
            return []
        actions = list(self._runtime.valid_actions())
        current_bank_id = self._runtime._current_bank_id()
        if current_bank_id is not None:
            known_result = self._runtime._require_state().known_results.get(current_bank_id)
            if known_result is not None and (known_result.compile_success is False or known_result.visible_test_passed is False):
                actions.append("REPAIR_FROM_FEEDBACK")
        return actions

    def current_action_mask(self) -> np.ndarray:
        return encode_action_mask(self.valid_actions())

    def heuristic_action(self) -> int:
        if self._runtime is None:
            raise RuntimeError("Environment not reset")
        return action_name_to_id(choose_policy_action(self._runtime, rng=self._rng))

    def step(self, action: int | str) -> tuple[dict[str, Any] | None, float, bool, bool, dict[str, Any]]:
        if self._runtime is None or self._task is None:
            raise RuntimeError("Environment not reset")
        action_name = action_id_to_name(action) if isinstance(action, int) else action
        if action_name == "REPAIR_FROM_FEEDBACK":
            proxy_action = "REFINE_CURRENT_PLAN" if "REFINE_CURRENT_PLAN" in self._runtime.valid_actions() else "BACKTRACK"
            next_obs, reward, done, info = self._runtime.step(proxy_action)
            info = {**info, "repair_proxy_used": True, "repair_proxy_action": proxy_action}
        else:
            next_obs, reward, done, info = self._runtime.step(action_name)
        self._episode_reward += reward
        self._episode_steps += 1
        if info.get("compile_success") is True:
            self._compile_successes += 1
        if info.get("visible_test_passed") is True:
            self._visible_test_passes += 1
        terminated = done
        truncated = False
        valid_actions = [] if next_obs is None else self.valid_actions()
        info = {**info, "task_id": self._task.task_id, "family": self._task.family, "valid_actions": valid_actions, "action_mask": encode_action_mask(valid_actions)}
        if terminated or truncated:
            info["episode_stats"] = asdict(self.episode_stats())
        return next_obs, reward, terminated, truncated, info

    def episode_stats(self) -> EpisodeStats:
        if self._task is None:
            raise RuntimeError("Environment not reset")
        return EpisodeStats(
            task_id=self._task.task_id,
            family=self._task.family,
            steps=self._episode_steps,
            total_reward=self._episode_reward,
            compile_successes=self._compile_successes,
            visible_test_passes=self._visible_test_passes,
        )

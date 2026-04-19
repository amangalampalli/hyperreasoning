"""Proposal pipeline for DSL search plans."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1

import orjson

from env.dsl_schema import PlanDSL, parse_plan_dicts
from llm.llm_client import LocalLLMClient, LocalLLMError
from llm.parsing import extract_first_json_value, extract_plan_list
from llm.prompt_utils import TaskContext, build_proposal_prompt


@dataclass(slots=True)
class _FallbackSpec:
    strategy: str
    bug_types: list[str]
    invariants: list[str]
    subgoals: list[str]
    checks: list[str]
    risks: list[str]
    symbols: list[str]
    edit_style: str
    notes: str


HEURISTIC_VARIANTS_BY_FAMILY: dict[str, list[_FallbackSpec]] = {
    "streaming_parser_reentrancy": [
        _FallbackSpec(
        strategy="state_fix",
        bug_types=["state_leakage", "boundary_split"],
        invariants=[
            "chunked feeds preserve carry-over state",
            "parser reset clears session state",
            "multi-frame chunks emit all complete frames in order",
        ],
        subgoals=[
            "inspect feed/reset/close state transitions",
            "preserve buffered suffixes across chunk boundaries",
            "ensure reuse after reset does not leak prior partial state",
        ],
        checks=["visible_tests", "hidden_chunk_boundaries", "repeated_call_semantics"],
        risks=["fix may clear too much buffered state", "reset semantics may still leave hidden stale fields"],
        symbols=["feed", "reset", "close"],
        edit_style="state_machine_repair",
        notes="Repair buffer lifecycle and parser reuse semantics.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["buffer_reset", "multi_frame_loss"],
            invariants=["completed frames emit in order", "leftover suffix remains buffered"],
            subgoals=["inspect frame consumption", "preserve trailing buffered bytes"],
            checks=["visible_tests", "hidden_chunk_boundaries"],
            risks=["may fix buffering but break reset semantics"],
            symbols=["feed"],
            edit_style="surgical_patch",
            notes="Patch frame consumption without losing buffered suffixes.",
        ),
        _FallbackSpec(
            strategy="state_machine_repair",
            bug_types=["reentrancy_state_corruption", "close_lifecycle_bug"],
            invariants=["close rejects truncated frames", "reused parser starts clean after reset"],
            subgoals=["repair close/reset lifecycle", "guard against stale pending headers"],
            checks=["repeated_call_semantics", "visible_tests"],
            risks=["close handling may still swallow buffered errors"],
            symbols=["reset", "close"],
            edit_style="state_machine_repair",
            notes="Repair parser lifecycle transitions.",
        ),
    ],
    "async_retry_contract": [
        _FallbackSpec(
        strategy="retry_policy_adjustment",
        bug_types=["retryable_exception_misclassification", "cancellation_swallowing"],
        invariants=[
            "cancelled operations surface immediately",
            "non-idempotent or committed failures do not retry",
            "retryable failures back off deterministically",
        ],
        subgoals=[
            "separate cancellation from retryable failures",
            "guard commit/idempotency exits",
            "preserve retry loop accounting",
        ],
        checks=["visible_tests", "repeated_call_semantics", "retry_contract_consistency"],
        risks=["broad except blocks may still catch cancellation", "retry guard order may still retry committed failures"],
        symbols=["run_with_retry", "invoke_with_retry", "call_with_retry"],
        edit_style="surgical_patch",
        notes="Tighten retry classification and exit conditions.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["cancellation_swallowing"],
            invariants=["cancelled calls are never retried", "non-retryable errors surface once"],
            subgoals=["split CancelledError from Exception path", "keep retry loop accounting intact"],
            checks=["visible_tests", "retry_contract_consistency"],
            risks=["commit/idempotency guards may still be wrong"],
            symbols=["run_with_retry", "call_with_retry"],
            edit_style="surgical_patch",
            notes="Patch broad exception handling around cancellation.",
        ),
        _FallbackSpec(
            strategy="state_fix",
            bug_types=["committed_retry", "idempotency_guard_bug"],
            invariants=["committed failures stop retries", "non-idempotent operations fail fast"],
            subgoals=["inspect commit/idempotent gates", "preserve backoff only for safe retries"],
            checks=["visible_tests", "repeated_call_semantics"],
            risks=["cancellation path may regress if guards reorder badly"],
            symbols=["run_with_retry", "invoke_with_retry"],
            edit_style="localized_refactor",
            notes="Repair safe-retry exit conditions.",
        ),
    ],
    "ast_transform_scope_bug": [
        _FallbackSpec(
        strategy="scope_handling_fix",
        bug_types=["scope_capture", "shadowed_binding_rename"],
        invariants=[
            "module binding is renamed consistently",
            "shadowed locals and comprehension targets are preserved",
            "nested closures that resolve to module scope still update",
        ],
        subgoals=[
            "reconstruct lexical scope boundaries",
            "differentiate module references from shadowed locals",
            "preserve valid syntax after transform",
        ],
        checks=["visible_tests", "scope_shadowing", "comprehension_scope_consistency"],
        risks=["fix may still miss comprehension scope", "renaming globals may break closure capture"],
        symbols=["visit_Name", "visit_FunctionDef", "visit_ListComp"],
        edit_style="localized_refactor",
        notes="Make rename decisions scope-aware instead of name-blind.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["comprehension_scope", "shadowed_binding_rename"],
            invariants=["comprehension targets stay local", "module references still rename"],
            subgoals=["treat comprehension as separate scope", "avoid renaming comprehension binders"],
            checks=["visible_tests", "comprehension_scope_consistency"],
            risks=["closure captures may still be mis-resolved"],
            symbols=["visit_ListComp", "visit_Name"],
            edit_style="surgical_patch",
            notes="Patch comprehension-specific scope handling.",
        ),
        _FallbackSpec(
            strategy="algorithm_switch",
            bug_types=["scope_resolution_bug"],
            invariants=["rename decisions follow lexical resolution", "globals remain renamable"],
            subgoals=["separate binding collection from resolution", "track globals explicitly per function"],
            checks=["scope_shadowing", "visible_tests"],
            risks=["broader refactor may introduce syntax-preservation bugs"],
            symbols=["ScopeAwareRenamer", "_gather_globals", "_collect_bindings"],
            edit_style="localized_refactor",
            notes="Restructure rename resolution around lexical scope lookup.",
        ),
    ],
    "cache_invalidation_dependency": [
        _FallbackSpec(
        strategy="invalidation_fix",
        bug_types=["transitive_invalidation", "reverse_edge_staleness"],
        invariants=[
            "all transitive dependents are invalidated",
            "unrelated cache entries stay valid",
            "dependency rewrites update reverse edges",
        ],
        subgoals=[
            "walk downstream dependency graph correctly",
            "remove stale reverse edges on rule rewrites",
            "preserve exact invalidation rather than global clears",
        ],
        checks=["visible_tests", "transitive_invalidation", "multi_update_consistency"],
        risks=["over-invalidation hides stale graph bugs", "rule rewrites may still leak old reverse edges"],
        symbols=["invalidate", "set_derived", "get"],
        edit_style="state_machine_repair",
        notes="Repair transitive invalidation and reverse-dependency maintenance.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["transitive_invalidation"],
            invariants=["all downstream dependents are invalidated", "unrelated cache entries survive"],
            subgoals=["walk reverse graph transitively", "avoid global clears"],
            checks=["visible_tests", "transitive_invalidation"],
            risks=["stale reverse-edge drift may remain"],
            symbols=["invalidate"],
            edit_style="surgical_patch",
            notes="Patch downstream invalidation traversal.",
        ),
        _FallbackSpec(
            strategy="algorithm_switch",
            bug_types=["reverse_edge_staleness"],
            invariants=["rule rewrites remove stale reverse links", "recomputation remains precise"],
            subgoals=["repair reverse-edge bookkeeping on set_derived", "preserve exact cache scope"],
            checks=["visible_tests", "multi_update_consistency"],
            risks=["fixing reverse links may still miss transitive invalidation"],
            symbols=["set_derived", "_reverse"],
            edit_style="localized_refactor",
            notes="Repair reverse dependency graph maintenance.",
        ),
    ],
    "descriptor_property_mro": [
        _FallbackSpec(
        strategy="interface_alignment",
        bug_types=["descriptor_precedence", "mro_shadowing"],
        invariants=[
            "subclass attributes shadow inherited descriptors",
            "leftmost MRO resolution wins",
            "exported state matches attribute lookup semantics",
        ],
        subgoals=[
            "collect managed fields in correct MRO order",
            "respect property/plain attribute shadowing",
            "preserve descriptor get/set behavior",
        ],
        checks=["visible_tests", "mro_shadowing_checks", "attribute_lookup_consistency"],
        risks=["collection may still include shadowed descriptors", "fix may reverse precedence across mixins"],
        symbols=["collect_managed_attributes", "export_managed_state"],
        edit_style="localized_refactor",
        notes="Align managed-field discovery with real Python attribute lookup.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["mro_shadowing"],
            invariants=["subclass overrides win", "properties shadow inherited descriptors"],
            subgoals=["check discovery order", "skip shadowed descriptor names"],
            checks=["visible_tests", "mro_shadowing_checks"],
            risks=["mixins may still resolve in reverse order"],
            symbols=["collect_managed_attributes"],
            edit_style="surgical_patch",
            notes="Patch managed-field discovery order and shadowing.",
        ),
        _FallbackSpec(
            strategy="scope_handling_fix",
            bug_types=["descriptor_precedence"],
            invariants=["export matches runtime attribute lookup", "leftmost MRO wins"],
            subgoals=["align export list with descriptor precedence", "respect non-descriptor overrides"],
            checks=["visible_tests", "attribute_lookup_consistency"],
            risks=["export may still include hidden base descriptors"],
            symbols=["export_managed_state", "collect_managed_attributes"],
            edit_style="localized_refactor",
            notes="Align export behavior with descriptor precedence.",
        ),
    ],
    "incremental_build_graph_bug": [
        _FallbackSpec(
        strategy="rebuild_propagation_fix",
        bug_types=["cycle_propagation_bug", "reverse_edge_staleness"],
        invariants=[
            "affected set includes all downstream nodes",
            "rebuild order respects dependencies",
            "dependency rewrites remove stale reverse edges",
        ],
        subgoals=[
            "repair affected-node traversal",
            "preserve topological rebuild ordering",
            "keep reverse graph in sync on edits",
        ],
        checks=["visible_tests", "rebuild_propagation_fix", "cycle_detection"],
        risks=["ordering may be correct while affected set is incomplete", "stale reverse edges can over-rebuild after rewrites"],
        symbols=["set_node", "affected_nodes", "plan_rebuild"],
        edit_style="state_machine_repair",
        notes="Repair downstream propagation and graph rewrite consistency.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["missing_transitive_rebuild"],
            invariants=["all downstream nodes rebuild", "dependency order remains valid"],
            subgoals=["fix affected set traversal", "keep topological ordering stable"],
            checks=["visible_tests", "rebuild_propagation_fix"],
            risks=["reverse-edge rewrite bugs may remain"],
            symbols=["affected_nodes", "plan_rebuild"],
            edit_style="surgical_patch",
            notes="Patch affected-set propagation only.",
        ),
        _FallbackSpec(
            strategy="algorithm_switch",
            bug_types=["reverse_edge_staleness", "cycle_detection_bug"],
            invariants=["rewritten dependencies update reverse graph", "cycles still raise"],
            subgoals=["repair reverse-edge cleanup", "preserve cycle detection"],
            checks=["visible_tests", "cycle_detection"],
            risks=["topological order could regress while fixing rewrites"],
            symbols=["set_node", "_reverse"],
            edit_style="localized_refactor",
            notes="Repair dependency rewrite bookkeeping.",
        ),
    ],
    "serializer_roundtrip_escape": [
        _FallbackSpec(
        strategy="roundtrip_escape_fix",
        bug_types=["escape_roundtrip_violation", "dangling_escape_handling"],
        invariants=[
            "encode/decode round-trip preserves delimiters and empty strings",
            "invalid escape sequences are rejected",
            "field count prefix stays authoritative",
        ],
        subgoals=[
            "repair escape decoding state machine",
            "validate malformed trailing escapes",
            "preserve exact field count checks",
        ],
        checks=["visible_tests", "roundtrip_stability", "malformed_escape_rejection"],
        risks=["happy-path fields may pass while malformed inputs still parse", "decoder may still mis-handle empty fields"],
        symbols=["encode_fields", "serialize_fields", "decode_fields", "deserialize_fields"],
        edit_style="surgical_patch",
        notes="Restore strict round-trip and malformed-input handling for the codec.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["escape_roundtrip_violation"],
            invariants=["delimiters round-trip correctly", "empty fields are preserved"],
            subgoals=["repair decoder split logic", "preserve exact separators"],
            checks=["visible_tests", "roundtrip_stability"],
            risks=["malformed escape validation may remain broken"],
            symbols=["decode_fields", "deserialize_fields"],
            edit_style="surgical_patch",
            notes="Patch decoder split/escape handling.",
        ),
        _FallbackSpec(
            strategy="state_fix",
            bug_types=["dangling_escape_handling", "count_mismatch"],
            invariants=["dangling escapes fail fast", "field count mismatch is rejected"],
            subgoals=["validate trailing escape state", "enforce count prefix"],
            checks=["visible_tests", "malformed_escape_rejection"],
            risks=["round-trip happy path may regress if decoder state machine changes badly"],
            symbols=["decode_fields", "deserialize_fields"],
            edit_style="localized_refactor",
            notes="Repair decoder validation path.",
        ),
    ],
    "stateful_iterator_resume_bug": [
        _FallbackSpec(
        strategy="state_fix",
        bug_types=["iterator_resume_duplication", "checkpoint_aliasing"],
        invariants=[
            "restored iterator resumes at exact next element",
            "exhausted checkpoints remain exhausted",
            "checkpoint state is isolated from external mutation",
        ],
        subgoals=[
            "store enough cursor state in checkpoints",
            "preserve terminal state",
            "avoid aliasing caller-owned groups during restore",
        ],
        checks=["visible_tests", "repeated_call_semantics", "resume_checkpoint_consistency"],
        risks=["resumption may still duplicate elements", "checkpoint snapshots may still share mutable nested lists"],
        symbols=["checkpoint", "from_checkpoint", "__next__"],
        edit_style="state_machine_repair",
        notes="Repair checkpoint fidelity and resumed iterator state isolation.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["iterator_resume_duplication"],
            invariants=["resume starts at exact next element", "empty groups remain skipped"],
            subgoals=["store inner cursor correctly", "avoid duplicating resumed values"],
            checks=["visible_tests", "resume_checkpoint_consistency"],
            risks=["exhausted-state handling may remain wrong"],
            symbols=["checkpoint", "__next__"],
            edit_style="surgical_patch",
            notes="Patch checkpoint cursor fidelity.",
        ),
        _FallbackSpec(
            strategy="algorithm_switch",
            bug_types=["checkpoint_aliasing", "terminal_state_bug"],
            invariants=["restored exhausted iterators stay exhausted", "checkpoint snapshots isolate mutable group state"],
            subgoals=["copy nested state into checkpoints", "restore exhausted flag correctly"],
            checks=["visible_tests", "repeated_call_semantics"],
            risks=["cursor positions may still drift after snapshot restore"],
            symbols=["checkpoint", "from_checkpoint"],
            edit_style="localized_refactor",
            notes="Repair restore semantics and snapshot isolation.",
        ),
    ],
    "multi_file_interface_drift": [
        _FallbackSpec(
        strategy="coordinated_multi_file_patch",
        bug_types=["interface_shape_drift", "multi_file_contract_mismatch"],
        invariants=[
            "api and client agree on parameters and return shape",
            "public contract stays stable across direct and helper callers",
            "empty or filtered results preserve documented semantics",
        ],
        subgoals=[
            "identify canonical API surface",
            "synchronize caller/callee assumptions across files",
            "preserve filtering defaults and summary keys",
        ],
        checks=["visible_tests", "multi_file_contract_consistency", "interface_shape_consistency"],
        risks=["fixing only one file leaves hidden direct-call failures", "compatibility shim may drift from prompt contract"],
        symbols=["summarize_records", "build_summary", "list_ids", "average_score"],
        edit_style="multi_file_sync",
        notes="Synchronize refactored caller/callee contract across all touched files.",
    ),
        _FallbackSpec(
            strategy="multi_file_contract_fix",
            bug_types=["interface_shape_drift"],
            invariants=["API keys match callers", "defaults remain stable"],
            subgoals=["restore canonical return shape", "sync keyword names across files"],
            checks=["visible_tests", "interface_shape_consistency"],
            risks=["direct callers may still break if only helper layer changes"],
            symbols=["summarize_records", "build_summary"],
            edit_style="multi_file_sync",
            notes="Restore canonical API contract.",
        ),
        _FallbackSpec(
            strategy="interface_alignment",
            bug_types=["caller_callee_mismatch"],
            invariants=["client helpers match API signature", "empty results keep documented semantics"],
            subgoals=["align helper arguments", "align summary key names"],
            checks=["visible_tests", "multi_file_contract_consistency"],
            risks=["return-shape shim may hide deeper semantic drift"],
            symbols=["list_ids", "average_score", "summarize_records"],
            edit_style="multi_file_sync",
            notes="Align helper usage with API surface.",
        ),
    ],
    "concurrency_safe_memoization": [
        _FallbackSpec(
        strategy="locking_fix",
        bug_types=["lock_scope_bug", "duplicate_work_under_contention"],
        invariants=[
            "same-key concurrent calls share one computation",
            "different keys can proceed in parallel",
            "exceptions are not cached as permanent values",
        ],
        subgoals=[
            "introduce per-key in-flight coordination",
            "avoid global lock around computation",
            "clear failed in-flight state without poisoning the cache",
        ],
        checks=["visible_tests", "thread_safety_under_contention", "exception_cache_policy"],
        risks=["global locking can serialize unrelated keys", "failure paths may still leave stale in-flight state"],
        symbols=["memoize_threadsafe", "concurrent_memoize", "safe_memoize"],
        edit_style="localized_refactor",
        notes="Repair key-scoped locking and non-poisoning failure handling.",
    ),
        _FallbackSpec(
            strategy="minimal_patch",
            bug_types=["duplicate_work_under_contention"],
            invariants=["same-key requests compute once", "cache hits remain fast"],
            subgoals=["introduce in-flight guard", "preserve cache lookup path"],
            checks=["visible_tests", "thread_safety_under_contention"],
            risks=["unrelated keys may still serialize"],
            symbols=["memoize_threadsafe", "safe_memoize"],
            edit_style="surgical_patch",
            notes="Patch same-key contention handling.",
        ),
        _FallbackSpec(
            strategy="algorithm_switch",
            bug_types=["exception_cache_poisoning", "global_lock_contention"],
            invariants=["exceptions are not cached permanently", "different keys can execute in parallel"],
            subgoals=["separate cache and in-flight state", "avoid global compute lock"],
            checks=["visible_tests", "exception_cache_policy"],
            risks=["same-key waiters may still observe stale in-flight state"],
            symbols=["memoize_threadsafe", "concurrent_memoize", "safe_memoize"],
            edit_style="localized_refactor",
            notes="Split in-flight coordination from cache state.",
        ),
    ],
}


def _dedupe_key(plan: PlanDSL) -> str:
    return sha1(
        orjson.dumps(
            {
                "strategy": plan.strategy,
                "target_files": sorted(plan.target_files),
                "bug_types": sorted(plan.suspected_bug_types),
                "invariants": sorted(plan.invariants),
            },
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()


def plan_signature(plan: PlanDSL) -> str:
    """Return a stable human-auditable signature for plan identity."""

    return "|".join(
        [
            plan.strategy,
            ",".join(sorted(plan.target_files)),
            ",".join(sorted(plan.suspected_bug_types)),
            ",".join(sorted(plan.touched_symbols)),
            ",".join(sorted(plan.validation_checks)),
        ]
    )


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _plan_similarity(left: PlanDSL, right: PlanDSL) -> float:
    """Estimate whether two plans are materially the same branch."""

    same_strategy = 1.0 if left.strategy == right.strategy else 0.0
    bug_similarity = _jaccard_similarity(set(left.suspected_bug_types), set(right.suspected_bug_types))
    symbol_similarity = _jaccard_similarity(set(left.touched_symbols), set(right.touched_symbols))
    check_similarity = _jaccard_similarity(set(left.validation_checks), set(right.validation_checks))
    target_similarity = _jaccard_similarity(set(left.target_files), set(right.target_files))
    return (
        0.35 * same_strategy
        + 0.25 * bug_similarity
        + 0.20 * symbol_similarity
        + 0.10 * check_similarity
        + 0.10 * target_similarity
    )


def _are_materially_distinct(candidate: PlanDSL, accepted: list[PlanDSL]) -> bool:
    """Reject near-clone branches even if wording differs."""

    for existing in accepted:
        if plan_signature(candidate) == plan_signature(existing):
            return False
        if _plan_similarity(candidate, existing) >= 0.78:
            return False
    return True


def _rank_for_diversity(plan: PlanDSL) -> tuple[float, int, int, int]:
    confidence = -1.0 if plan.confidence is None else plan.confidence
    return (
        confidence,
        len(set(plan.suspected_bug_types)),
        len(set(plan.touched_symbols)),
        len(set(plan.validation_checks)),
    )


def _select_diverse_plans(candidates: list[PlanDSL], k: int) -> list[PlanDSL]:
    """Keep up to k plans while forcing branch diversity."""

    chosen: list[PlanDSL] = []
    used_strategies: set[str] = set()
    ordered = sorted(candidates, key=_rank_for_diversity, reverse=True)

    # First pass: one materially distinct plan per strategy where possible.
    for plan in ordered:
        if plan.strategy in used_strategies:
            continue
        if _are_materially_distinct(plan, chosen):
            chosen.append(plan)
            used_strategies.add(plan.strategy)
        if len(chosen) >= k:
            return chosen

    # Second pass: fill remaining slots with distinct plans even if strategy repeats.
    for plan in ordered:
        if plan in chosen:
            continue
        if _are_materially_distinct(plan, chosen):
            chosen.append(plan)
        if len(chosen) >= k:
            return chosen

    return chosen


def select_diverse_plans(candidates: list[PlanDSL], k: int) -> list[PlanDSL]:
    """Public wrapper for diversity-filtered plan selection."""

    return _select_diverse_plans(candidates, k)


def _choose_best(existing: PlanDSL, candidate: PlanDSL) -> PlanDSL:
    existing_confidence = -1.0 if existing.confidence is None else existing.confidence
    candidate_confidence = -1.0 if candidate.confidence is None else candidate.confidence
    return candidate if candidate_confidence > existing_confidence else existing


def heuristic_fallback_plan(task: TaskContext) -> PlanDSL:
    """Generate one family-aware fallback plan."""

    specs = HEURISTIC_VARIANTS_BY_FAMILY.get(task.family)
    if not specs:
        return PlanDSL.from_dict(
            {
                "strategy": "minimal_patch",
                "target_files": task.target_files,
                "suspected_bug_types": ["implementation_bug"],
                "invariants": ["restore visible tests", "preserve surrounding behavior"],
                "subgoals": ["inspect target files", "make the smallest fix consistent with the prompt"],
                "validation_checks": ["visible_tests"],
                "risks": ["fallback heuristic may be too generic"],
                "touched_symbols": [],
                "edit_style": "surgical_patch",
                "confidence": 0.2,
                "notes": "Fallback plan based on task prompt and target files.",
            },
            task_id=task.task_id,
            family=task.family,
            language=task.language,
            task_target_files=task.target_files,
        )

    spec = specs[0]
    return PlanDSL.from_dict(
        {
            "strategy": spec.strategy,
            "target_files": task.target_files,
            "suspected_bug_types": spec.bug_types,
            "invariants": spec.invariants,
            "subgoals": spec.subgoals,
            "validation_checks": spec.checks,
            "risks": spec.risks,
            "touched_symbols": spec.symbols,
            "edit_style": spec.edit_style,
            "confidence": 0.2,
            "notes": spec.notes,
        },
        task_id=task.task_id,
        family=task.family,
        language=task.language,
        task_target_files=task.target_files,
    )


def heuristic_candidate_plans(task: TaskContext, k: int) -> list[PlanDSL]:
    """Generate multiple cheap family-aware candidate plans."""

    specs = HEURISTIC_VARIANTS_BY_FAMILY.get(task.family)
    if not specs:
        return [heuristic_fallback_plan(task)]

    plans: list[PlanDSL] = []
    for index, spec in enumerate(specs[:k], start=1):
        confidence = max(0.15, 0.8 - 0.15 * (index - 1))
        plans.append(
            PlanDSL.from_dict(
                {
                    "plan_id": f"{task.task_id}:heuristic:{index}",
                    "strategy": spec.strategy,
                    "target_files": task.target_files,
                    "suspected_bug_types": spec.bug_types,
                    "invariants": spec.invariants[:3],
                    "subgoals": spec.subgoals[:3],
                    "validation_checks": spec.checks[:3],
                    "risks": spec.risks[:2],
                    "touched_symbols": spec.symbols[:3],
                    "edit_style": spec.edit_style,
                    "confidence": confidence,
                    "notes": spec.notes,
                },
                task_id=task.task_id,
                family=task.family,
                language=task.language,
                task_target_files=task.target_files,
            )
        )
    return _select_diverse_plans(plans, k)


def _parse_plans_from_output(raw_output: str, task: TaskContext) -> list[PlanDSL]:
    payload = extract_first_json_value(raw_output)
    raw_plans = extract_plan_list(payload)
    return parse_plan_dicts(
        raw_plans,
        task_id=task.task_id,
        family=task.family,
        language=task.language,
        task_target_files=task.target_files,
    )


def propose_dsl_candidates(
    task: TaskContext,
    k: int = 3,
    temperature: float = 0.7,
    *,
    client: LocalLLMClient | None = None,
    return_debug: bool = False,
    source: str = "heuristic",
) -> list[PlanDSL] | dict[str, object]:
    """Ask the local model for candidate DSL branches, with heuristic fallback."""

    llm_client = client or LocalLLMClient()
    prompt = build_proposal_prompt(task, k=k)
    raw_output: str | None = None
    fallback_used = False
    proposal_source = source
    parsed_plans: list[PlanDSL] = []

    if source in {"llm", "hybrid"}:
        try:
            raw_output = llm_client.complete(
                prompt,
                temperature=temperature,
                max_tokens=700,
                request_label="proposal",
            )
            parsed_plans = _parse_plans_from_output(raw_output, task)
        except (LocalLLMError, ValueError):
            parsed_plans = []
    if source in {"heuristic", "hybrid"}:
        heuristic_plans = heuristic_candidate_plans(task, k=k)
        if source == "heuristic":
            parsed_plans = heuristic_plans
        else:
            parsed_plans = parsed_plans + heuristic_plans

    deduped: dict[str, PlanDSL] = {}
    for plan in parsed_plans:
        key = _dedupe_key(plan)
        if key in deduped:
            deduped[key] = _choose_best(deduped[key], plan)
        else:
            deduped[key] = plan

    plans = _select_diverse_plans(list(deduped.values()), k)
    if not plans:
        plans = [heuristic_fallback_plan(task)]
        fallback_used = True
        proposal_source = "fallback"
    plans = plans[:k]
    if not return_debug:
        return plans
    return {
        "plans": plans,
        "raw_prompt": prompt,
        "raw_response": raw_output,
        "fallback_used": fallback_used,
        "proposal_source": proposal_source,
        "plan_signatures": [plan_signature(plan) for plan in plans],
    }


def pretty_print_plans(plans: list[PlanDSL]) -> str:
    """Render a compact multi-plan debug view."""

    lines: list[str] = []
    for index, plan in enumerate(plans, start=1):
        lines.append(f"{index}. {plan.to_compact_text()}")
        lines.append(f"   invariants: {', '.join(plan.invariants)}")
        lines.append(f"   subgoals: {', '.join(plan.subgoals)}")
        lines.append(f"   risks: {', '.join(plan.risks) if plan.risks else '-'}")
    return "\n".join(lines)


def generate_candidate_plans(
    task_context: TaskContext,
    *,
    parent_plan: PlanDSL | None = None,
    depth: int = 0,
    k: int = 3,
    proposal_source: str = "heuristic",
    client: LocalLLMClient | None = None,
) -> list[PlanDSL]:
    """Generate root or refined child plans for search-control collection."""

    if parent_plan is None:
        if proposal_source == "heuristic":
            return heuristic_candidate_plans(task_context, k)
        result = propose_dsl_candidates(
            task_context,
            k=k,
            client=client,
            return_debug=False,
            source=proposal_source,
        )
        assert isinstance(result, list)
        return result

    candidates = []
    bug_types = list(parent_plan.suspected_bug_types) or ["implementation_bug"]
    touched_symbols = list(parent_plan.touched_symbols)
    validation_checks = list(parent_plan.validation_checks)
    subgoals = list(parent_plan.subgoals)
    invariants = list(parent_plan.invariants)

    variants = [
        {
            "strategy": parent_plan.strategy,
            "target_files": parent_plan.target_files,
            "suspected_bug_types": bug_types[:1],
            "invariants": invariants[:2],
            "subgoals": subgoals[:2] or ["narrow fix scope"],
            "validation_checks": validation_checks[:2] or ["visible_tests"],
            "risks": list(parent_plan.risks)[:1],
            "touched_symbols": touched_symbols[:1] or touched_symbols,
            "edit_style": parent_plan.edit_style,
            "confidence": (parent_plan.confidence or 0.5) - 0.05,
            "notes": f"focus {bug_types[0]}",
        },
        {
            "strategy": parent_plan.strategy,
            "target_files": parent_plan.target_files,
            "suspected_bug_types": bug_types[-1:],
            "invariants": invariants[:2],
            "subgoals": subgoals[-2:] or subgoals[:2],
            "validation_checks": validation_checks[-2:] or validation_checks[:2] or ["visible_tests"],
            "risks": list(parent_plan.risks)[-1:],
            "touched_symbols": touched_symbols[-2:] or touched_symbols[:1],
            "edit_style": parent_plan.edit_style,
            "confidence": (parent_plan.confidence or 0.5) - 0.08,
            "notes": "refine validation focus",
        },
        {
            "strategy": "minimal_patch" if parent_plan.strategy != "minimal_patch" else "algorithm_switch",
            "target_files": parent_plan.target_files,
            "suspected_bug_types": bug_types[:1],
            "invariants": invariants[:2],
            "subgoals": ["alternate branch", *subgoals[:1]],
            "validation_checks": validation_checks[:2] or ["visible_tests"],
            "risks": list(parent_plan.risks)[:1],
            "touched_symbols": touched_symbols[:2],
            "edit_style": "surgical_patch" if parent_plan.edit_style != "surgical_patch" else parent_plan.edit_style,
            "confidence": (parent_plan.confidence or 0.5) - 0.10,
            "notes": "alternate local branch",
        },
    ]
    for payload in variants:
        candidates.append(
            PlanDSL.from_dict(
                payload,
                task_id=task_context.task_id,
                family=task_context.family,
                language=task_context.language,
                task_target_files=task_context.target_files,
            )
        )
    for heuristic_plan in heuristic_candidate_plans(task_context, k + 2):
        if plan_signature(heuristic_plan) != plan_signature(parent_plan):
            candidates.append(heuristic_plan)
    return select_diverse_plans(candidates, k)

"""Pydantic schema and helpers for the DSL planning layer."""

from __future__ import annotations

from hashlib import sha1
from typing import Any, Iterable, Mapping, Sequence

import orjson
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


DEFAULT_ALLOWED_STRATEGIES: set[str] = {
    "minimal_patch",
    "state_fix",
    "interface_alignment",
    "retry_policy_adjustment",
    "scope_handling_fix",
    "invalidation_fix",
    "roundtrip_escape_fix",
    "locking_fix",
    "rebuild_propagation_fix",
    "multi_file_contract_fix",
    "algorithm_switch",
    "coordinated_multi_file_patch",
    "state_machine_repair",
}


ALLOWED_STRATEGIES_BY_FAMILY: dict[str, set[str]] = {
    "streaming_parser_reentrancy": {"minimal_patch", "state_fix", "state_machine_repair"},
    "async_retry_contract": {"minimal_patch", "retry_policy_adjustment", "state_fix"},
    "ast_transform_scope_bug": {"minimal_patch", "scope_handling_fix", "algorithm_switch"},
    "cache_invalidation_dependency": {"minimal_patch", "invalidation_fix", "algorithm_switch"},
    "descriptor_property_mro": {"minimal_patch", "interface_alignment", "scope_handling_fix"},
    "incremental_build_graph_bug": {"minimal_patch", "rebuild_propagation_fix", "algorithm_switch"},
    "serializer_roundtrip_escape": {"minimal_patch", "roundtrip_escape_fix", "state_fix"},
    "stateful_iterator_resume_bug": {"minimal_patch", "state_fix", "algorithm_switch"},
    "multi_file_interface_drift": {
        "interface_alignment",
        "multi_file_contract_fix",
        "coordinated_multi_file_patch",
    },
    "concurrency_safe_memoization": {"minimal_patch", "locking_fix", "algorithm_switch"},
}


ALLOWED_EDIT_STYLES: set[str] = {
    "surgical_patch",
    "localized_refactor",
    "multi_file_sync",
    "state_machine_repair",
}


def normalize_token(value: str) -> str:
    """Normalize short categorical values into underscore-delimited tokens."""

    return value.strip().lower().replace(" ", "_").replace("-", "_")


def normalize_string_list(value: Any) -> list[str]:
    """Normalize list-like fields into a deduplicated string list."""

    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Sequence):
        items = [item for item in value if item is not None]
    else:
        raise ValueError(f"Expected a string or sequence of strings, got {type(value)!r}")

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        token = item.strip()
        if token and token not in seen:
            normalized.append(token)
            seen.add(token)
    return normalized


def validate_target_files(candidate_files: Iterable[str], task_target_files: Sequence[str]) -> list[str]:
    """Return only the candidate paths that match task target files."""

    allowed = set(task_target_files)
    result: list[str] = []
    seen: set[str] = set()
    for path in candidate_files:
        if path in allowed and path not in seen:
            result.append(path)
            seen.add(path)
    return result


def _make_plan_id(task_id: str, strategy: str, target_files: Sequence[str], notes: str) -> str:
    digest = sha1(
        orjson.dumps(
            {
                "task_id": task_id,
                "strategy": strategy,
                "target_files": list(target_files),
                "notes": notes,
            },
            option=orjson.OPT_SORT_KEYS,
        )
    ).hexdigest()[:12]
    return f"{task_id}:{strategy}:{digest}"


class PlanDSL(BaseModel):
    """Compact, structured branch/search plan."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    plan_id: str = ""
    task_id: str
    family: str
    language: str
    strategy: str
    target_files: list[str]
    suspected_bug_types: list[str] = Field(default_factory=list)
    invariants: list[str]
    subgoals: list[str]
    validation_checks: list[str]
    risks: list[str] = Field(default_factory=list)
    touched_symbols: list[str] = Field(default_factory=list)
    edit_style: str
    confidence: float | None = None
    notes: str

    @field_validator("family", mode="before")
    @classmethod
    def _normalize_family(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("family must be a string")
        normalized = normalize_token(value)
        if not normalized:
            raise ValueError("family is required")
        return normalized

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("language must be a string")
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("language is required")
        return normalized

    @field_validator("strategy", mode="before")
    @classmethod
    def _normalize_strategy(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("strategy must be a string")
        normalized = normalize_token(value)
        if not normalized:
            raise ValueError("strategy is required")
        return normalized

    @field_validator(
        "target_files",
        "suspected_bug_types",
        "invariants",
        "subgoals",
        "validation_checks",
        "risks",
        "touched_symbols",
        mode="before",
    )
    @classmethod
    def _normalize_lists(cls, value: Any) -> list[str]:
        return normalize_string_list(value)

    @field_validator("edit_style", mode="before")
    @classmethod
    def _normalize_edit_style(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("edit_style must be a string")
        normalized = normalize_token(value)
        if normalized not in ALLOWED_EDIT_STYLES:
            raise ValueError(f"Unsupported edit_style {normalized!r}")
        return normalized

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def _validate_model(self) -> "PlanDSL":
        allowed = ALLOWED_STRATEGIES_BY_FAMILY.get(self.family, set()) | DEFAULT_ALLOWED_STRATEGIES
        if self.strategy not in allowed:
            raise ValueError(f"Unsupported strategy {self.strategy!r} for family {self.family!r}")
        if not self.task_id:
            raise ValueError("task_id is required")
        if not self.target_files:
            raise ValueError("target_files must contain at least one file")
        if not self.invariants:
            raise ValueError("invariants must contain at least one entry")
        if not self.subgoals:
            raise ValueError("subgoals must contain at least one entry")
        if not self.validation_checks:
            raise ValueError("validation_checks must contain at least one entry")
        if not self.notes:
            raise ValueError("notes is required")
        if not self.plan_id:
            self.plan_id = _make_plan_id(self.task_id, self.strategy, self.target_files, self.notes)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready dictionary."""

        return self.model_dump()

    def to_compact_text(self) -> str:
        """Render a compact single-line plan description."""

        confidence_text = "?" if self.confidence is None else f"{self.confidence:.2f}"
        return (
            f"[{self.plan_id}] strategy={self.strategy} files={','.join(self.target_files)} "
            f"bugs={','.join(self.suspected_bug_types) or '-'} checks={','.join(self.validation_checks)} "
            f"style={self.edit_style} confidence={confidence_text} notes={self.notes}"
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        task_id: str,
        family: str,
        language: str,
        task_target_files: Sequence[str] | None = None,
    ) -> "PlanDSL":
        """Parse one plan dictionary with task-derived defaults."""

        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected mapping payload, got {type(payload)!r}")

        raw_target_files = normalize_string_list(payload.get("target_files"))
        if task_target_files:
            target_files = validate_target_files(raw_target_files, task_target_files) or list(task_target_files)
        else:
            target_files = raw_target_files

        hydrated = {
            "plan_id": payload.get("plan_id", ""),
            "task_id": payload.get("task_id") or task_id,
            "family": payload.get("family") or family,
            "language": payload.get("language") or language,
            "strategy": payload.get("strategy", ""),
            "target_files": target_files,
            "suspected_bug_types": payload.get("suspected_bug_types", []),
            "invariants": payload.get("invariants", []),
            "subgoals": payload.get("subgoals", []),
            "validation_checks": payload.get("validation_checks", []),
            "risks": payload.get("risks", []),
            "touched_symbols": payload.get("touched_symbols", []),
            "edit_style": payload.get("edit_style", ""),
            "confidence": payload.get("confidence"),
            "notes": payload.get("notes", ""),
        }
        return cls.model_validate(hydrated)

    @classmethod
    def parse_list(
        cls,
        items: Sequence[Any],
        *,
        task_id: str,
        family: str,
        language: str,
        task_target_files: Sequence[str] | None = None,
    ) -> list["PlanDSL"]:
        """Best-effort parse of a list of raw plan payloads."""

        plans: list[PlanDSL] = []
        for item in items:
            try:
                plans.append(
                    cls.from_dict(
                        item,
                        task_id=task_id,
                        family=family,
                        language=language,
                        task_target_files=task_target_files,
                    )
                )
            except ValidationError:
                continue
            except ValueError:
                continue
        return plans


def parse_plan_dicts(
    items: Sequence[Any],
    *,
    task_id: str,
    family: str,
    language: str,
    task_target_files: Sequence[str] | None = None,
) -> list[PlanDSL]:
    """Convenience wrapper for parsing lists of raw plan payloads."""

    return PlanDSL.parse_list(
        items,
        task_id=task_id,
        family=family,
        language=language,
        task_target_files=task_target_files,
    )

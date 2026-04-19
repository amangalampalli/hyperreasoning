"""Filesystem persistence for DSL attempt artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from .records import AttemptRecord, PlanRecord


def make_run_id() -> str:
    """Generate a stable run identifier for one script invocation."""

    return datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")


def make_attempt_id(task_id: str, idx: int | None = None) -> str:
    """Generate a deterministic attempt identifier within a run."""

    if idx is None:
        return f"attempt_{task_id}"
    return f"attempt_{task_id}_{idx:04d}"


def save_json(path: Path, obj: Any) -> None:
    """Write pretty UTF-8 JSON to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = obj
    if hasattr(obj, "model_dump"):
        payload = obj.model_dump()
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS))


def save_raw_text(path: Path, text: str) -> None:
    """Write raw UTF-8 text to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_jsonl(path: Path, rows: list[Any]) -> None:
    """Write JSONL rows to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded_lines: list[bytes] = []
    for row in rows:
        payload = row.model_dump() if hasattr(row, "model_dump") else row
        encoded_lines.append(orjson.dumps(payload))
    path.write_bytes(b"\n".join(encoded_lines) + (b"\n" if encoded_lines else b""))


def save_attempt_record(record: AttemptRecord, output_dir: Path) -> Path:
    """Save one attempt record plus compiled file payloads."""

    attempt_dir = output_dir / record.run_id / record.task_id / record.attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    save_json(attempt_dir / "attempt_record.json", record)
    save_json(attempt_dir / "plans.json", [plan.model_dump() for plan in record.dsl_candidates])
    save_raw_text(attempt_dir / "plans.dsl.txt", render_plan_records(record.dsl_candidates))
    for index, execution in enumerate(record.plan_executions):
        if execution.compiled_files:
            save_json(
                attempt_dir / f"plan_{index}_compiled_files.json",
                execution.compiled_files,
            )
    return attempt_dir


def render_plan_records(plans: list[PlanRecord]) -> str:
    """Render stored DSL plans in a compact human-readable text format."""

    lines: list[str] = []
    for index, plan in enumerate(plans, start=1):
        lines.append(f"Plan {index}")
        lines.append(f"  plan_id: {plan.plan_id}")
        lines.append(f"  strategy: {plan.strategy}")
        lines.append(f"  target_files: {', '.join(plan.target_files)}")
        lines.append(f"  suspected_bug_types: {', '.join(plan.suspected_bug_types) or '-'}")
        lines.append(f"  invariants: {', '.join(plan.invariants)}")
        lines.append(f"  subgoals: {', '.join(plan.subgoals)}")
        lines.append(f"  validation_checks: {', '.join(plan.validation_checks)}")
        lines.append(f"  risks: {', '.join(plan.risks) or '-'}")
        lines.append(f"  touched_symbols: {', '.join(plan.touched_symbols) or '-'}")
        lines.append(f"  edit_style: {plan.edit_style}")
        lines.append(f"  confidence: {plan.confidence if plan.confidence is not None else '?'}")
        lines.append(f"  notes: {plan.notes}")
        if index != len(plans):
            lines.append("")
    return "\n".join(lines) + ("\n" if lines else "")

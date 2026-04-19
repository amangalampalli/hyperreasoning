"""Helpers for recovering JSON payloads from noisy LLM output."""

from __future__ import annotations

import json
import re
from typing import Any

import orjson


def strip_markdown_fences(text: str) -> str:
    """Remove common Markdown code fences while keeping inner content."""

    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)```", stripped, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return stripped


def extract_first_json_value(text: str) -> Any:
    """Extract the first valid JSON object or array embedded in text.

    The stdlib decoder is used only for its raw-decode boundary finding;
    the actual parse is then done with orjson for speed and consistency.
    """

    cleaned = strip_markdown_fences(text)
    decoder = json.JSONDecoder()
    for index, character in enumerate(cleaned):
        if character not in "{[":
            continue
        try:
            _, end = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        candidate = cleaned[index : index + end]
        try:
            return orjson.loads(candidate)
        except orjson.JSONDecodeError:
            continue
    try:
        return orjson.loads(cleaned)
    except orjson.JSONDecodeError as exc:
        raise ValueError("Could not extract a valid JSON payload from model output") from exc


def extract_plan_list(payload: Any) -> list[dict[str, Any]]:
    """Extract a list of plan dictionaries from a parsed JSON payload."""

    if isinstance(payload, dict):
        plans = payload.get("plans")
        if isinstance(plans, list):
            return [item for item in plans if isinstance(item, dict)]
        if "strategy" in payload:
            return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError("Expected a plan list or {'plans': [...]} payload")


def extract_file_mapping(payload: Any) -> dict[str, str]:
    """Extract a filename -> content mapping from a parsed JSON payload."""

    if isinstance(payload, dict):
        if payload and all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
            return dict(payload)
        files_payload = payload.get("files")
        if isinstance(files_payload, dict) and all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in files_payload.items()
        ):
            return dict(files_payload)
        if isinstance(files_payload, list):
            mapping: dict[str, str] = {}
            for item in files_payload:
                if not isinstance(item, dict):
                    continue
                path = item.get("path") or item.get("filename")
                content = item.get("content")
                if isinstance(path, str) and isinstance(content, str):
                    mapping[path] = content
            if mapping:
                return mapping
    raise ValueError("Expected a JSON object mapping filenames to full file contents")


def extract_edit_operations(payload: Any) -> list[dict[str, str]]:
    """Extract a list of textual edit operations from parsed JSON."""

    raw_edits: Any
    if isinstance(payload, dict):
        raw_edits = payload.get("edits")
    elif isinstance(payload, list):
        raw_edits = payload
    else:
        raise ValueError("Expected an edit list or {'edits': [...]} payload")

    if not isinstance(raw_edits, list):
        raise ValueError("Expected 'edits' to be a list")

    edits: list[dict[str, str]] = []
    for item in raw_edits:
        if not isinstance(item, dict):
            continue
        file_name = item.get("file") or item.get("path")
        old_snippet = item.get("old_snippet")
        new_snippet = item.get("new_snippet")
        if all(isinstance(value, str) for value in (file_name, old_snippet, new_snippet)):
            edits.append(
                {
                    "file": file_name,
                    "old_snippet": old_snippet,
                    "new_snippet": new_snippet,
                }
            )
    if not edits:
        raise ValueError("Expected at least one valid edit operation")
    return edits

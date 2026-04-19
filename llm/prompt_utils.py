"""Task loading and prompt-construction helpers for the DSL loop."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Mapping
import warnings

import orjson
from pydantic import BaseModel, ConfigDict

from env.dsl_schema import PlanDSL


class TaskContext(BaseModel):
    """Loaded task directory with prompt-friendly views."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    task_dir: Path
    task_json: dict[str, Any]
    prompt: str
    language: str
    target_files: list[str]
    metadata: dict[str, Any]
    files: dict[str, str]
    editable_files: dict[str, str]
    visible_test_file: str | None = None
    hidden_test_file: str | None = None
    family: str
    task_id: str
    entrypoint: str | None = None


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def truncate_file_content(content: str, *, max_chars: int = 2500) -> str:
    """Trim file content conservatively for prompt budgets."""

    if len(content) <= max_chars:
        return content
    return content[: max_chars - 80] + "\n... [truncated for prompt budget] ...\n"


def load_task_context(task_dir: Path) -> TaskContext:
    """Load task metadata and editable source files from disk."""

    task_dir = task_dir.resolve()
    task_json = orjson.loads((task_dir / "task.json").read_bytes())

    files: dict[str, str] = {}
    for path in sorted(task_dir.rglob("*")):
        if not path.is_file():
            continue
        if "reference" in path.parts:
            continue
        if "__pycache__" in path.parts:
            continue
        relative_path = path.relative_to(task_dir).as_posix()
        if relative_path == "task.json":
            continue
        if path.suffix == ".pyc":
            continue
        try:
            files[relative_path] = _read_text(path)
        except UnicodeDecodeError:
            warnings.warn(
                f"Skipping non-UTF8 task artifact while loading {task_dir.name}: {relative_path}",
                stacklevel=2,
            )

    raw_target_files = task_json.get("target_files")
    if not isinstance(raw_target_files, list):
        raw_target_files = task_json.get("metadata", {}).get("target_files", [])
    target_files = [str(item) for item in raw_target_files if isinstance(item, str)]
    editable_files = {
        path: files[path]
        for path in target_files
        if path in files and not path.startswith("test_")
    }
    if not editable_files:
        editable_files = {path: content for path, content in files.items() if not path.startswith("test_")}

    return TaskContext.model_validate(
        {
            "task_dir": task_dir,
            "task_json": task_json,
            "prompt": str(task_json.get("prompt", "")),
            "language": str(task_json.get("language", "python")),
            "target_files": target_files,
            "metadata": dict(task_json.get("metadata", {})),
            "files": files,
            "editable_files": editable_files,
            "visible_test_file": task_json.get("visible_test_file"),
            "hidden_test_file": task_json.get("hidden_test_file"),
            "family": str(task_json.get("family", "")),
            "task_id": str(task_json.get("task_id", task_dir.name)),
            "entrypoint": task_json.get("entrypoint"),
        }
    )


def _render_file_sections(files: Mapping[str, str], *, max_chars: int = 2500) -> str:
    sections: list[str] = []
    for path, content in files.items():
        sections.append(
            "\n".join(
                [
                    f"<file path=\"{path}\">",
                    truncate_file_content(content, max_chars=max_chars),
                    "</file>",
                ]
            )
        )
    return "\n\n".join(sections)


def _render_exact_file_sections(files: Mapping[str, str]) -> str:
    """Render full file contents without truncation."""

    sections: list[str] = []
    for path, content in files.items():
        sections.append(
            "\n".join(
                [
                    f"<file path=\"{path}\">",
                    content,
                    "</file>",
                ]
            )
        )
    return "\n\n".join(sections)


def _json_text(payload: Any) -> str:
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode("utf-8")


def render_compact_task_context(task: TaskContext, *, include_visible_test: bool = True) -> str:
    """Render a compact task summary for prompts and logs."""

    lines = [
        f"task_id: {task.task_id}",
        f"family: {task.family}",
        f"language: {task.language}",
        f"target_files: {', '.join(task.target_files)}",
        f"prompt: {task.prompt}",
    ]
    if task.metadata:
        lines.append(f"metadata: {_json_text(task.metadata)}")
    editable_sections = _render_file_sections(task.editable_files)
    if editable_sections:
        lines.append("editable_files:")
        lines.append(editable_sections)
    if include_visible_test and task.visible_test_file and task.visible_test_file in task.files:
        lines.append("visible_test:")
        lines.append(
            _render_file_sections(
                {task.visible_test_file: task.files[task.visible_test_file]},
                max_chars=2200,
            )
        )
    return "\n".join(lines)


def build_proposal_prompt(task: TaskContext, *, k: int) -> str:
    """Build the planner prompt requesting structured DSL branches."""

    return "\n".join(
        [
            "You are a search planner.",
            "Return compact JSON only.",
            "No code. No explanations. Be extremely concise.",
            "",
            f"Return one object with key \"plans\" containing up to {k} plans.",
            "Each plan must contain:",
            "strategy, target_files, suspected_bug_types, invariants, subgoals, validation_checks, risks, touched_symbols, edit_style, confidence, notes",
            "",
            "Rules:",
            "- Use enum-like tags whenever possible.",
            "- Keep list fields to at most 3 items.",
            "- Keep each item short, 2-6 words.",
            "- Keep notes under 8 words.",
            "- Only target editable source files.",
            "",
            render_compact_task_context(task, include_visible_test=False),
        ]
    )


def build_compiler_prompt(task: TaskContext, plan: PlanDSL) -> str:
    """Build the code-realization prompt for one selected DSL plan."""

    relevant_paths = [path for path in plan.target_files if path in task.files]
    relevant_files = {path: task.files[path] for path in relevant_paths}

    return "\n".join(
        [
            "You are a constrained code realizer.",
            "Implement the selected DSL plan exactly.",
            "Be as concise as possible.",
            "Output ONLY one strict JSON object mapping filenames to full updated file contents.",
            "Do not output Markdown. Do not output explanations. Do not output diffs. Do not ask to inspect more files.",
            "If only one target file is listed, return exactly one key for that file.",
            "Only write files that already exist in the task and stay within the allowed target files.",
            "Preserve behavior outside the targeted fix where possible.",
            "",
            "Selected plan (JSON):",
            _json_text(plan.to_dict()),
            "",
            "Selected plan (compact):",
            plan.to_compact_text(),
            "",
            "Task context:",
            f"task_id: {task.task_id}",
            f"family: {task.family}",
            f"prompt: {task.prompt}",
            f"target_files: {', '.join(task.target_files)}",
            "",
            "Exact editable file contents:",
            _render_exact_file_sections(relevant_files),
            "",
            "Return exactly this JSON shape:",
            '{"target.py": "full updated file contents here"}',
            "",
            "Begin JSON now:",
        ]
    )


def _slice_source_by_lines(source: str, start_line: int, end_line: int) -> str:
    lines = source.splitlines()
    return "\n".join(lines[start_line - 1 : end_line]) + "\n"


def extract_symbol_snippets(source: str, touched_symbols: list[str]) -> list[tuple[str, str]]:
    """Extract exact source snippets for touched top-level symbols and methods."""

    if not touched_symbols:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    wanted = set(touched_symbols)
    snippets: list[tuple[str, str]] = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            if hasattr(node, "end_lineno") and node.end_lineno is not None:
                snippets.append((node.name, _slice_source_by_lines(source, node.lineno, node.end_lineno)))
        elif isinstance(node, ast.ClassDef):
            if node.name in wanted and hasattr(node, "end_lineno") and node.end_lineno is not None:
                snippets.append((node.name, _slice_source_by_lines(source, node.lineno, node.end_lineno)))
                continue
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name in wanted:
                    if hasattr(child, "end_lineno") and child.end_lineno is not None:
                        snippets.append(
                            (f"{node.name}.{child.name}", _slice_source_by_lines(source, child.lineno, child.end_lineno))
                        )
    return snippets


def build_edit_compiler_prompt(task: TaskContext, plan: PlanDSL) -> str:
    """Build a compact compiler prompt asking for targeted search/replace edits."""

    relevant_paths = [path for path in plan.target_files if path in task.files]
    snippet_sections: list[str] = []
    fallback_files: dict[str, str] = {}
    for path in relevant_paths:
        source = task.files[path]
        snippets = extract_symbol_snippets(source, list(plan.touched_symbols))
        if snippets:
            snippet_text = "\n\n".join(
                f"<snippet file=\"{path}\" symbol=\"{symbol}\">\n{snippet}</snippet>"
                for symbol, snippet in snippets
            )
            snippet_sections.append(snippet_text)
        else:
            fallback_files[path] = truncate_file_content(source, max_chars=2500)

    parts = [
        "You are a constrained code editor.",
        "Be as concise as possible.",
        "Return ONLY strict JSON with an 'edits' array.",
        "Each edit must be: {\"file\": \"...\", \"old_snippet\": \"exact existing text\", \"new_snippet\": \"replacement text\"}.",
        "Use exact old_snippet text copied from the provided snippets so replacements can be applied mechanically.",
        "Do not output full files. Do not output markdown. Do not output explanations.",
        "Only edit allowed target files.",
        "",
        "Selected plan (JSON):",
        _json_text(plan.to_dict()),
        "",
        "Task:",
        f"task_id: {task.task_id}",
        f"family: {task.family}",
        f"prompt: {task.prompt}",
        f"target_files: {', '.join(task.target_files)}",
    ]
    if snippet_sections:
        parts.extend(["", "Exact editable snippets:", "\n\n".join(snippet_sections)])
    if fallback_files:
        parts.extend(["", "Fallback file excerpts:", _render_file_sections(fallback_files, max_chars=2500)])
    parts.extend(
        [
            "",
            "Return exactly this shape:",
            '{"edits": [{"file": "target.py", "old_snippet": "old text", "new_snippet": "new text"}]}',
            "",
            "Begin JSON now:",
        ]
    )
    return "\n".join(parts)

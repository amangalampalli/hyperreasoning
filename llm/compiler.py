"""Compile a selected DSL plan into concrete file contents."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import textwrap
import re

from env.dsl_schema import PlanDSL
from llm.llm_client import LocalLLMClient, LocalLLMError
from llm.parsing import extract_edit_operations, extract_file_mapping, extract_first_json_value
from llm.prompt_utils import TaskContext, build_compiler_prompt, build_edit_compiler_prompt


class CompilePlanError(RuntimeError):
    """Raised when a plan cannot be compiled into valid file outputs."""

    def __init__(
        self,
        message: str,
        *,
        raw_prompt: str | None = None,
        raw_response: str | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_prompt = raw_prompt
        self.raw_response = raw_response


def _validate_compiled_mapping(task: TaskContext, compiled: dict[str, str]) -> dict[str, str]:
    allowed_files = set(task.target_files) | set(task.editable_files)
    if not compiled:
        raise CompilePlanError("Compiler returned an empty file mapping")
    invalid_files = [path for path in compiled if path not in allowed_files]
    if invalid_files:
        raise CompilePlanError(f"Compiler touched invalid files: {invalid_files}")
    if not any(path in task.target_files for path in compiled):
        raise CompilePlanError("Compiler output did not update any target file")
    for path, content in compiled.items():
        if not isinstance(content, str):
            raise CompilePlanError(f"Compiled content for {path!r} is not a string")
    return compiled


def _try_exact_unique_match(original: str, snippet: str) -> tuple[int, int] | None:
    if not snippet:
        return None
    start = original.find(snippet)
    if start == -1:
        return None
    if original.find(snippet, start + 1) != -1:
        return None
    return start, start + len(snippet)


def _normalize_line_for_match(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


def _find_whitespace_tolerant_block(original: str, snippet: str) -> tuple[int, int] | None:
    snippet_lines = [line for line in snippet.strip("\n").splitlines() if line.strip()]
    if not snippet_lines:
        return None

    original_lines = original.splitlines(keepends=True)
    normalized_snippet = [_normalize_line_for_match(line) for line in snippet_lines]
    matches: list[tuple[int, int]] = []

    for start_idx in range(len(original_lines) - len(snippet_lines) + 1):
        window = original_lines[start_idx : start_idx + len(snippet_lines)]
        normalized_window = [_normalize_line_for_match(line.rstrip("\r\n")) for line in window]
        if normalized_window == normalized_snippet:
            start_char = sum(len(line) for line in original_lines[:start_idx])
            end_char = start_char + sum(len(line) for line in window)
            matches.append((start_char, end_char))

    if len(matches) == 1:
        return matches[0]
    return None


def _strip_whitespace_with_index(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    for index, character in enumerate(text):
        if character.isspace():
            continue
        normalized_chars.append(character)
        index_map.append(index)
    return "".join(normalized_chars), index_map


def _find_whitespace_insensitive_span(original: str, snippet: str) -> tuple[int, int] | None:
    normalized_snippet, _ = _strip_whitespace_with_index(snippet)
    if not normalized_snippet:
        return None
    normalized_original, index_map = _strip_whitespace_with_index(original)
    first_match = normalized_original.find(normalized_snippet)
    if first_match == -1:
        return None
    if normalized_original.find(normalized_snippet, first_match + 1) != -1:
        return None
    start_char = index_map[first_match]
    end_char = index_map[first_match + len(normalized_snippet) - 1] + 1
    return start_char, end_char


def _find_replacement_span(original: str, old_snippet: str) -> tuple[int, int] | None:
    candidates: list[str] = []
    seen: set[str] = set()
    for candidate in (
        old_snippet,
        old_snippet.replace("\r\n", "\n"),
        old_snippet.strip(),
        textwrap.dedent(old_snippet).strip(),
    ):
        if candidate and candidate not in seen:
            candidates.append(candidate)
            seen.add(candidate)

    for candidate in candidates:
        match = _try_exact_unique_match(original, candidate)
        if match is not None:
            return match

    for candidate in candidates:
        match = _find_whitespace_tolerant_block(original, candidate)
        if match is not None:
            return match

    for candidate in candidates:
        match = _find_whitespace_insensitive_span(original, candidate)
        if match is not None:
            return match
    return None


def _apply_edit_operations(task: TaskContext, edits: list[dict[str, str]]) -> dict[str, str]:
    """Apply exact textual replacements to target files."""

    allowed_files = set(task.target_files) | set(task.editable_files)
    working_files = {path: content for path, content in task.files.items()}
    touched_files: set[str] = set()
    noop_files: set[str] = set()

    for edit in edits:
        file_name = edit["file"]
        if file_name not in allowed_files:
            raise CompilePlanError(f"Edit touched invalid file: {file_name}")
        original = working_files.get(file_name)
        if original is None:
            raise CompilePlanError(f"Edit referenced unknown file: {file_name}")
        old_snippet = edit["old_snippet"]
        new_snippet = edit["new_snippet"]
        if old_snippet == new_snippet:
            noop_files.add(file_name)
            continue
        exact_occurrences = original.count(old_snippet)
        if exact_occurrences == 1:
            working_files[file_name] = original.replace(old_snippet, new_snippet, 1)
            touched_files.add(file_name)
            continue

        span = _find_replacement_span(original, old_snippet)
        if span is None:
            raise CompilePlanError(
                f"Edit old_snippet must match exactly once in {file_name}, matched {exact_occurrences} times"
            )
        start, end = span
        working_files[file_name] = original[:start] + new_snippet + original[end:]
        touched_files.add(file_name)

    if not touched_files and noop_files:
        compiled = {path: working_files[path] for path in sorted(noop_files)}
    else:
        compiled = {path: working_files[path] for path in sorted(touched_files)}
    return _validate_compiled_mapping(task, compiled)


def _repair_compiler_output(
    llm_client: LocalLLMClient,
    *,
    raw_output: str,
    target_files: list[str],
    want_edits: bool,
) -> str:
    """Ask the model to convert a bad compiler answer into strict JSON only."""

    example_file = target_files[0] if target_files else "target.py"
    repair_prompt = "\n".join(
        [
            "Convert the following model output into strict JSON only.",
            f"Allowed filenames: {', '.join(target_files)}",
            "",
            "Bad output:",
            raw_output,
            "",
            "Return exactly this shape:"
            if not want_edits
            else "Return exactly this shape:",
            f'{{"{example_file}": "full updated file contents here"}}'
            if not want_edits
            else f'{{"edits": [{{"file": "{example_file}", "old_snippet": "old text", "new_snippet": "new text"}}]}}',
        ]
    )
    return llm_client.complete(
        repair_prompt,
        temperature=0.0,
        max_tokens=900,
        mode_override="chat",
        extra_payload={"response_format": {"type": "json_object"}},
        request_label="repair_edits" if want_edits else "repair_files",
    )


def _extract_code_fence(text: str) -> str | None:
    match = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip() + "\n"
    return None


def _looks_like_source_code(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 20:
        return False
    code_markers = (
        "def ",
        "class ",
        "import ",
        "from ",
        "return ",
        "if ",
        "for ",
        "while ",
        "async ",
        "with ",
    )
    return any(marker in stripped for marker in code_markers)


def _recover_single_target_mapping(task: TaskContext, payload: Any | None, raw_output: str) -> dict[str, str] | None:
    if len(task.target_files) != 1:
        return None
    target_file = task.target_files[0]

    if isinstance(payload, dict):
        candidate_keys = ("code", "content", "output", "source", "text")
        for key in candidate_keys:
            value = payload.get(key)
            if isinstance(value, str) and _looks_like_source_code(value):
                return {target_file: value if value.endswith("\n") else value + "\n"}

        if len(payload) == 1:
            [(only_key, only_value)] = list(payload.items())
            if isinstance(only_value, str) and _looks_like_source_code(only_value):
                return {target_file: only_value if only_value.endswith("\n") else only_value + "\n"}

    fenced = _extract_code_fence(raw_output)
    if fenced and _looks_like_source_code(fenced):
        return {target_file: fenced}

    stripped = raw_output.strip()
    if _looks_like_source_code(stripped):
        return {target_file: stripped + ("\n" if not stripped.endswith("\n") else "")}
    return None


def _coerce_compiled_mapping(task: TaskContext, compiled: dict[str, str]) -> dict[str, str]:
    allowed_files = set(task.target_files) | set(task.editable_files)
    invalid_files = [path for path in compiled if path not in allowed_files]
    if len(task.target_files) == 1 and len(compiled) == 1 and len(invalid_files) == 1:
        only_invalid = invalid_files[0]
        only_content = compiled[only_invalid]
        if isinstance(only_content, str) and _looks_like_source_code(only_content):
            return {task.target_files[0]: only_content}
    return compiled


def _compile_via_edit_mode(
    task: TaskContext,
    plan: PlanDSL,
    *,
    llm_client: LocalLLMClient,
    temperature: float,
    max_tokens: int | None = 700,
) -> tuple[dict[str, str], str, str]:
    prompt = build_edit_compiler_prompt(task, plan)
    raw_output = llm_client.complete(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        mode_override="chat",
        extra_payload={"response_format": {"type": "json_object"}},
        request_label="compile_edits",
    )
    try:
        try:
            payload = extract_first_json_value(raw_output)
            edits = extract_edit_operations(payload)
        except ValueError:
            repaired_output = _repair_compiler_output(
                llm_client,
                raw_output=raw_output,
                target_files=list(plan.target_files),
                want_edits=True,
            )
            payload = extract_first_json_value(repaired_output)
            edits = extract_edit_operations(payload)
            raw_output = repaired_output
        compiled = _apply_edit_operations(task, edits)
    except (ValueError, CompilePlanError) as exc:
        raise CompilePlanError(
            f"Could not parse edit-mode compiler output: {exc}",
            raw_prompt=prompt,
            raw_response=raw_output,
        ) from exc
    return compiled, prompt, raw_output


def compile_plan_to_code(
    task: TaskContext,
    plan: PlanDSL,
    temperature: float = 0.2,
    *,
    client: LocalLLMClient | None = None,
    return_debug: bool = False,
    allow_full_file_fallback: bool = False,
    max_tokens: int | None = 700,
) -> dict[str, str] | dict[str, object]:
    """Compile one plan into complete updated file contents."""

    llm_client = client or LocalLLMClient()
    try:
        compiled, prompt, raw_output = _compile_via_edit_mode(
            task,
            plan,
            llm_client=llm_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not return_debug:
            return compiled
        return {
            "compiled_files": compiled,
            "raw_prompt": prompt,
            "raw_response": raw_output,
        }
    except (CompilePlanError, LocalLLMError) as edit_exc:
        if not allow_full_file_fallback:
            if isinstance(edit_exc, CompilePlanError):
                raise edit_exc
            raise CompilePlanError(f"Compiler edit-mode request failed: {edit_exc}") from edit_exc
        prompt = build_compiler_prompt(task, plan)
        raw_output: str | None = None
        try:
            raw_output = llm_client.complete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else None,
                mode_override="completion",
                request_label="compile_files",
            )
        except LocalLLMError:
            raise CompilePlanError(
                f"Compiler LLM request failed: {edit_exc}",
                raw_prompt=prompt,
                raw_response=raw_output,
            ) from edit_exc
    try:
        payload = extract_first_json_value(raw_output)
        try:
            compiled = extract_file_mapping(payload)
        except ValueError:
            recovered = _recover_single_target_mapping(task, payload, raw_output)
            if recovered is None:
                raise
            compiled = recovered
    except ValueError as exc:
        recovered = _recover_single_target_mapping(task, None, raw_output)
        if recovered is not None:
            compiled = recovered
            compiled = _coerce_compiled_mapping(task, compiled)
            compiled = _validate_compiled_mapping(task, compiled)
            if not return_debug:
                return compiled
            return {
                "compiled_files": compiled,
                "raw_prompt": prompt,
                "raw_response": raw_output,
            }
        try:
            repaired_output = _repair_compiler_output(
                llm_client,
                raw_output=raw_output,
                target_files=list(plan.target_files),
                want_edits=False,
            )
            try:
                payload = extract_first_json_value(repaired_output)
                try:
                    compiled = extract_file_mapping(payload)
                except ValueError:
                    recovered = _recover_single_target_mapping(task, payload, repaired_output)
                    if recovered is None:
                        raise
                    compiled = recovered
            except ValueError:
                recovered = _recover_single_target_mapping(task, None, repaired_output)
                if recovered is None:
                    raise
                compiled = recovered
            raw_output = repaired_output
        except (LocalLLMError, ValueError) as repair_exc:
            raise CompilePlanError(
                f"Could not parse compiler JSON output: {exc}; repair attempt failed: {repair_exc}",
                raw_prompt=prompt,
                raw_response=raw_output,
            ) from repair_exc
    compiled = _coerce_compiled_mapping(task, compiled)
    compiled = _validate_compiled_mapping(task, compiled)
    if not return_debug:
        return compiled
    return {
        "compiled_files": compiled,
        "raw_prompt": prompt,
        "raw_response": raw_output,
    }


def apply_compiled_files(
    task: TaskContext,
    compiled: dict[str, str],
    workspace_dir: Path | None = None,
) -> Path:
    """Copy a task into a workspace and overwrite compiled files."""

    _validate_compiled_mapping(task, compiled)
    if workspace_dir is None:
        destination = Path(tempfile.mkdtemp(prefix=f"dsl_{task.task_id}_"))
    else:
        destination = workspace_dir / f"{task.task_id}_workspace"
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=True)

    shutil.copytree(task.task_dir, destination, dirs_exist_ok=True)
    for relative_path, content in compiled.items():
        target_path = destination / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content, encoding="utf-8")
    return destination

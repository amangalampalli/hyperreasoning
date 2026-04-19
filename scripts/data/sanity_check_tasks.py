#!/usr/bin/env python3
"""Validate generated tasks and optionally execute reference implementations."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import which

ROOT = Path(__file__).resolve().parents[2]
HYPERREASONING_PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/hyperreasoning/bin/python")


REQUIRED_TASK_JSON_KEYS = {
    "task_id",
    "family",
    "difficulty",
    "language",
    "prompt",
    "target_files",
    "entrypoint",
    "visible_test_file",
    "hidden_test_file",
    "metadata",
}


@dataclass(slots=True)
class TaskCheckResult:
    """Per-task validation result."""

    task_dir: Path
    ok: bool
    messages: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT / "data/generated_tasks",
        help="Root directory containing generated task folders",
    )
    parser.add_argument(
        "--run-visible-tests",
        action="store_true",
        help="Execute visible tests against the reference implementation",
    )
    parser.add_argument(
        "--run-hidden-tests",
        action="store_true",
        help="Execute hidden tests against the reference implementation",
    )
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python interpreter to use when executing pytest checks",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Subprocess timeout per pytest invocation in seconds",
    )
    return parser.parse_args()


def interpreter_has_pytest(python_bin: str) -> bool:
    completed = subprocess.run(
        [python_bin, "-c", "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('pytest') else 1)"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def resolve_python_bin(raw_python_bin: str | None) -> str:
    if raw_python_bin:
        return raw_python_bin
    if interpreter_has_pytest(sys.executable):
        return sys.executable
    if HYPERREASONING_PYTHON.exists() and interpreter_has_pytest(str(HYPERREASONING_PYTHON)):
        return str(HYPERREASONING_PYTHON)
    conda_python = which("python")
    if conda_python and interpreter_has_pytest(conda_python):
        return conda_python
    raise RuntimeError(
        "Could not find a Python interpreter with pytest installed. "
        "Pass --python-bin explicitly, for example the hyperreasoning conda env."
    )


def iter_task_dirs(root: Path) -> list[Path]:
    task_dirs: list[Path] = []
    for difficulty_dir in sorted(root.glob("*")):
        if not difficulty_dir.is_dir():
            continue
        for task_dir in sorted(difficulty_dir.glob("*")):
            if task_dir.is_dir():
                task_dirs.append(task_dir)
    return task_dirs


def load_task_json(task_dir: Path) -> dict[str, object]:
    return json.loads((task_dir / "task.json").read_text(encoding="utf-8"))


def validate_task_dir(task_dir: Path) -> TaskCheckResult:
    messages: list[str] = []
    task_json_path = task_dir / "task.json"
    if not task_json_path.exists():
        return TaskCheckResult(task_dir=task_dir, ok=False, messages=["missing task.json"])
    try:
        payload = load_task_json(task_dir)
    except json.JSONDecodeError as exc:
        return TaskCheckResult(task_dir=task_dir, ok=False, messages=[f"invalid JSON: {exc}"])

    missing_keys = REQUIRED_TASK_JSON_KEYS - set(payload)
    if missing_keys:
        messages.append(f"task.json missing keys: {sorted(missing_keys)}")

    difficulty = payload.get("difficulty")
    if difficulty != task_dir.parent.name:
        messages.append(
            f"difficulty mismatch: task.json has {difficulty!r}, directory is {task_dir.parent.name!r}"
        )

    for file_key in ["entrypoint", "visible_test_file", "hidden_test_file"]:
        file_name = payload.get(file_key)
        if not isinstance(file_name, str) or not (task_dir / file_name).exists():
            messages.append(f"missing referenced file for {file_key}: {file_name!r}")

    target_files = payload.get("target_files")
    if not isinstance(target_files, list) or not target_files:
        messages.append("target_files must be a non-empty list")
    else:
        for relative_path in target_files:
            if not isinstance(relative_path, str) or not (task_dir / relative_path).exists():
                messages.append(f"missing target file: {relative_path!r}")
            if not (task_dir / "reference" / relative_path).exists():
                messages.append(f"missing reference file: {relative_path!r}")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        messages.append("prompt must be a non-empty string")

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        messages.append("metadata must be a JSON object")

    return TaskCheckResult(task_dir=task_dir, ok=not messages, messages=messages)


def run_reference_tests(
    task_dir: Path,
    task_payload: dict[str, object],
    *,
    python_bin: str,
    run_visible: bool,
    run_hidden: bool,
    timeout: float,
) -> list[str]:
    failures: list[str] = []
    test_files: list[str] = []
    if run_visible:
        test_files.append(str(task_payload["visible_test_file"]))
    if run_hidden:
        test_files.append(str(task_payload["hidden_test_file"]))
    if not test_files:
        return failures

    with tempfile.TemporaryDirectory(prefix="task_sanity_") as temp_dir:
        temp_path = Path(temp_dir)
        shutil.copytree(task_dir, temp_path, dirs_exist_ok=True)
        reference_dir = temp_path / "reference"
        for reference_file in reference_dir.rglob("*"):
            if reference_file.is_file():
                relative_path = reference_file.relative_to(reference_dir)
                destination = temp_path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(reference_file, destination)
        for test_file in test_files:
            module_name = Path(test_file).stem
            try:
                completed = subprocess.run(
                    [python_bin, "-m", "pytest", "-q", test_file],
                    cwd=temp_path,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                failures.append(f"{module_name}: timed out after {timeout}s")
                continue
            if completed.returncode != 0:
                stderr = completed.stderr.strip()
                stdout = completed.stdout.strip()
                excerpt = stderr or stdout or "<no output>"
                failures.append(f"{module_name}: failed with exit code {completed.returncode}: {excerpt}")
    return failures


def main() -> int:
    args = parse_args()
    python_bin = resolve_python_bin(args.python_bin)
    task_dirs = iter_task_dirs(args.root)
    if not task_dirs:
        print(f"No generated tasks found under {args.root}")
        return 1

    results: list[TaskCheckResult] = []
    for task_dir in task_dirs:
        result = validate_task_dir(task_dir)
        payload = None
        if result.ok and (args.run_visible_tests or args.run_hidden_tests):
            payload = load_task_json(task_dir)
            failures = run_reference_tests(
                task_dir,
                payload,
                python_bin=python_bin,
                run_visible=args.run_visible_tests,
                run_hidden=args.run_hidden_tests,
                timeout=args.timeout,
            )
            if failures:
                result.ok = False
                result.messages.extend(failures)
        results.append(result)

    failures = [result for result in results if not result.ok]
    print(f"Checked {len(results)} tasks")
    if not failures:
        print("All tasks passed sanity checks")
        return 0

    print(f"{len(failures)} task(s) failed sanity checks:")
    for result in failures:
        print(f"- {result.task_dir}")
        for message in result.messages:
            print(f"    {message}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

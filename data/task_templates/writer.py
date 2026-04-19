"""Disk writer for generated task instances."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import TaskSpec
from .utils import ensure_dir, render_json, write_text


def write_task(task: TaskSpec, output_root: Path) -> Path:
    """Write a task folder with metadata, visible files, and reference files."""

    task_dir = output_root / task.difficulty / task.task_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    ensure_dir(task_dir)
    write_text(task_dir / "task.json", render_json(task.to_task_json()))
    for relative_path, content in sorted(task.files.items()):
        write_text(task_dir / relative_path, content)
    reference_dir = task_dir / "reference"
    ensure_dir(reference_dir)
    for relative_path, content in sorted(task.reference_files.items()):
        write_text(reference_dir / relative_path, content)
    return task_dir

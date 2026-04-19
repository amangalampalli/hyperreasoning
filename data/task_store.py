"""Task discovery and sampling for DSL search training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

from llm.prompt_utils import TaskContext, load_task_context
from data.task_manifest import discover_task_dirs, filter_task_dirs, load_task_manifest


@dataclass(slots=True)
class TaskStore:
    """Resolved task set with convenience sampling helpers."""

    task_dirs: list[Path]

    @classmethod
    def from_tasks_root(
        cls,
        tasks_root: Path,
        *,
        families: set[str] | None = None,
        difficulties: set[str] | None = None,
        limit: int | None = None,
    ) -> "TaskStore":
        task_dirs = discover_task_dirs(tasks_root)
        task_dirs = filter_task_dirs(task_dirs, families=families, difficulties=difficulties)
        if limit is not None:
            task_dirs = task_dirs[:limit]
        return cls(task_dirs=task_dirs)

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        limit: int | None = None,
    ) -> "TaskStore":
        task_dirs = load_task_manifest(manifest_path)
        if limit is not None:
            task_dirs = task_dirs[:limit]
        return cls(task_dirs=task_dirs)

    def __len__(self) -> int:
        return len(self.task_dirs)

    def iter_contexts(self) -> Iterable[TaskContext]:
        for task_dir in self.task_dirs:
            yield load_task_context(task_dir)

    def sample(self, rng: random.Random) -> TaskContext:
        if not self.task_dirs:
            raise ValueError("TaskStore is empty")
        return load_task_context(rng.choice(self.task_dirs))

    def get(self, task: str | Path | TaskContext) -> TaskContext:
        if isinstance(task, TaskContext):
            return task
        if isinstance(task, Path):
            return load_task_context(task)
        for task_dir in self.task_dirs:
            if task_dir.name == task:
                return load_task_context(task_dir)
        raise KeyError(f"Unknown task identifier {task!r}")

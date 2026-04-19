"""Task template framework for procedurally generated coding tasks."""

from .base import TaskSpec, TaskTemplate
from .registry import get_template, list_families, registry
from .writer import write_task

__all__ = [
    "TaskSpec",
    "TaskTemplate",
    "get_template",
    "list_families",
    "registry",
    "write_task",
]

"""Core task template abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


VALID_DIFFICULTIES = {"medium", "hard"}


@dataclass(slots=True)
class TaskSpec:
    """Concrete, serializable task instance."""

    task_id: str
    family: str
    difficulty: str
    language: str
    prompt: str
    files: dict[str, str]
    metadata: dict[str, Any]
    reference_files: dict[str, str]
    entrypoint: str
    visible_test_file: str
    hidden_test_file: str
    target_files: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.difficulty not in VALID_DIFFICULTIES:
            raise ValueError(f"Unsupported difficulty: {self.difficulty}")
        if self.visible_test_file not in self.files:
            raise ValueError(f"Visible test file missing: {self.visible_test_file}")
        if self.hidden_test_file not in self.files:
            raise ValueError(f"Hidden test file missing: {self.hidden_test_file}")
        if self.entrypoint not in self.files:
            raise ValueError(f"Entrypoint missing: {self.entrypoint}")
        if not self.target_files:
            self.target_files = sorted(
                path
                for path in self.files
                if path not in {self.visible_test_file, self.hidden_test_file}
            )
        missing_references = [path for path in self.target_files if path not in self.reference_files]
        if missing_references:
            raise ValueError(
                f"Reference implementation missing target files: {', '.join(missing_references)}"
            )

    def to_task_json(self) -> dict[str, Any]:
        """Return the standardized on-disk metadata payload."""

        return {
            "task_id": self.task_id,
            "family": self.family,
            "difficulty": self.difficulty,
            "language": self.language,
            "prompt": self.prompt,
            "target_files": self.target_files,
            "entrypoint": self.entrypoint,
            "visible_test_file": self.visible_test_file,
            "hidden_test_file": self.hidden_test_file,
            "metadata": self.metadata,
        }


class TaskTemplate(ABC):
    """Base class for seed-driven task families."""

    family: str
    family_version: str = "1.0"
    language: str = "python"

    @abstractmethod
    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        """Create one deterministic task instance."""

    def _validate_difficulty(self, difficulty: str) -> None:
        if difficulty not in VALID_DIFFICULTIES:
            raise ValueError(
                f"{self.family} only supports difficulties {sorted(VALID_DIFFICULTIES)}; "
                f"got {difficulty!r}"
            )

    def build_spec(
        self,
        *,
        seed: int,
        difficulty: str,
        prompt: str,
        files: dict[str, str],
        entrypoint: str,
        visible_test_file: str,
        hidden_test_file: str,
        reference_files: dict[str, str],
        metadata: dict[str, Any],
        task_id: str,
    ) -> TaskSpec:
        """Small helper to normalize task metadata."""

        enriched_metadata = {
            "family_version": self.family_version,
            "seed": seed,
            **metadata,
        }
        return TaskSpec(
            task_id=task_id,
            family=self.family,
            difficulty=difficulty,
            language=self.language,
            prompt=prompt.strip(),
            files=files,
            metadata=enriched_metadata,
            reference_files=reference_files,
            entrypoint=entrypoint,
            visible_test_file=visible_test_file,
            hidden_test_file=hidden_test_file,
        )

"""Stateful iterator resume task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class StatefulIteratorResumeBugTemplate(TaskTemplate):
    """Generate iterator checkpoint/resume repair tasks."""

    family = "stateful_iterator_resume_bug"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        class_name = choose_variant(rng, ["FlattenedCheckpointIterator", "ResumableCursor", "CheckpointingIterator"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{class_name}` in `iterator_impl.py`. The iterator flattens a sequence of groups and "
                "supports checkpoints plus restore. It must resume from the exact next element, handle empty "
                "groups, preserve exhaustion semantics, and isolate its internal state from external mutation.",
                f"`iterator_impl.py` contains a resumable iterator over grouped items. Fix checkpoint and "
                "restore behavior so resuming never duplicates or skips values and terminal state remains stable.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations


            class {class_name}:
                def __init__(self, groups: list[list[int]]) -> None:
                    self._groups = [list(group) for group in groups]
                    self._outer_index = 0
                    self._inner_index = 0
                    self._exhausted = False

                def __iter__(self) -> "{class_name}":
                    return self

                def __next__(self) -> int:
                    if self._exhausted:
                        raise StopIteration
                    while self._outer_index < len(self._groups):
                        current_group = self._groups[self._outer_index]
                        if self._inner_index < len(current_group):
                            value = current_group[self._inner_index]
                            self._inner_index += 1
                            return value
                        self._outer_index += 1
                        self._inner_index = 0
                    self._exhausted = True
                    raise StopIteration

                def checkpoint(self) -> dict[str, object]:
                    return {{
                        "outer_index": self._outer_index,
                        "inner_index": self._inner_index,
                        "exhausted": self._exhausted,
                        "groups_snapshot": [list(group) for group in self._groups],
                    }}

                @classmethod
                def from_checkpoint(
                    cls,
                    groups: list[list[int]],
                    checkpoint: dict[str, object],
                ) -> "{class_name}":
                    snapshot = checkpoint.get("groups_snapshot", groups)
                    source_groups = snapshot if isinstance(snapshot, list) else groups
                    iterator = cls(source_groups)
                    iterator._outer_index = int(checkpoint["outer_index"])
                    iterator._inner_index = int(checkpoint["inner_index"])
                    iterator._exhausted = bool(checkpoint.get("exhausted", False))
                    return iterator
            """
        )

        if difficulty == "medium":
            buggy = reference.replace(
                '                    return {\n                        "outer_index": self._outer_index,\n                        "inner_index": self._inner_index,\n                        "exhausted": self._exhausted,\n                        "groups_snapshot": [list(group) for group in self._groups],\n                    }\n',
                '                    return {\n                        "outer_index": self._outer_index,\n                        "inner_index": 0,\n                        "exhausted": self._exhausted,\n                        "groups_snapshot": [list(group) for group in self._groups],\n                    }\n',
            )
            bug_types = ["checkpoint drops the intra-group cursor position"]
            strategy_traps = [
                "Resuming at the start of a group duplicates data even when outer indices look correct",
                "Iterator bugs show up after checkpoint/restore, not in a single uninterrupted pass",
            ]
        else:
            buggy = reference.replace(
                "                    self._groups = [list(group) for group in groups]\n",
                "                    self._groups = groups\n",
            ).replace(
                "                    snapshot = checkpoint.get(\"groups_snapshot\", groups)\n                    source_groups = snapshot if isinstance(snapshot, list) else groups\n                    iterator = cls(source_groups)\n",
                "                    iterator = cls(groups)\n",
            ).replace(
                '                    iterator._exhausted = bool(checkpoint.get("exhausted", False))\n',
                '                    iterator._exhausted = False\n',
            )
            bug_types = ["iterator state aliases caller-owned groups and restore ignores exhausted checkpoints"]
            strategy_traps = [
                "A fix that only stores the right indices still fails if restored terminal state is wrong",
                "Mutable input aliasing can corrupt resumed iterators after external list edits",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                from iterator_impl import {class_name}


                class IteratorVisibleTests(unittest.TestCase):
                    def test_resume_from_mid_group(self) -> None:
                        iterator = {class_name}([[1, 2, 3], [4]])
                        self.assertEqual(next(iterator), 1)
                        checkpoint = iterator.checkpoint()
                        resumed = {class_name}.from_checkpoint([[1, 2, 3], [4]], checkpoint)
                        self.assertEqual(list(resumed), [2, 3, 4])

                    def test_empty_groups_are_skipped(self) -> None:
                        iterator = {class_name}([[], [5], [], [6, 7]])
                        self.assertEqual(list(iterator), [5, 6, 7])
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from iterator_impl import {class_name}


                class IteratorHiddenTests(unittest.TestCase):
                    def test_exhausted_checkpoint_stays_exhausted(self) -> None:
                        iterator = {class_name}([[1]])
                        self.assertEqual(list(iterator), [1])
                        checkpoint = iterator.checkpoint()
                        resumed = {class_name}.from_checkpoint([[1]], checkpoint)
                        self.assertEqual(list(resumed), [])

                    def test_constructor_copies_nested_groups(self) -> None:
                        groups = [[10, 20], [30]]
                        iterator = {class_name}(groups)
                        checkpoint = iterator.checkpoint()
                        groups[0].append(999)
                        resumed = {class_name}.from_checkpoint(groups, checkpoint)
                        self.assertEqual(list(resumed), [10, 20, 30])
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "iterator_impl.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"iterator_impl.py": reference},
            entrypoint="iterator_impl.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["iterator_impl.py"],
                "expected_skill_tags": ["iterators", "checkpointing", "state", "resume-logic"],
                "niche_topic": "flattened iterator checkpoint recovery",
                "repairable": True,
            },
        )

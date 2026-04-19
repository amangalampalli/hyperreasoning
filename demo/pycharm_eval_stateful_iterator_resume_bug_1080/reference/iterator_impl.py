from __future__ import annotations


class CheckpointingIterator:
    def __init__(self, groups: list[list[int]]) -> None:
        self._groups = [list(group) for group in groups]
        self._outer_index = 0
        self._inner_index = 0
        self._exhausted = False

    def __iter__(self) -> "CheckpointingIterator":
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
        return {
            "outer_index": self._outer_index,
            "inner_index": self._inner_index,
            "exhausted": self._exhausted,
            "groups_snapshot": [list(group) for group in self._groups],
        }

    @classmethod
    def from_checkpoint(
        cls,
        groups: list[list[int]],
        checkpoint: dict[str, object],
    ) -> "CheckpointingIterator":
        snapshot = checkpoint.get("groups_snapshot", groups)
        source_groups = snapshot if isinstance(snapshot, list) else groups
        iterator = cls(source_groups)
        iterator._outer_index = int(checkpoint["outer_index"])
        iterator._inner_index = int(checkpoint["inner_index"])
        iterator._exhausted = bool(checkpoint.get("exhausted", False))
        return iterator

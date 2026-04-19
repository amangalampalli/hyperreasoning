from __future__ import annotations

from collections import defaultdict
from typing import Callable


class DependencyCache:
    def __init__(self) -> None:
        self._sources: dict[str, object] = {}
        self._rules: dict[str, tuple[tuple[str, ...], Callable[..., object]]] = {}
        self._cache: dict[str, object] = {}
        self._reverse: dict[str, set[str]] = defaultdict(set)

    def set_source(self, name: str, value: object) -> None:
        self._sources[name] = value
        self.invalidate(name)

    def set_derived(
        self,
        name: str,
        dependencies: list[str] | tuple[str, ...],
        compute: Callable[..., object],
    ) -> None:
        old_rule = self._rules.get(name)
        if old_rule is not None:
            old_dependencies, _ = old_rule
            for dependency in old_dependencies:
                self._reverse[dependency].discard(name)
        deps_tuple = tuple(dependencies)
        self._rules[name] = (deps_tuple, compute)
        for dependency in deps_tuple:
            self._reverse[dependency].add(name)
        self.invalidate(name)

    def invalidate(self, name: str) -> None:
        queue = [name]
        seen: set[str] = set()
        while queue:
            current = queue.pop()
            self._cache.pop(current, None)
            for dependent in self._reverse.get(current, set()):
                if dependent not in seen:
                    seen.add(dependent)
                    queue.append(dependent)

    def get(self, name: str) -> object:
        if name in self._cache:
            return self._cache[name]
        if name in self._sources:
            return self._sources[name]
        if name not in self._rules:
            raise KeyError(name)
        dependencies, compute = self._rules[name]
        value = compute(*(self.get(dependency) for dependency in dependencies))
        self._cache[name] = value
        return value

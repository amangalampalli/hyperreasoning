"""Dependency cache invalidation task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class CacheInvalidationDependencyTemplate(TaskTemplate):
    """Generate incremental dependency cache tasks."""

    family = "cache_invalidation_dependency"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        class_name = choose_variant(rng, ["DependencyCache", "IncrementalCache", "DerivedValueCache"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{class_name}` in `cache.py`. The cache tracks source values and derived nodes with "
                "dependencies. Updates must invalidate exactly the right downstream nodes, including transitive "
                "dependents, while keeping the reverse dependency graph consistent when rules change.",
                f"`cache.py` contains an incremental dependency cache. Fix invalidation and graph maintenance so "
                "source updates and rule rewrites leave no stale cached values or stale reverse edges.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            from collections import defaultdict
            from typing import Callable


            class {class_name}:
                def __init__(self) -> None:
                    self._sources: dict[str, object] = {{}}
                    self._rules: dict[str, tuple[tuple[str, ...], Callable[..., object]]] = {{}}
                    self._cache: dict[str, object] = {{}}
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
            """
        )

        if difficulty == "medium":
            buggy = reference.replace(
                "                def invalidate(self, name: str) -> None:\n                    queue = [name]\n                    seen: set[str] = set()\n                    while queue:\n                        current = queue.pop()\n                        self._cache.pop(current, None)\n                        for dependent in self._reverse.get(current, set()):\n                            if dependent not in seen:\n                                seen.add(dependent)\n                                queue.append(dependent)\n",
                "                def invalidate(self, name: str) -> None:\n                    self._cache.pop(name, None)\n                    for dependent in self._reverse.get(name, set()):\n                        self._cache.pop(dependent, None)\n",
            )
            bug_types = ["only direct dependents are invalidated, leaving transitive cache entries stale"]
            strategy_traps = [
                "A direct-dependency fix still fails if invalidation does not walk the full downstream graph",
                "Caching behavior must remain precise rather than clearing everything globally",
            ]
        else:
            buggy = reference.replace(
                "                    if old_rule is not None:\n                        old_dependencies, _ = old_rule\n                        for dependency in old_dependencies:\n                            self._reverse[dependency].discard(name)\n",
                "",
            )
            bug_types = ["rule rewrites leave stale reverse dependencies behind"]
            strategy_traps = [
                "Transitive invalidation can look correct while stale reverse edges still trigger the wrong recomputations",
                "Removing all cache entries globally hides the correctness bug but breaks incremental behavior",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                from cache import {class_name}


                class DependencyCacheVisibleTests(unittest.TestCase):
                    def test_transitive_invalidation_recomputes_chain(self) -> None:
                        cache = {class_name}()
                        trace: list[str] = []
                        cache.set_source("price", 10)
                        cache.set_source("tax", 2)
                        cache.set_derived("subtotal", ["price", "tax"], lambda price, tax: trace.append("subtotal") or price + tax)
                        cache.set_derived("grand", ["subtotal"], lambda subtotal: trace.append("grand") or subtotal * 2)

                        self.assertEqual(cache.get("grand"), 24)
                        self.assertEqual(cache.get("grand"), 24)
                        self.assertEqual(trace, ["subtotal", "grand"])

                        cache.set_source("price", 20)
                        self.assertEqual(cache.get("grand"), 44)
                        self.assertEqual(trace, ["subtotal", "grand", "subtotal", "grand"])
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from cache import {class_name}


                class DependencyCacheHiddenTests(unittest.TestCase):
                    def test_rewriting_rule_removes_old_reverse_edges(self) -> None:
                        cache = {class_name}()
                        calls: list[str] = []
                        cache.set_source("a", 1)
                        cache.set_source("b", 10)

                        def compute_from_a(a: int) -> int:
                            calls.append("from_a")
                            return a + 100

                        def compute_from_b(b: int) -> int:
                            calls.append("from_b")
                            return b + 1000

                        cache.set_derived("derived", ["a"], compute_from_a)
                        self.assertEqual(cache.get("derived"), 101)
                        cache.set_derived("derived", ["b"], compute_from_b)
                        self.assertEqual(cache.get("derived"), 1010)
                        cache.set_source("a", 2)
                        self.assertEqual(cache.get("derived"), 1010)
                        self.assertEqual(calls.count("from_b"), 1)

                    def test_exact_invalidation_keeps_unrelated_entries_cached(self) -> None:
                        cache = {class_name}()
                        hits: list[str] = []
                        cache.set_source("left", 3)
                        cache.set_source("right", 4)
                        cache.set_derived("x", ["left"], lambda left: hits.append("x") or left * 2)
                        cache.set_derived("y", ["right"], lambda right: hits.append("y") or right * 3)
                        self.assertEqual(cache.get("x"), 6)
                        self.assertEqual(cache.get("y"), 12)
                        cache.set_source("left", 5)
                        self.assertEqual(cache.get("x"), 10)
                        self.assertEqual(cache.get("y"), 12)
                        self.assertEqual(hits, ["x", "y", "x"])
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "cache.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"cache.py": reference},
            entrypoint="cache.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["cache.py"],
                "expected_skill_tags": ["graphs", "cache-invalidation", "incremental", "state-management"],
                "niche_topic": "dependency-directed cache invalidation",
                "repairable": True,
            },
        )

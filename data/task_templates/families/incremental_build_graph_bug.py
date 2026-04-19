"""Incremental build graph task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class IncrementalBuildGraphBugTemplate(TaskTemplate):
    """Generate rebuild planner tasks with graph drift bugs."""

    family = "incremental_build_graph_bug"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        class_name = choose_variant(rng, ["BuildGraph", "RebuildPlanner", "IncrementalBuildGraph"])
        error_name = choose_variant(rng, ["BuildGraphError", "GraphConsistencyError"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{class_name}` in `build_graph.py`. It should compute the minimal rebuild set after "
                "source changes, return those nodes in dependency order, and keep reverse edges correct when "
                "node definitions are rewritten.",
                f"`build_graph.py` contains an incremental rebuild planner. Fix affected-node selection, "
                "topological ordering, and cycle handling so changed inputs rebuild exactly the right targets.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            from collections import defaultdict, deque


            class {error_name}(ValueError):
                pass


            class {class_name}:
                def __init__(self) -> None:
                    self._deps: dict[str, tuple[str, ...]] = {{}}
                    self._reverse: dict[str, set[str]] = defaultdict(set)

                def set_node(self, name: str, dependencies: list[str] | tuple[str, ...]) -> None:
                    old_dependencies = self._deps.get(name, ())
                    for dependency in old_dependencies:
                        self._reverse[dependency].discard(name)
                    deps_tuple = tuple(dependencies)
                    self._deps[name] = deps_tuple
                    for dependency in deps_tuple:
                        self._reverse[dependency].add(name)

                def affected_nodes(self, changed: set[str] | list[str] | tuple[str, ...]) -> set[str]:
                    queue = deque(changed)
                    affected = set(changed)
                    while queue:
                        current = queue.popleft()
                        for dependent in self._reverse.get(current, set()):
                            if dependent not in affected:
                                affected.add(dependent)
                                queue.append(dependent)
                    return affected

                def plan_rebuild(self, changed: set[str] | list[str] | tuple[str, ...]) -> list[str]:
                    affected = self.affected_nodes(changed)
                    indegree = {{name: 0 for name in affected}}
                    for name in affected:
                        for dependency in self._deps.get(name, ()):
                            if dependency in affected:
                                indegree[name] += 1
                    ready = deque(sorted(name for name, degree in indegree.items() if degree == 0))
                    order: list[str] = []
                    while ready:
                        current = ready.popleft()
                        order.append(current)
                        for dependent in sorted(self._reverse.get(current, set())):
                            if dependent not in indegree:
                                continue
                            indegree[dependent] -= 1
                            if indegree[dependent] == 0:
                                ready.append(dependent)
                    if len(order) != len(affected):
                        raise {error_name}("cycle detected in affected subgraph")
                    return order
            """
        )

        if difficulty == "medium":
            buggy = reference.replace(
                "                def affected_nodes(self, changed: set[str] | list[str] | tuple[str, ...]) -> set[str]:\n                    queue = deque(changed)\n                    affected = set(changed)\n                    while queue:\n                        current = queue.popleft()\n                        for dependent in self._reverse.get(current, set()):\n                            if dependent not in affected:\n                                affected.add(dependent)\n                                queue.append(dependent)\n                    return affected\n",
                "                def affected_nodes(self, changed: set[str] | list[str] | tuple[str, ...]) -> set[str]:\n                    affected = set(changed)\n                    for current in list(changed):\n                        affected.update(self._reverse.get(current, set()))\n                    return affected\n",
            )
            bug_types = ["rebuild selection only includes direct dependents"]
            strategy_traps = [
                "Planning order can look fine while the rebuild set is still incomplete",
                "A blunt fix that rebuilds everything removes the incremental guarantee",
            ]
        else:
            buggy = reference.replace(
                "                    old_dependencies = self._deps.get(name, ())\n                    for dependency in old_dependencies:\n                        self._reverse[dependency].discard(name)\n",
                "                    old_dependencies = self._deps.get(name, ())\n",
            )
            bug_types = ["rewriting node dependencies leaves stale reverse edges in the build graph"]
            strategy_traps = [
                "Affected-node traversal can be correct while stale reverse edges still over-rebuild or cycle",
                "Fixing only topological sort misses reverse-edge drift after node rewrites",
            ]

        visible_tests = render_test_module(
            dedent(
                f"""
                from build_graph import {class_name}


                class BuildGraphVisibleTests(unittest.TestCase):
                    def test_rebuild_order_respects_dependency_chain(self) -> None:
                        graph = {class_name}()
                        graph.set_node("parse", ["grammar"])
                        graph.set_node("optimize", ["parse"])
                        graph.set_node("bundle", ["optimize"])

                        self.assertEqual(
                            graph.plan_rebuild({{"grammar"}}),
                            ["grammar", "parse", "optimize", "bundle"],
                        )

                    def test_multiple_changes_are_deduplicated(self) -> None:
                        graph = {class_name}()
                        graph.set_node("a", ["src"])
                        graph.set_node("b", ["src"])
                        graph.set_node("c", ["a", "b"])
                        order = graph.plan_rebuild({{"src", "a"}})
                        self.assertEqual(order, ["src", "a", "b", "c"])
                """
            )
        )

        hidden_tests = render_test_module(
            dedent(
                f"""
                from build_graph import {class_name}, {error_name}


                class BuildGraphHiddenTests(unittest.TestCase):
                    def test_rewriting_dependencies_removes_stale_reverse_edges(self) -> None:
                        graph = {class_name}()
                        graph.set_node("compile", ["headers"])
                        graph.set_node("compile", ["sources"])
                        self.assertEqual(graph.plan_rebuild({{"headers"}}), ["headers"])
                        self.assertEqual(graph.plan_rebuild({{"sources"}}), ["sources", "compile"])

                    def test_cycle_in_affected_subgraph_raises(self) -> None:
                        graph = {class_name}()
                        graph.set_node("a", ["b"])
                        graph.set_node("b", ["a"])
                        with self.assertRaises({error_name}):
                            graph.plan_rebuild({{"a"}})
                """
            )
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "build_graph.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"build_graph.py": reference},
            entrypoint="build_graph.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["build_graph.py"],
                "expected_skill_tags": ["graphs", "topological-sort", "incremental-builds", "state-management"],
                "niche_topic": "incremental rebuild scheduling",
                "repairable": True,
            },
        )

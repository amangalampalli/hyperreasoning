import unittest

from cache import DependencyCache


class DependencyCacheHiddenTests(unittest.TestCase):
    def test_rewriting_rule_removes_old_reverse_edges(self) -> None:
        cache = DependencyCache()
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
        cache = DependencyCache()
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

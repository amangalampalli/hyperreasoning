import unittest

from cache import DependencyCache


class DependencyCacheVisibleTests(unittest.TestCase):
    def test_transitive_invalidation_recomputes_chain(self) -> None:
        cache = DependencyCache()
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

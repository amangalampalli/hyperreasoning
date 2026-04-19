import unittest

from transform import rename_global_binding


class ASTTransformHiddenTests(unittest.TestCase):
    def test_comprehension_target_is_not_renamed(self) -> None:
        source = 'shared_total = 2\ndef inspect(values):\n    return [(shared_total * 2) for shared_total in values], shared_total\n'
        rewritten = rename_global_binding(source, "shared_total", "global_total")
        namespace: dict[str, object] = {}
        exec(rewritten, namespace)
        self.assertEqual(namespace["inspect"]([4, 5]), ([8, 10], 2))

    def test_global_statement_tracks_renamed_binding(self) -> None:
        source = 'shared_total = 7\ndef bump():\n    global shared_total\n    shared_total += 1\n    return shared_total\n'
        rewritten = rename_global_binding(source, "shared_total", "global_total")
        namespace: dict[str, object] = {}
        exec(rewritten, namespace)
        self.assertEqual(namespace["bump"](), 8)
        self.assertEqual(namespace["global_total"], 8)

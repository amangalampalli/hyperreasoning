import unittest

from transform import rename_global_binding


class ASTTransformVisibleTests(unittest.TestCase):
    def test_preserves_shadowed_function_local(self) -> None:
        source = 'shared_total = 3\ndef use_shadow():\n    shared_total = 9\n    return shared_total\ndef use_global():\n    return shared_total\n'
        rewritten = rename_global_binding(source, "shared_total", "global_total")
        namespace: dict[str, object] = {}
        exec(rewritten, namespace)
        self.assertNotIn("shared_total", namespace)
        self.assertEqual(namespace["global_total"], 3)
        self.assertEqual(namespace["use_shadow"](), 9)
        self.assertEqual(namespace["use_global"](), 3)

    def test_renames_global_capture_in_nested_function(self) -> None:
        source = 'shared_total = 10\ndef outer(delta):\n    def inner():\n        return shared_total + delta\n    return inner()\n'
        rewritten = rename_global_binding(source, "shared_total", "global_total")
        namespace: dict[str, object] = {}
        exec(rewritten, namespace)
        self.assertEqual(namespace["outer"](5), 15)

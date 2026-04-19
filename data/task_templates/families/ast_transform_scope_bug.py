"""AST scope-aware transform task family."""

from __future__ import annotations

from data.task_templates.base import TaskSpec, TaskTemplate
from data.task_templates.utils import build_rng, choose_variant, dedent, make_task_id, render_test_module


class ASTTransformScopeBugTemplate(TaskTemplate):
    """Generate lexical-scope-sensitive AST rename tasks."""

    family = "ast_transform_scope_bug"

    def generate_instance(self, seed: int, difficulty: str) -> TaskSpec:
        self._validate_difficulty(difficulty)
        rng = build_rng(seed, self.family, difficulty)
        function_name = choose_variant(rng, ["rename_module_binding", "rename_global_binding"])
        old_name = choose_variant(rng, ["value", "shared_total", "baseline"])
        new_name = choose_variant(rng, ["renamed_value", "global_total", "base_amount"])
        prompt = choose_variant(
            rng,
            [
                f"Repair `{function_name}` in `transform.py`. It should rename the module-level binding "
                f"`{old_name}` to `{new_name}` in Python source while preserving lexical scoping. Do not "
                "rename shadowed locals, parameters, or comprehension targets that no longer refer to the "
                "module binding.",
                f"`transform.py` contains a scope-aware AST rename utility. Fix `{function_name}` so it only "
                f"renames references that still resolve to the module-level `{old_name}` binding, including "
                "nested functions that capture it, while leaving shadowed bindings alone.",
            ],
        )
        reference = dedent(
            f"""
            from __future__ import annotations

            import ast
            from dataclasses import dataclass, field


            @dataclass
            class ScopeInfo:
                kind: str
                bindings: set[str] = field(default_factory=set)
                global_names: set[str] = field(default_factory=set)


            def _collect_bindings(node: ast.AST) -> set[str]:
                bindings: set[str] = set()

                def add_target(target: ast.AST) -> None:
                    if isinstance(target, ast.Name):
                        bindings.add(target.id)
                    elif isinstance(target, (ast.Tuple, ast.List)):
                        for item in target.elts:
                            add_target(item)

                class Collector(ast.NodeVisitor):
                    def visit_Assign(self, inner: ast.Assign) -> None:
                        for target in inner.targets:
                            add_target(target)
                        self.generic_visit(inner.value)

                    def visit_AnnAssign(self, inner: ast.AnnAssign) -> None:
                        add_target(inner.target)
                        if inner.value is not None:
                            self.visit(inner.value)

                    def visit_AugAssign(self, inner: ast.AugAssign) -> None:
                        add_target(inner.target)
                        self.visit(inner.value)

                    def visit_For(self, inner: ast.For) -> None:
                        add_target(inner.target)
                        for child in inner.body + inner.orelse:
                            self.visit(child)

                    visit_AsyncFor = visit_For

                    def visit_With(self, inner: ast.With) -> None:
                        for item in inner.items:
                            if item.optional_vars is not None:
                                add_target(item.optional_vars)
                        for child in inner.body:
                            self.visit(child)

                    visit_AsyncWith = visit_With

                    def visit_Import(self, inner: ast.Import) -> None:
                        for alias in inner.names:
                            bindings.add(alias.asname or alias.name.split(".")[0])

                    def visit_ImportFrom(self, inner: ast.ImportFrom) -> None:
                        for alias in inner.names:
                            bindings.add(alias.asname or alias.name)

                    def visit_FunctionDef(self, inner: ast.FunctionDef) -> None:
                        bindings.add(inner.name)

                    visit_AsyncFunctionDef = visit_FunctionDef

                    def visit_ClassDef(self, inner: ast.ClassDef) -> None:
                        bindings.add(inner.name)

                    def visit_NamedExpr(self, inner: ast.NamedExpr) -> None:
                        add_target(inner.target)
                        self.visit(inner.value)

                Collector().visit(node)
                return bindings


            def _gather_globals(node: ast.AST) -> set[str]:
                names: set[str] = set()
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Global):
                        names.update(inner.names)
                return names


            class ScopeAwareRenamer(ast.NodeTransformer):
                def __init__(self, old_name: str, new_name: str) -> None:
                    self.old_name = old_name
                    self.new_name = new_name
                    self.scope_stack: list[ScopeInfo] = []

                def _push_scope(self, kind: str, bindings: set[str], global_names: set[str] | None = None) -> None:
                    self.scope_stack.append(ScopeInfo(kind=kind, bindings=bindings, global_names=global_names or set()))

                def _pop_scope(self) -> None:
                    self.scope_stack.pop()

                def _resolve_to_module(self) -> bool:
                    for scope in reversed(self.scope_stack[1:]):
                        if self.old_name in scope.global_names:
                            return True
                        if self.old_name in scope.bindings:
                            return False
                    return True

                def _rename_name_if_needed(self, node: ast.Name) -> ast.Name:
                    if node.id == self.old_name and self._resolve_to_module():
                        node.id = self.new_name
                    return node

                def visit_Module(self, node: ast.Module) -> ast.Module:
                    self._push_scope("module", _collect_bindings(node))
                    node = self.generic_visit(node)
                    self._pop_scope()
                    return node

                def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                    local_bindings = _collect_bindings(ast.Module(body=node.body, type_ignores=[]))
                    local_bindings.update(arg.arg for arg in node.args.args)
                    local_bindings.update(arg.arg for arg in node.args.posonlyargs)
                    local_bindings.update(arg.arg for arg in node.args.kwonlyargs)
                    if node.args.vararg is not None:
                        local_bindings.add(node.args.vararg.arg)
                    if node.args.kwarg is not None:
                        local_bindings.add(node.args.kwarg.arg)
                    self._push_scope("function", local_bindings, _gather_globals(ast.Module(body=node.body, type_ignores=[])))
                    node = self.generic_visit(node)
                    self._pop_scope()
                    return node

                visit_AsyncFunctionDef = visit_FunctionDef

                def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
                    local_bindings = {{arg.arg for arg in node.args.args}}
                    local_bindings.update(arg.arg for arg in node.args.posonlyargs)
                    local_bindings.update(arg.arg for arg in node.args.kwonlyargs)
                    self._push_scope("lambda", local_bindings)
                    node = self.generic_visit(node)
                    self._pop_scope()
                    return node

                def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
                    local_bindings = {{
                        target.id
                        for gen in node.generators
                        for target in ast.walk(gen.target)
                        if isinstance(target, ast.Name)
                    }}
                    self._push_scope("comprehension", local_bindings)
                    node = self.generic_visit(node)
                    self._pop_scope()
                    return node

                visit_SetComp = visit_ListComp
                visit_GeneratorExp = visit_ListComp
                visit_DictComp = visit_ListComp

                def visit_Global(self, node: ast.Global) -> ast.AST:
                    node.names = [self.new_name if name == self.old_name else name for name in node.names]
                    return node

                def visit_Name(self, node: ast.Name) -> ast.AST:
                    return self._rename_name_if_needed(node)


            def {function_name}(source: str, old_name: str, new_name: str) -> str:
                tree = ast.parse(source)
                renamed = ScopeAwareRenamer(old_name, new_name).visit(tree)
                ast.fix_missing_locations(renamed)
                return ast.unparse(renamed) + "\\n"
            """
        )

        if difficulty == "medium":
            buggy = dedent(
                f"""
                from __future__ import annotations

                import ast


                class BlindRenamer(ast.NodeTransformer):
                    def __init__(self, old_name: str, new_name: str) -> None:
                        self.old_name = old_name
                        self.new_name = new_name

                    def visit_Name(self, node: ast.Name) -> ast.AST:
                        if node.id == self.old_name:
                            node.id = self.new_name
                        return node

                    def visit_arg(self, node: ast.arg) -> ast.AST:
                        if node.arg == self.old_name:
                            node.arg = self.new_name
                        return node

                    def visit_Global(self, node: ast.Global) -> ast.AST:
                        node.names = [self.new_name if name == self.old_name else name for name in node.names]
                        return node


                def {function_name}(source: str, old_name: str, new_name: str) -> str:
                    tree = ast.parse(source)
                    renamed = BlindRenamer(old_name, new_name).visit(tree)
                    ast.fix_missing_locations(renamed)
                    return ast.unparse(renamed) + "\\n"
                """
            )
            bug_types = ["blind AST rename ignores lexical shadowing"]
            strategy_traps = [
                "Renaming every matching Name node breaks local bindings and parameters",
                "Passing visible examples does not prove comprehension or closure safety",
            ]
        else:
            buggy = reference.replace(
                "                def visit_ListComp(self, node: ast.ListComp) -> ast.AST:\n                    local_bindings = {\n                        target.id\n                        for gen in node.generators\n                        for target in ast.walk(gen.target)\n                        if isinstance(target, ast.Name)\n                    }\n                    self._push_scope(\"comprehension\", local_bindings)\n                    node = self.generic_visit(node)\n                    self._pop_scope()\n                    return node\n\n                visit_SetComp = visit_ListComp\n                visit_GeneratorExp = visit_ListComp\n                visit_DictComp = visit_ListComp\n",
                "",
            )
            bug_types = ["comprehension scopes are treated like surrounding module scope"]
            strategy_traps = [
                "Function-level shadow tracking can still be wrong for comprehensions",
                "The rename utility must preserve semantics, not just produce valid syntax",
            ]

        shadow_source = (
            f"{old_name} = 3\n"
            "def use_shadow():\n"
            f"    {old_name} = 9\n"
            f"    return {old_name}\n"
            "def use_global():\n"
            f"    return {old_name}\n"
        )
        nested_source = (
            f"{old_name} = 10\n"
            "def outer(delta):\n"
            "    def inner():\n"
            f"        return {old_name} + delta\n"
            "    return inner()\n"
        )
        comprehension_source = (
            f"{old_name} = 2\n"
            "def inspect(values):\n"
            f"    return [({old_name} * 2) for {old_name} in values], {old_name}\n"
        )
        global_source = (
            f"{old_name} = 7\n"
            "def bump():\n"
            f"    global {old_name}\n"
            f"    {old_name} += 1\n"
            f"    return {old_name}\n"
        )

        visible_tests = render_test_module(
            f"""
            from transform import {function_name}


            class ASTTransformVisibleTests(unittest.TestCase):
                def test_preserves_shadowed_function_local(self) -> None:
                    source = {shadow_source!r}
                    rewritten = {function_name}(source, "{old_name}", "{new_name}")
                    namespace: dict[str, object] = {{}}
                    exec(rewritten, namespace)
                    self.assertNotIn("{old_name}", namespace)
                    self.assertEqual(namespace["{new_name}"], 3)
                    self.assertEqual(namespace["use_shadow"](), 9)
                    self.assertEqual(namespace["use_global"](), 3)

                def test_renames_global_capture_in_nested_function(self) -> None:
                    source = {nested_source!r}
                    rewritten = {function_name}(source, "{old_name}", "{new_name}")
                    namespace: dict[str, object] = {{}}
                    exec(rewritten, namespace)
                    self.assertEqual(namespace["outer"](5), 15)
            """
        )

        hidden_tests = render_test_module(
            f"""
            from transform import {function_name}


            class ASTTransformHiddenTests(unittest.TestCase):
                def test_comprehension_target_is_not_renamed(self) -> None:
                    source = {comprehension_source!r}
                    rewritten = {function_name}(source, "{old_name}", "{new_name}")
                    namespace: dict[str, object] = {{}}
                    exec(rewritten, namespace)
                    self.assertEqual(namespace["inspect"]([4, 5]), ([8, 10], 2))

                def test_global_statement_tracks_renamed_binding(self) -> None:
                    source = {global_source!r}
                    rewritten = {function_name}(source, "{old_name}", "{new_name}")
                    namespace: dict[str, object] = {{}}
                    exec(rewritten, namespace)
                    self.assertEqual(namespace["bump"](), 8)
                    self.assertEqual(namespace["{new_name}"], 8)
            """
        )

        return self.build_spec(
            seed=seed,
            difficulty=difficulty,
            prompt=prompt,
            files={
                "transform.py": buggy,
                "test_visible.py": visible_tests,
                "test_hidden.py": hidden_tests,
            },
            reference_files={"transform.py": reference},
            entrypoint="transform.py",
            visible_test_file="test_visible.py",
            hidden_test_file="test_hidden.py",
            task_id=make_task_id(self.family, seed),
            metadata={
                "bug_type": bug_types,
                "strategy_traps": strategy_traps,
                "target_files": ["transform.py"],
                "expected_skill_tags": ["ast", "scope-analysis", "python-semantics", "transformations"],
                "niche_topic": "scope-aware AST rewriting",
                "repairable": True,
            },
        )

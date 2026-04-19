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
        local_bindings = {arg.arg for arg in node.args.args}
        local_bindings.update(arg.arg for arg in node.args.posonlyargs)
        local_bindings.update(arg.arg for arg in node.args.kwonlyargs)
        self._push_scope("lambda", local_bindings)
        node = self.generic_visit(node)
        self._pop_scope()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        local_bindings = {
            target.id
            for gen in node.generators
            for target in ast.walk(gen.target)
            if isinstance(target, ast.Name)
        }
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


def rename_global_binding(source: str, old_name: str, new_name: str) -> str:
    tree = ast.parse(source)
    renamed = ScopeAwareRenamer(old_name, new_name).visit(tree)
    ast.fix_missing_locations(renamed)
    return ast.unparse(renamed) + "\n"

from __future__ import annotations

import ast
import builtins
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set


class UnsafeCodeError(Exception):
    pass


@dataclass
class ExecResult:
    ok: bool
    locals: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    executed_code: Optional[str] = None  # code actually executed (imports stripped)


class _RemoveImports(ast.NodeTransformer):
    """Remove `import ...` and `from ... import ...` statements."""
    def visit_Import(self, node: ast.Import):
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom):
        return None


def strip_imports(code: str) -> str:
    """
    Remove import statements from Python code using AST.
    Python 3.9+ required for ast.unparse.
    """
    tree = ast.parse(code, mode="exec")
    tree = _RemoveImports().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


class SafeishPythonExecutor:
    """
    Simple "safe-ish" Python executor (NOT a real sandbox).
    - Strips imports from code before running (LLM often adds them)
    - Blocks dunder attribute access (.__class__, .__dict__, etc.)
    - Restricts builtins
    - Allows passing only explicit context vars (alt, pd, df, ...)
    """

    DEFAULT_BLOCKED_CALLS: Set[str] = {"eval", "exec", "compile", "open", "__import__"}

    def __init__(
        self,
        *,
        safe_globals: Optional[Dict[str, Any]] = None,
        blocked_names: Optional[Set[str]] = None,
        blocked_calls: Optional[Set[str]] = None,
        max_chars: int = 20_000,
        max_ast_nodes: int = 4000,
    ):
        self.max_chars = max_chars
        self.max_ast_nodes = max_ast_nodes
        self.blocked_names = set(blocked_names or set())
        self.blocked_calls = set(blocked_calls or set()) | set(self.DEFAULT_BLOCKED_CALLS)

        # Build safe builtins by copying non-dunder builtins and removing risky ones
        safe_builtins = {
            name: getattr(builtins, name)
            for name in dir(builtins)
            if not name.startswith("__")
        }
        for name in [
            "open", "exec", "eval", "compile", "input", "help",
            "dir", "vars", "globals", "locals",
            "exit", "quit", "getattr", "setattr", "delattr",
            "__import__", "breakpoint",
        ]:
            safe_builtins.pop(name, None)

        self.base_globals: Dict[str, Any] = {
            "__builtins__": safe_builtins,
        }

        # Add user-provided safe globals (e.g., alt, pd, math...)
        if safe_globals:
            self.base_globals.update(safe_globals)

    def _validate(self, code: str) -> None:
        if not isinstance(code, str):
            raise UnsafeCodeError("Code must be a string.")
        if len(code) > self.max_chars:
            raise UnsafeCodeError(f"Code too long (>{self.max_chars} chars).")

        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as e:
            raise UnsafeCodeError(f"Syntax error: {e}") from e

        # Complexity guard
        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.max_ast_nodes:
            raise UnsafeCodeError(f"Code too complex (>{self.max_ast_nodes} AST nodes).")

        for node in ast.walk(tree):
            # Block dunder attribute access: x.__class__, x.__dict__, ...
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__") and node.attr.endswith("__"):
                    raise UnsafeCodeError(f"Dunder attribute blocked: .{node.attr}")

            # Block specific names (optional)
            if isinstance(node, ast.Name) and node.id in self.blocked_names:
                raise UnsafeCodeError(f"Name blocked: {node.id}")

            # Block direct calls like eval(...), open(...)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in self.blocked_calls:
                    raise UnsafeCodeError(f"Call blocked: {node.func.id}(...)")

    def run(
        self,
        code: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        return_locals: bool = True,
        strip_imports_before_run: bool = True,
    ) -> ExecResult:
        """
        Execute code with additional context variables.
        Example context: {"df": df, "rows": rows}

        If strip_imports_before_run=True, import statements are removed from code
        before validation and execution.
        """
        try:
            executed_code = strip_imports(code) if strip_imports_before_run else code

            # Validate the code we will actually run
            self._validate(executed_code)

            g = dict(self.base_globals)
            if context:
                g.update(context)

            l: Dict[str, Any] = {}
            exec(compile(executed_code, "<safeish>", "exec"), g, l)  # intentional

            return ExecResult(
                ok=True,
                locals=(l if return_locals else None),
                error=None,
                executed_code=executed_code,
            )
        except Exception:
            return ExecResult(
                ok=False,
                locals=None,
                error=traceback.format_exc(),
                executed_code=None,
            )

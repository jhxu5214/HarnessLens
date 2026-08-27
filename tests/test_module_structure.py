"""Structural invariants that compiling and importing a module cannot catch.

A top-level block accidentally inserted into the middle of a function does not
raise: the statements after the insertion point keep the same indentation, so
Python happily attaches them to whatever definition now precedes them. The
truncated function then returns None and every caller fails at runtime, far
away from the real cause. These tests assert the shape of the package rather
than its behaviour, so that class of damage surfaces immediately.

Each check scans the whole package in one test and reports every offender at
once, which is both faster and less noisy than parametrising over 60 modules.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACKAGE = Path(__file__).resolve().parents[1] / "harnesslens"

# Functions whose contract is "return a payload dict". A None return from any of
# them shows up as `TypeError: 'NoneType' object is not iterable` inside a
# thread pool, which is exactly as unhelpful as it sounds.
TRIAL_RUNNERS = (
    ("benchmarks/opencode_tau2.py", "run_single_opencode_tau2_trial"),
    ("benchmarks/codex_tau2.py", "run_single_codex_tau2_trial"),
    ("benchmarks/pi_tau2.py", "run_single_pi_tau2_trial"),
)


def _parsed_modules() -> list[tuple[str, ast.Module]]:
    return [
        (
            path.relative_to(PACKAGE).as_posix(),
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path)),
        )
        for path in sorted(PACKAGE.rglob("*.py"))
        if "__pycache__" not in path.parts
    ]


def _functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_no_function_body_ends_on_an_import():
    """An import as the final statement of a function means the body was cut."""
    offenders = [
        f"{name}:{node.body[-1].lineno} {function}"
        for name, tree in _parsed_modules()
        for function, node in _functions(tree).items()
        if isinstance(node.body[-1], (ast.Import, ast.ImportFrom))
    ]
    assert offenders == []


def test_no_statement_follows_an_unconditional_return():
    """Dead code after a return at the same level is the other half of that damage."""
    offenders = []
    for name, tree in _parsed_modules():
        for function, node in _functions(tree).items():
            for index, statement in enumerate(node.body[:-1]):
                if isinstance(statement, ast.Return):
                    offenders.append(
                        f"{name}:{node.body[index + 1].lineno} unreachable in "
                        f"{function} after return on line {statement.lineno}"
                    )
    assert offenders == []


@pytest.mark.parametrize(("module", "function"), TRIAL_RUNNERS)
def test_trial_runner_returns_a_value_on_its_success_path(module: str, function: str):
    """Every trial runner is called as ``dict(runner(...))`` and must return one."""
    path = PACKAGE / module
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    node = _functions(tree).get(function)
    assert node is not None, f"{module}: {function} is missing"
    returns = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Return) and child.value is not None
    ]
    assert returns, f"{module}: {function} never returns a value"

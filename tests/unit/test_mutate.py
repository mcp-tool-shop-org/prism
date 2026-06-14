"""Tests for the AST mutation operators (family-AB buggy-label generation)."""

from __future__ import annotations

import ast

from prism.eval.mutate import generate_mutants


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def test_relational_operator_is_mutated() -> None:
    code = "def sign(x):\n    if x > 0:\n        return 1\n    return 0\n"
    mutants = generate_mutants(code)
    assert any(m.operator == "ror" for m in mutants)
    assert any("Gt->GtE" in m.description for m in mutants)
    assert any(">= 0" in m.code for m in mutants)


def test_arithmetic_operator_is_mutated() -> None:
    mutants = generate_mutants("def add(a, b):\n    return a + b\n")
    assert any(m.operator == "aor" for m in mutants)
    assert any("a - b" in m.code for m in mutants)


def test_logical_connector_is_mutated() -> None:
    code = "def both(a, b):\n    return a and b\n"
    mutants = generate_mutants(code)
    assert any(m.operator == "lcr" for m in mutants)
    assert any(" or " in m.code for m in mutants)


def test_all_mutants_parse_and_differ_from_source() -> None:
    code = "def f(a, b):\n    if a >= b and a > 0:\n        return a + b\n    return a - b\n"
    base = ast.unparse(ast.parse(code))
    mutants = generate_mutants(code)
    assert mutants
    for m in mutants:
        assert _parses(m.code)
        assert m.code != base


def test_mutants_are_unique() -> None:
    code = "def f(a, b):\n    if a >= b and a > 0:\n        return a + b\n    return a - b\n"
    mutants = generate_mutants(code)
    assert len({m.code for m in mutants}) == len(mutants)


def test_no_mutable_operator_yields_empty() -> None:
    assert generate_mutants("def const():\n    return 42\n") == []


def test_syntax_error_yields_empty() -> None:
    assert generate_mutants("def f(:\n    return\n") == []


def test_limit_is_respected() -> None:
    code = "def f(a, b, c, d):\n    return a + b - c + d - a + b\n"
    assert len(generate_mutants(code, limit=2)) <= 2

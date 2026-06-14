"""Tests for the execution-labeling sandbox (family-AB ground truth)."""

from __future__ import annotations

from prism.eval.sandbox import ERROR, FAIL, PASS, TIMEOUT, run_candidate

_ADD_TEST = (
    "def check(candidate):\n"
    "    assert candidate(1, 2) == 3\n"
    "    assert candidate(-1, 1) == 0\n"
)


def test_passing_candidate_is_clean() -> None:
    out = run_candidate("def add(a, b):\n    return a + b\n", _ADD_TEST, "add")
    assert out.passed is True
    assert out.status == PASS
    assert out.is_buggy is False


def test_wrong_logic_is_buggy_fail() -> None:
    out = run_candidate("def add(a, b):\n    return a - b\n", _ADD_TEST, "add")
    assert out.passed is False
    assert out.status == FAIL
    assert out.is_buggy is True


def test_runtime_crash_is_error() -> None:
    out = run_candidate("def add(a, b):\n    return a + undefined_name\n", _ADD_TEST, "add")
    assert out.passed is False
    assert out.status == ERROR


def test_syntax_error_is_error() -> None:
    out = run_candidate("def add(a, b)\n    return a + b\n", _ADD_TEST, "add")
    assert out.passed is False
    assert out.status == ERROR


def test_missing_entry_point_is_error() -> None:
    out = run_candidate("def other(a, b):\n    return a + b\n", _ADD_TEST, "add")
    assert out.passed is False
    assert out.status == ERROR


def test_infinite_loop_times_out() -> None:
    code = "def add(a, b):\n    while True:\n        pass\n"
    out = run_candidate(code, _ADD_TEST, "add", timeout_s=1.0)
    assert out.passed is False
    assert out.status == TIMEOUT
    assert out.is_buggy is True

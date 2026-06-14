"""Tests for difficulty-stratified problem selection (deconfound-stratum composition)."""

from __future__ import annotations

from prism.eval.familygen import ProblemSpec
from prism.eval.problem_select import count_by_difficulty, stratified_by_difficulty


def _p(pid: str, diff: str) -> ProblemSpec:
    return ProblemSpec(
        id=pid, intent="i", entry_point="f", test_code="def check(c): pass", difficulty=diff
    )


def test_count_by_difficulty() -> None:
    probs = [_p("a", "easy"), _p("b", "easy"), _p("c", "hard"), _p("d", "")]
    assert count_by_difficulty(probs) == {"easy": 2, "hard": 1, "": 1}


def test_takes_up_to_n_per_difficulty() -> None:
    probs = [_p(f"e{i}", "easy") for i in range(5)] + [_p(f"h{i}", "hard") for i in range(3)]
    sel = stratified_by_difficulty(probs, {"easy": 2, "hard": 10})
    assert count_by_difficulty(sel) == {"easy": 2, "hard": 3}  # easy capped; hard takes all


def test_deterministic_given_seed() -> None:
    probs = [_p(f"e{i}", "easy") for i in range(10)]
    a = stratified_by_difficulty(probs, {"easy": 3}, seed=7)
    b = stratified_by_difficulty(probs, {"easy": 3}, seed=7)
    assert [p.id for p in a] == [p.id for p in b]
    assert len(a) == 3


def test_ignores_unrequested_and_absent_difficulties() -> None:
    probs = [_p("a", "easy"), _p("b", "hard")]
    easy_only = stratified_by_difficulty(probs, {"easy": 5})  # hard not requested -> excluded
    assert [p.id for p in easy_only] == ["a"]
    assert stratified_by_difficulty(probs, {"medium": 5}) == []  # none present

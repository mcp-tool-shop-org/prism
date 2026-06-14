"""Tests for the failing-test SIGNATURE (the deconfounder's same-bug key).

A signature is the SET of hidden tests a candidate fails. The deconfounder keeps a restyled variant
only when its signature exactly matches the canonical bug's — so a clean candidate -> empty set, the
same bug -> an identical set, and two bugs that fail different cases -> different sets.
"""

from __future__ import annotations

from prism.eval.bigcodebench import load_bigcodebench_offline
from prism.eval.livecodebench import load_livecodebench_offline
from prism.eval.sandbox import run_failure_signature


def test_clean_candidate_has_empty_signature() -> None:
    spec = load_livecodebench_offline()[0]  # Solution.sumList(nums)
    sig = run_failure_signature(
        "class Solution:\n    def sumList(self, nums):\n        return sum(nums)\n",
        spec.test_code,
        spec.entry_point,
    )
    assert sig == frozenset()  # nothing fails


def test_same_bug_matches_different_bug_differs() -> None:
    spec = load_livecodebench_offline()[0]  # cases: [1,2,3]->6 (idx 0), []->0 (idx 1)
    bug_a = (
        "class Solution:\n    def sumList(self, nums):\n        return sum(nums) if nums else 1\n"
    )
    bug_b = (
        "class Solution:\n"
        "    def sumList(self, nums):\n"
        "        return sum(nums) + (1 if nums else 0)\n"
    )
    sig_a1 = run_failure_signature(bug_a, spec.test_code, spec.entry_point)
    sig_a2 = run_failure_signature(bug_a, spec.test_code, spec.entry_point)
    sig_b = run_failure_signature(bug_b, spec.test_code, spec.entry_point)
    assert sig_a1 == sig_a2  # the SAME bug -> an identical failing-test signature
    assert sig_a1 and sig_b and sig_a1 != sig_b  # different bugs fail different cases


def test_bigcodebench_signature_clean_vs_buggy() -> None:
    spec = load_bigcodebench_offline()[0]  # task_func -> sum
    clean = run_failure_signature(
        "def task_func(numbers):\n    return sum(numbers)\n", spec.test_code, spec.entry_point
    )
    assert clean == frozenset()
    buggy = run_failure_signature(
        "def task_func(numbers):\n    return sum(numbers) + 1\n", spec.test_code, spec.entry_point
    )
    assert buggy  # a non-empty failing-test set


def test_errored_candidate_has_no_signature() -> None:
    spec = load_livecodebench_offline()[0]
    # no Solution class defined -> entry point missing -> signature undefined (None), not a match
    sig = run_failure_signature("x = 1\n", spec.test_code, spec.entry_point)
    assert sig is None

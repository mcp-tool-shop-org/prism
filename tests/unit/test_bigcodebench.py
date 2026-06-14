"""Tests for the BigCodeBench loader (family-AB harder-corpus source).

The fixture rows are stdlib-only and runnable, so the check-wrapper is validated END-TO-END through
the real ``sandbox.run_candidate``: a correct candidate PASSES, a buggy one is labeled buggy —
proving the unittest-class -> ``check(candidate)`` contract works against the existing sandbox.
"""

from __future__ import annotations

import pytest

from prism.eval.bigcodebench import (
    BCBColumnError,
    _parse_libs,
    bigcodebench_content_hash,
    load_bigcodebench,
    load_bigcodebench_offline,
)
from prism.eval.sandbox import run_candidate


def test_offline_maps_fields() -> None:
    specs = load_bigcodebench_offline()
    assert len(specs) == 2
    s = specs[0]
    assert s.id == "bcb-bigcodebench-fixture-sum"
    assert s.entry_point == "task_func"
    assert s.source == "bigcodebench"
    assert "sum of a list" in s.intent
    assert "def check(candidate)" in s.test_code  # the appended wrapper
    assert "class TestCases(unittest.TestCase)" in s.test_code  # the original row test survives


def test_libs_parsed_from_stringified_list() -> None:
    vowels = load_bigcodebench_offline()[1]
    assert vowels.libs == ("re",)
    assert vowels.difficulty == "medium"


def test_parse_libs_variants() -> None:
    assert _parse_libs("['random', 'itertools']") == ("random", "itertools")
    assert _parse_libs("[]") == ()
    assert _parse_libs(["numpy", "pandas"]) == ("numpy", "pandas")
    assert _parse_libs("not-a-list") == ()
    assert _parse_libs(None) == ()


def test_andon_on_missing_required_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    bad = tmp_path / "bad.jsonl"
    # a row missing entry_point must fail loud, never silently mislabel
    bad.write_text('{"task_id": "x", "test": "import unittest"}\n', encoding="utf-8")
    with pytest.raises(BCBColumnError):
        load_bigcodebench_offline(source=bad)


def test_content_hash_is_deterministic_and_drift_sensitive() -> None:
    specs = load_bigcodebench_offline()
    assert bigcodebench_content_hash(specs) == bigcodebench_content_hash(specs)
    # a content edit flips the hash (drift visible)
    assert bigcodebench_content_hash(specs) != bigcodebench_content_hash(specs[:1])


def test_online_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        load_bigcodebench(split="v9.9.9-bogus")


def test_limit_caps_the_slice() -> None:
    assert len(load_bigcodebench_offline(limit=1)) == 1


def test_check_wrapper_labels_correct_and_buggy_candidates() -> None:
    spec = load_bigcodebench_offline()[0]  # task_func -> sum of a list
    good_code = "def task_func(numbers):\n    return sum(numbers)\n"
    good = run_candidate(good_code, spec.test_code, spec.entry_point)
    assert good.passed is True
    assert good.is_buggy is False
    bug_code = "def task_func(numbers):\n    return sum(numbers) + 1\n"
    bug = run_candidate(bug_code, spec.test_code, spec.entry_point)
    assert bug.passed is False
    assert bug.is_buggy is True

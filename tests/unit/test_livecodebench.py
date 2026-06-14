"""Tests for the LiveCodeBench loader (the contamination-dated family-AB source).

Covers the two footguns directly: the base64->zlib->pickle->json private-test decode chain, and the
multi-arg newline-separated functional input parsing (unit-tested on ``build_cases``, escaping
explicit). The check-synthesis is validated END-TO-END through the real sandbox.
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib

import pytest

from prism.eval.livecodebench import (
    LCB_VERSIONS,
    build_cases,
    decode_private_tests,
    livecodebench_content_hash,
    load_livecodebench,
    load_livecodebench_offline,
)
from prism.eval.sandbox import run_candidate


def test_offline_keeps_functional_skips_stdin() -> None:
    specs = load_livecodebench_offline()
    # 3 fixture rows: 2 functional + 1 stdin (skipped) -> 2 specs
    assert len(specs) == 2
    s = specs[0]
    assert s.id == "lcb-sumlist-1"
    assert s.entry_point == "Solution"
    assert s.source == "livecodebench"
    assert s.contest_date == "2025-06-01T00:00:00"
    assert "class Solution" in s.intent  # starter_code folded into the generation intent
    assert "def check(candidate)" in s.test_code
    assert "_CASES" in s.test_code


def test_decode_private_tests_plain_json() -> None:
    blob = json.dumps([{"input": "[1]", "output": "1", "testtype": "functional"}])
    assert decode_private_tests(blob) == [{"input": "[1]", "output": "1", "testtype": "functional"}]


def test_decode_private_tests_b64_zlib_pickle() -> None:
    # the LCB hot path: base64(zlib(pickle(JSON-STRING))) — json.loads runs AFTER pickle.loads
    payload = json.dumps([{"input": "[5]", "output": "5", "testtype": "functional"}])
    blob = base64.b64encode(zlib.compress(pickle.dumps(payload))).decode("utf-8")
    assert decode_private_tests(blob) == [{"input": "[5]", "output": "5", "testtype": "functional"}]


def test_build_cases_parses_multiline_multi_arg_input() -> None:
    # the footgun: a functional input is newline-separated per-arg JSON literals
    public = json.dumps([{"input": "2\n3", "output": "5", "testtype": "functional"}])
    cases = build_cases(public, "[]")
    assert cases == [{"args": [2, 3], "expected": 5}]


def test_build_cases_jsonifies_tuple_fallback() -> None:
    # a non-JSON literal arg falls back to literal_eval; tuples are jsonified to lists for baking
    public = json.dumps([{"input": "(1, 2)", "output": "[1, 2]", "testtype": "functional"}])
    cases = build_cases(public, "[]")
    assert cases == [{"args": [[1, 2]], "expected": [1, 2]}]


def test_check_synthesis_labels_correct_and_buggy_via_sandbox() -> None:
    spec = load_livecodebench_offline()[0]  # Solution.sumList(nums)
    good = run_candidate(
        "class Solution:\n    def sumList(self, nums):\n        return sum(nums)\n",
        spec.test_code,
        spec.entry_point,
    )
    assert good.passed is True
    bug = run_candidate(
        "class Solution:\n    def sumList(self, nums):\n        return sum(nums) + 1\n",
        spec.test_code,
        spec.entry_point,
    )
    assert bug.is_buggy is True


def test_date_window_filters_by_contest_date() -> None:
    late = load_livecodebench_offline(start_date="2025-09-01")
    assert [s.id for s in late] == ["lcb-sumlist-2"]  # only the 2025-12-01 problem
    early = load_livecodebench_offline(end_date="2025-09-01")
    assert [s.id for s in early] == ["lcb-sumlist-1"]  # only the 2025-06-01 problem


def test_content_hash_deterministic_and_drift_sensitive() -> None:
    specs = load_livecodebench_offline()
    assert livecodebench_content_hash(specs) == livecodebench_content_hash(specs)
    assert livecodebench_content_hash(specs) != livecodebench_content_hash(specs[:1])


def test_online_rejects_unknown_version() -> None:
    assert "release_v6" in LCB_VERSIONS
    with pytest.raises(ValueError):
        load_livecodebench(version="release_v99")

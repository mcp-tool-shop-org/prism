"""Authored seed problems for the family-AB generation corpus (offline default set).

These are HAND-AUTHORED for prism (novel phrasings, not lifted from HumanEval/MBPP/LeetCode), so they
are genuinely post-cutoff / uncontaminated at authoring time — the contamination control the
family-AB needs so a same-family judge recognizes a family's STYLE, not a memorized public solution
(LiveCodeBench temporal-holdout rationale, Jain et al. 2024, arXiv:2403.07974). Each ships a hardened
``check`` with edge cases (empty input, off-by-one, distinctness), so a thin generation FAILS by
execution and becomes a natural buggy artifact. Downloaded post-cutoff problem sets are the scale
upgrade; this seed proves the pipeline end-to-end offline.
"""

# ruff: noqa: E501 — data module: embedded problem specs + their test code are clearer unwrapped.
from __future__ import annotations

from prism.eval.familygen import ProblemSpec

FRESH_PROBLEMS: list[ProblemSpec] = [
    ProblemSpec(
        id="running-max",
        intent="Given a list of integers, return a new list where each element is the maximum of the input list up to and including that position. Return an empty list for an empty input.",
        entry_point="running_max",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate([3, 1, 4, 1, 5]) == [3, 3, 4, 4, 5]\n"
            "    assert candidate([]) == []\n"
            "    assert candidate([-1, -3, -2]) == [-1, -1, -1]\n"
            "    assert candidate([7]) == [7]\n"
        ),
    ),
    ProblemSpec(
        id="second-smallest",
        intent="Return the second smallest DISTINCT value in a list of integers. If there are fewer than two distinct values, return None.",
        entry_point="second_smallest",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate([3, 1, 2]) == 2\n"
            "    assert candidate([2, 1]) == 2\n"
            "    assert candidate([1, 1, 2]) == 2\n"
            "    assert candidate([5]) is None\n"
            "    assert candidate([4, 4, 4]) is None\n"
        ),
    ),
    ProblemSpec(
        id="is-rotation",
        intent="Return True if string b is a rotation of string a (b can be obtained by moving some prefix of a to its end), otherwise False. Two empty strings are rotations of each other; strings of different lengths are never rotations.",
        entry_point="is_rotation",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate('abcde', 'cdeab') is True\n"
            "    assert candidate('abc', 'acb') is False\n"
            "    assert candidate('', '') is True\n"
            "    assert candidate('a', 'a') is True\n"
            "    assert candidate('abc', 'ab') is False\n"
        ),
    ),
    ProblemSpec(
        id="collapse-spaces",
        intent="Collapse every run of whitespace characters in a string to a single space and strip leading/trailing whitespace. An empty string maps to an empty string.",
        entry_point="collapse_spaces",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate('  a   b  ') == 'a b'\n"
            "    assert candidate('') == ''\n"
            "    assert candidate('x') == 'x'\n"
            "    assert candidate('a\\t\\nb') == 'a b'\n"
        ),
    ),
    ProblemSpec(
        id="digit-root",
        intent="Return the digital root of a non-negative integer: repeatedly sum its decimal digits until a single digit remains. The digital root of 0 is 0.",
        entry_point="digit_root",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate(0) == 0\n"
            "    assert candidate(9) == 9\n"
            "    assert candidate(10) == 1\n"
            "    assert candidate(9875) == 2\n"
        ),
    ),
    ProblemSpec(
        id="first-unique-char",
        intent="Return the index of the first character in a string that does not repeat anywhere in the string. Return -1 if every character repeats or the string is empty.",
        entry_point="first_unique_char",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate('leetcode') == 0\n"
            "    assert candidate('loveleetcode') == 2\n"
            "    assert candidate('aabb') == -1\n"
            "    assert candidate('') == -1\n"
        ),
    ),
    ProblemSpec(
        id="balanced-brackets",
        intent="Return True if the brackets in a string are balanced and properly nested, considering only (), [], and {} and ignoring all other characters. The empty string is balanced.",
        entry_point="balanced_brackets",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate('()[]{}') is True\n"
            "    assert candidate('([{}])') is True\n"
            "    assert candidate('(]') is False\n"
            "    assert candidate('(') is False\n"
            "    assert candidate('') is True\n"
            "    assert candidate('ab(c)d') is True\n"
        ),
    ),
    ProblemSpec(
        id="clamp-list",
        intent="Given a list of numbers and bounds lo and hi (lo <= hi), return a new list with each value clamped into the inclusive range [lo, hi].",
        entry_point="clamp_list",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate([-5, 0, 5, 10], 0, 7) == [0, 0, 5, 7]\n"
            "    assert candidate([], 0, 1) == []\n"
            "    assert candidate([3], 0, 2) == [2]\n"
            "    assert candidate([1, 2, 3], 0, 10) == [1, 2, 3]\n"
        ),
    ),
    ProblemSpec(
        id="chunk",
        intent="Split a list into consecutive chunks of size n (n >= 1); the final chunk may be shorter. An empty list yields an empty list.",
        entry_point="chunk",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]\n"
            "    assert candidate([], 3) == []\n"
            "    assert candidate([1, 2], 5) == [[1, 2]]\n"
            "    assert candidate([1, 2, 3], 1) == [[1], [2], [3]]\n"
        ),
    ),
    ProblemSpec(
        id="count-vowels",
        intent="Count the vowels (a, e, i, o, u, case-insensitive) in a string.",
        entry_point="count_vowels",
        test_code=(
            "def check(candidate):\n"
            "    assert candidate('Hello World') == 3\n"
            "    assert candidate('') == 0\n"
            "    assert candidate('xyz') == 0\n"
            "    assert candidate('AEIOU') == 5\n"
        ),
    ),
]

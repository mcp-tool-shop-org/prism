"""BigCodeBench loader — hard, function-level, execution-labelable problems for the family-AB corpus.

Loads the HF dataset ``bigcode/bigcodebench`` (arXiv:2406.15877, Apache-2.0). BigCodeBench is the
workhorse HARDER source the study-swarm (wf_127b6806-943) recommended: self-contained functions where
top models fail ~40-50%, so model generations land in the false-accept VARIANCE regime the pilot's
trivial mutants lacked. Each row carries a ``test`` that is a full ``unittest.TestCase`` calling the
``entry_point`` function — so a family's GENERATED solution is execution-labelable with no LLM in the
label loop (prism's family-agnostic ground-truth requirement).

VERIFIED against the HF datasets-server ``/info`` + ``/first-rows`` (2026-06-14):

  * Config ``default``; splits are dataset VERSIONS: ``v0.1.0_hf`` .. ``v0.1.4`` (1140 problems each).
  * Columns (verbatim): ``task_id``, ``complete_prompt``, ``instruct_prompt``, ``canonical_solution``,
    ``code_prompt``, ``test``, ``entry_point``, ``doc_struct``, ``libs``.
  * ``test`` is a complete ``class TestCases(unittest.TestCase)`` that invokes ``entry_point`` (e.g.
    ``task_func``) by global name; ``libs`` is a stringified list of required imports (often stdlib,
    sometimes pandas/numpy/etc. — those route the generation to the CONTAINERIZED labeler).

The loader maps a row to a ``ProblemSpec`` whose ``test_code`` is the row's ``test`` plus an appended
``check(candidate)`` that discovers and runs every ``unittest.TestCase`` in the namespace — so the
existing ``sandbox.run_candidate`` (which execs the candidate + test_code in one namespace and calls
``check``) labels the generation PASS/FAIL with no change to the sandbox contract.

Like the CodeJudgeBench loader, the default install does NOT carry HF ``datasets``: the ONLINE path
lazy-imports it (clear ``pip install 'prism-verify[bench]'`` error if absent), the OFFLINE path reads a
committed fixture so tests need neither network nor the lib. Columns are ANDON-validated on every row.
"""

# ruff: noqa: E501 — schema-contract module: the column/split contract comments (mirroring the HF
# dataset card verbatim) are clearer unwrapped, as in corpus.py / codejudgebench.py.
from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path

from prism.eval.familygen import ProblemSpec

# Splits ARE dataset versions; the newest committed version is the default (verified on the server).
BCB_SPLITS = ("v0.1.0_hf", "v0.1.1", "v0.1.2", "v0.1.3", "v0.1.4")
DEFAULT_SPLIT = "v0.1.4"
HF_DATASET = "bigcode/bigcodebench"

# Columns the loader REQUIRES. A missing/renamed one ANDON-fails rather than silently mislabeling.
_REQUIRED_COLUMNS = ("task_id", "test", "entry_point")

CONTENT_HASH_SCHEMA = "prism-bigcodebench-problems/v1"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = _REPO_ROOT / "eval" / "sources" / "bigcodebench" / "fixture.jsonl"

# Appended to each row's ``test`` so the unittest suite satisfies the sandbox ``check(candidate)``
# contract. It discovers EVERY TestCase subclass in the exec namespace (robust to the class name) and
# raises AssertionError on any failure/error — exactly what ``sandbox.run_candidate`` labels as buggy.
_CHECK_WRAPPER = '''

def check(candidate):
    import io as _io
    import unittest as _unittest

    _loader = _unittest.TestLoader()
    _suite = _unittest.TestSuite()
    _added = 0
    for _obj in list(globals().values()):
        if (
            isinstance(_obj, type)
            and issubclass(_obj, _unittest.TestCase)
            and _obj is not _unittest.TestCase
        ):
            _suite.addTests(_loader.loadTestsFromTestCase(_obj))
            _added += 1
    if _added == 0:
        raise AssertionError("bigcodebench: no unittest.TestCase subclass found in test code")
    _result = _unittest.TextTestRunner(stream=_io.StringIO(), verbosity=0).run(_suite)
    if not _result.wasSuccessful():
        raise AssertionError(
            f"bigcodebench tests failed: {len(_result.failures)} failures, "
            f"{len(_result.errors)} errors"
        )
'''


class BCBColumnError(ValueError):
    """ANDON: a BigCodeBench row is missing/renamed an expected column (fail loud, never mislabel)."""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def _parse_libs(value: object) -> tuple[str, ...]:
    """Coerce the stringified-list ``libs`` field (e.g. ``"['random', 'itertools']"``) to a tuple.

    Already-a-list rows are handled too. Anything unparseable yields ``()`` (treated as stdlib-only).
    """
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return ()
        if isinstance(parsed, (list, tuple)):
            return tuple(str(v) for v in parsed)
    return ()


def _row_to_spec(row: dict[str, object], *, index: int) -> ProblemSpec:
    """Map ONE validated HF/fixture row to a ``ProblemSpec``. ANDON-raises on a missing column."""
    for col in _REQUIRED_COLUMNS:
        if col not in row:
            raise BCBColumnError(
                f"BigCodeBench row (index={index}) is missing required column {col!r}; columns "
                f"present: {sorted(row)}. Refusing to map — a renamed test/entry_point would "
                "silently mislabel every generation."
            )
    task_id = str(row["task_id"])
    entry_point = str(row["entry_point"]).strip()
    if not entry_point:
        raise BCBColumnError(f"BigCodeBench row (index={index}, task_id={task_id!r}) has empty entry_point")
    # Prefer the natural-language instruct prompt for generation; fall back to the code prompt.
    intent = str(row.get("instruct_prompt") or row.get("complete_prompt") or "").strip()
    if not intent:
        raise BCBColumnError(
            f"BigCodeBench row (index={index}, task_id={task_id!r}) has neither instruct_prompt "
            "nor complete_prompt to generate from"
        )
    test_code = str(row["test"]) + _CHECK_WRAPPER
    return ProblemSpec(
        id=f"bcb-{_slug(task_id) or index}",
        intent=intent,
        entry_point=entry_point,
        test_code=test_code,
        libs=_parse_libs(row.get("libs")),
        contest_date=None,  # BigCodeBench is a fixed (post-cutoff, 2024) set with no per-item date
        difficulty=str(row.get("difficulty", "")),
        source="bigcodebench",
    )


def load_bigcodebench_offline(
    limit: int | None = None, *, source: Path = DEFAULT_FIXTURE
) -> list[ProblemSpec]:
    """Load the committed fixture (no network, no ``datasets``). ANDON-validates each row like HF."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"BigCodeBench fixture not found at {source} — ship eval/sources/bigcodebench/fixture.jsonl "
            "or load online with [bench] installed."
        )
    specs: list[ProblemSpec] = []
    for index, raw in enumerate(source.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise BCBColumnError(f"fixture line {index} is not a JSON object: {line[:80]!r}")
        specs.append(_row_to_spec(row, index=index))
        if limit is not None and len(specs) >= limit:
            break
    return specs


def _load_hf(split: str, limit: int | None) -> list[ProblemSpec]:
    """ONLINE path: lazy-import ``datasets``, stream the split, ANDON-validate columns."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - exercised only without the optional [bench] extra
        raise ImportError(
            "BigCodeBench online load needs the HF 'datasets' library. Install the optional extra: "
            "pip install 'prism-verify[bench]'  (the offline fixture path needs neither network nor "
            "datasets)."
        ) from exc

    ds = load_dataset(HF_DATASET, split=split)
    specs: list[ProblemSpec] = []
    for index, row in enumerate(ds):  # row is a dict-like mapping column -> value
        specs.append(_row_to_spec(dict(row), index=index))
        if limit is not None and len(specs) >= limit:
            break
    return specs


def load_bigcodebench(
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    *,
    offline: bool = False,
    fixture: Path = DEFAULT_FIXTURE,
) -> list[ProblemSpec]:
    """Load BigCodeBench problems as ``ProblemSpec``s (one function-level problem per row).

    ``offline=True`` reads the committed fixture (no network, no ``datasets``). Otherwise the HF online
    path streams ``split`` (a dataset VERSION, e.g. ``v0.1.4``). ``limit`` caps the loaded slice.
    Columns are ANDON-validated on every row (see ``BCBColumnError``).
    """
    if offline:
        return load_bigcodebench_offline(limit, source=fixture)
    if split not in BCB_SPLITS:
        raise ValueError(f"unknown BigCodeBench split {split!r}; expected one of {BCB_SPLITS}")
    return _load_hf(split, limit)


def bigcodebench_content_hash(specs: list[ProblemSpec]) -> str:
    """Deterministic SHA-256 over the canonicalized spec set (mirrors corpus.corpus_content_hash)."""
    canonical = sorted(json.dumps(asdict(s), sort_keys=True) for s in specs)
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_SCHEMA.encode("utf-8"))
    digest.update(b"\x00")
    for line in canonical:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

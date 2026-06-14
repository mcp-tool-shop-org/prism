"""LiveCodeBench loader — contamination-dated, function-level problems for the family-AB corpus.

Loads ``livecodebench/code_generation_lite`` (arXiv:2403.07974, MIT). LiveCodeBench is the study-swarm's
PRIMARY source (wf_127b6806-943) because every problem carries a ``contest_date``, giving a per-family
post-cutoff temporal holdout — the contamination control a same-family self-preference probe needs.

VERIFIED recipe (HF dataset card + the repo loader + datasets-version issues, 2026-06-14):

  * SCRIPT-based dataset: ``load_dataset("livecodebench/code_generation_lite", split="test",
    version_tag="release_v6", trust_remote_code=True)``. Works on ``datasets`` 3.x; ``datasets>=4``
    REMOVES dataset scripts (our ``[bench]`` pin is ``datasets<4`` — load-bearing).
  * Every column arrives as a STRING. ``question_content`` is the prompt; ``public_test_cases`` is a
    JSON string; ``private_test_cases`` is JSON, else base64 -> zlib -> pickle -> json (the hot path
    in the lite set); ``metadata`` is a JSON string whose ``func_name`` is the entry point.
  * FUNCTIONAL iff ``metadata["func_name"]`` is set (LeetCode-style ``class Solution``); otherwise the
    problem is STDIN/STDOUT. **STDIN problems are DEFERRED** — they need a whole-program stdin harness,
    not the ``check(candidate)`` contract — so the loader keeps FUNCTIONAL problems only and records the
    skipped stdin count. ``contest_date`` (ISO-8601) drives the date-window holdout.

SECURITY: the online path runs the dataset repo's LOADING SCRIPT (``trust_remote_code=True``) — arbitrary
code from the HF repo at load time. That is the upstream's design; pin the dataset revision in a real run
and treat it as a trusted-source dependency. The OFFLINE fixture path runs no remote code.

The functional test cases (per-arg JSON literals) are parsed to ``(args, expected)`` at LOAD time (here,
unit-testable) and baked into a trivial ``check(candidate)`` that instantiates ``Solution`` and calls the
entry point — so ``sandbox.run_candidate`` labels a generation with no sandbox change.
"""

# ruff: noqa: E501 — schema-contract module: the column/encoding contract comments (mirroring the
# verified HF recipe verbatim) are clearer unwrapped, as in corpus.py / bigcodebench.py.
from __future__ import annotations

import ast
import base64
import hashlib
import json
import pickle  # noqa: S403 - used ONLY to decode the dataset's own test-case blobs (see decode note)
import re
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from prism.eval.familygen import ProblemSpec

HF_DATASET = "livecodebench/code_generation_lite"
DEFAULT_VERSION = "release_v6"
LCB_VERSIONS = (
    "release_v1", "release_v2", "release_v3", "release_v4", "release_v5", "release_v6",
    "release_latest",
)

_REQUIRED_COLUMNS = ("question_content", "public_test_cases", "private_test_cases", "metadata")
CONTENT_HASH_SCHEMA = "prism-livecodebench-problems/v1"
# The LeetCode-style entry symbol: generations implement ``class Solution`` with the func_name method.
SOLUTION_CLASS = "Solution"

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_FIXTURE = _REPO_ROOT / "eval" / "sources" / "livecodebench" / "fixture.jsonl"


class LCBColumnError(ValueError):
    """ANDON: a LiveCodeBench row is missing/renamed an expected column (fail loud, never mislabel)."""


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def decode_private_tests(blob: str) -> list[dict[str, object]]:
    """Decode ``private_test_cases`` with the EXACT LiveCodeBench chain: JSON first, else b64/zlib/pickle.

    The pickled payload is itself a JSON string, so ``json.loads`` runs AFTER ``pickle.loads`` (do not
    reorder). ``pickle`` is used only on the dataset's OWN blob — not on candidate output — but it is
    still remote data, so a real run should pin the dataset revision.
    """
    try:
        parsed = json.loads(blob)
    except (json.JSONDecodeError, ValueError):
        parsed = json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode("utf-8")))))  # noqa: S301
    return list(parsed)


def _parse_arg(line: str) -> object:
    """Parse one functional-input line (a JSON literal per LeetCode convention); fall back gracefully."""
    try:
        return json.loads(line)
    except (json.JSONDecodeError, ValueError):
        try:
            return ast.literal_eval(line)
        except (ValueError, SyntaxError):
            return line


def _parse_expected(output: str) -> object:
    return _parse_arg(output)


def _jsonify(value: object) -> object:
    """Make a parsed value JSON-bakeable: tuples/sets -> lists, recursively (others pass through)."""
    if isinstance(value, (list, tuple, set)):
        return [_jsonify(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    return value


def build_cases(
    public_blob: str, private_blob: str, *, max_cases: int = 50
) -> list[dict[str, object]]:
    """Parse the functional public+private tests into ``[{"args": [...], "expected": ...}]`` at load time.

    Each test's ``input`` is newline-separated per-argument JSON literals; ``output`` is the expected
    return. Parsing here (not in the generated check) keeps the intricate arg-decoding unit-testable.
    PUBLIC tests come first and are always kept (up to the cap); ``max_cases`` then bounds the total —
    real LCB problems can carry HUNDREDS of private cases (observed: 3 MB of baked test_code), which a
    representative subset reliably labels clean/buggy without ballooning the corpus + content-hash.
    """
    raw = list(json.loads(public_blob)) + decode_private_tests(private_blob)
    cases: list[dict[str, object]] = []
    for t in raw:
        if not isinstance(t, dict) or "input" not in t or "output" not in t:
            continue
        lines = [ln for ln in str(t["input"]).split("\n") if ln.strip() != ""]
        args = [_jsonify(_parse_arg(ln)) for ln in lines]
        cases.append({"args": args, "expected": _jsonify(_parse_expected(str(t["output"])))})
        if len(cases) >= max_cases:
            break
    return cases


def _build_check_code(func_name: str, cases: list[dict[str, object]]) -> str:
    """A trivial ``check(candidate)`` over pre-parsed cases: instantiate Solution, call the entry point."""
    return (
        "import json as _json\n\n"
        f"_FUNC_NAME = {func_name!r}\n"
        f"_CASES = _json.loads({json.dumps(cases)!r})\n\n"
        "def check(candidate):\n"
        "    _inst = candidate() if isinstance(candidate, type) else candidate\n"
        "    _fn = getattr(_inst, _FUNC_NAME)\n"
        "    for _case in _CASES:\n"
        "        _result = _fn(*_case['args'])\n"
        "        assert _result == _case['expected'], (\n"
        "            f\"{_FUNC_NAME}({_case['args']}) = {_result!r} != {_case['expected']!r}\"\n"
        "        )\n"
        "\n"
        "def failing_tests(candidate):\n"
        "    # The SET of failing case indices (the deconfounder's same-bug key), sorted.\n"
        "    _inst = candidate() if isinstance(candidate, type) else candidate\n"
        "    _fn = getattr(_inst, _FUNC_NAME)\n"
        "    _fails = []\n"
        "    for _i, _case in enumerate(_CASES):\n"
        "        try:\n"
        "            _ok = _fn(*_case['args']) == _case['expected']\n"
        "        except Exception:\n"
        "            _ok = False\n"
        "        if not _ok:\n"
        "            _fails.append(str(_i))\n"
        "    return sorted(_fails)\n"
    )


def _row_to_spec(row: dict[str, object], *, index: int, max_cases: int = 50) -> ProblemSpec | None:
    """Map ONE row to a functional ``ProblemSpec``, or ``None`` if it is a (deferred) stdin problem."""
    for col in _REQUIRED_COLUMNS:
        if col not in row:
            raise LCBColumnError(
                f"LiveCodeBench row (index={index}) is missing required column {col!r}; columns "
                f"present: {sorted(row)}. Refusing to map — a renamed test column would mislabel."
            )
    metadata = json.loads(str(row["metadata"])) if row["metadata"] else {}
    func_name = metadata.get("func_name") if isinstance(metadata, dict) else None
    if not func_name:
        return None  # stdin/stdout problem — deferred (needs a whole-program harness, not check())
    cases = build_cases(
        str(row["public_test_cases"]), str(row["private_test_cases"]), max_cases=max_cases
    )
    if not cases:
        return None  # no usable functional cases
    qid = str(row.get("question_id") or f"{index}")
    starter = str(row.get("starter_code", ""))
    intent = str(row["question_content"])
    if starter.strip():
        intent = f"{intent}\n\nImplement the provided class/method signature:\n{starter}"
    return ProblemSpec(
        id=f"lcb-{_slug(qid) or index}",
        intent=intent,
        entry_point=SOLUTION_CLASS,
        test_code=_build_check_code(str(func_name), cases),
        libs=(),  # LiveCodeBench functional problems are algorithmic / stdlib
        contest_date=str(row.get("contest_date") or "") or None,
        difficulty=str(row.get("difficulty", "")),
        source="livecodebench",
    )


def _within_window(spec: ProblemSpec, start_date: str | None, end_date: str | None) -> bool:
    """True if the spec's contest_date is inside [start_date, end_date] (YYYY-MM-DD bounds)."""
    if start_date is None and end_date is None:
        return True
    if not spec.contest_date:
        return False
    try:
        when = datetime.fromisoformat(spec.contest_date)
    except ValueError:
        return False
    if start_date is not None and when < datetime.fromisoformat(start_date):
        return False
    if end_date is not None and when > datetime.fromisoformat(end_date):
        return False
    return True


def load_livecodebench_offline(
    limit: int | None = None,
    *,
    source: Path = DEFAULT_FIXTURE,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cases: int = 50,
) -> list[ProblemSpec]:
    """Load the committed fixture (no network, no remote code). Functional-only; stdin rows are skipped."""
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(
            f"LiveCodeBench fixture not found at {source} — ship eval/sources/livecodebench/fixture.jsonl "
            "or load online with [bench] installed."
        )
    specs: list[ProblemSpec] = []
    for index, raw in enumerate(source.read_text(encoding="utf-8").splitlines()):
        line = raw.strip()
        if not line:
            continue
        rowobj = json.loads(line)
        if not isinstance(rowobj, dict):
            raise LCBColumnError(f"fixture line {index} is not a JSON object: {line[:80]!r}")
        spec = _row_to_spec(rowobj, index=index, max_cases=max_cases)
        if spec is None or not _within_window(spec, start_date, end_date):
            continue
        specs.append(spec)
        if limit is not None and len(specs) >= limit:
            break
    return specs


def _load_hf(
    version: str,
    limit: int | None,
    start_date: str | None,
    end_date: str | None,
    max_cases: int,
) -> list[ProblemSpec]:
    """ONLINE path: lazy-import ``datasets``, run the dataset's loading script, keep functional rows."""
    try:
        from datasets import load_dataset  # type: ignore[import-not-found, unused-ignore]
    except ImportError as exc:  # pragma: no cover - exercised only without the optional [bench] extra
        raise ImportError(
            "LiveCodeBench online load needs the HF 'datasets' library (and datasets<4 for the loading "
            "script). Install: pip install 'prism-verify[bench]'  (the offline fixture path needs neither "
            "network nor datasets)."
        ) from exc

    ds = load_dataset(HF_DATASET, split="test", version_tag=version, trust_remote_code=True)
    specs: list[ProblemSpec] = []
    for index, row in enumerate(ds):
        spec = _row_to_spec(dict(row), index=index, max_cases=max_cases)
        if spec is None or not _within_window(spec, start_date, end_date):
            continue
        specs.append(spec)
        if limit is not None and len(specs) >= limit:
            break
    return specs


def load_livecodebench(
    version: str = DEFAULT_VERSION,
    limit: int | None = None,
    *,
    offline: bool = False,
    fixture: Path = DEFAULT_FIXTURE,
    start_date: str | None = None,
    end_date: str | None = None,
    max_cases: int = 50,
) -> list[ProblemSpec]:
    """Load LiveCodeBench FUNCTIONAL problems as ``ProblemSpec``s (stdin problems are skipped).

    ``offline=True`` reads the committed fixture. Otherwise the HF online path runs the dataset's loading
    script (``version_tag``, ``trust_remote_code=True`` — needs ``datasets<4``). ``start_date`` /
    ``end_date`` (YYYY-MM-DD) apply the contest-date holdout — set ``start_date`` after a family's cutoff.
    ``max_cases`` bounds the baked test cases per problem (real LCB problems can carry MBs of tests).
    """
    if offline:
        return load_livecodebench_offline(
            limit, source=fixture, start_date=start_date, end_date=end_date, max_cases=max_cases
        )
    if version not in LCB_VERSIONS:
        raise ValueError(f"unknown LiveCodeBench version {version!r}; expected one of {LCB_VERSIONS}")
    return _load_hf(version, limit, start_date, end_date, max_cases)


def livecodebench_content_hash(specs: list[ProblemSpec]) -> str:
    """Deterministic SHA-256 over the canonicalized spec set (mirrors corpus.corpus_content_hash)."""
    canonical = sorted(json.dumps(asdict(s), sort_keys=True) for s in specs)
    digest = hashlib.sha256()
    digest.update(CONTENT_HASH_SCHEMA.encode("utf-8"))
    digest.update(b"\x00")
    for line in canonical:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()

"""Sandboxed execution-labeling for generated code artifacts (family-AB ground truth).

prism's family-different A/B (Lock-1) measures whether a *same-family* verifier under-refutes its
OWN family's buggy code. That needs GROUND-TRUTH buggy/clean labels that are *family-agnostic* — an
LLM grading the code would re-introduce the exact self-preference the experiment isolates (the
label-circularity threat the study-swarm completeness-critic flagged). So labels here come from
RUNNING the code against a hidden test suite: PASS => clean, FAIL/TIMEOUT/ERROR => buggy. Execution
ground truth, no model in the label loop (SWE-bench, Jimenez et al. 2023, arXiv:2310.06770;
EvalPlus, Liu et al. 2023, arXiv:2305.01210 — thin suites mislabel, so seed problems ship hardened
tests).

SECURITY (threat model). This module EXECUTES model-generated code. Each candidate runs in a fresh
CHILD PROCESS with (a) a hard wall-clock timeout — kills runaway / infinite-loop code, itself a
common real defect, so a timeout is *labeled buggy*, not discarded — and (b) a ``reliability_guard``
that neuters the most destructive ``os`` / ``shutil`` / ``subprocess`` entry points BEFORE the
candidate runs (the HumanEval / EvalPlus pattern). This is DEFENSE-IN-DEPTH for *our own* local
model output on self-contained function problems, NOT a security boundary for adversarial code: a
determined exploit can still escape an in-process guard. For large or untrusted runs, run the whole
build inside Docker / WSL with no network and a throwaway volume (the SWE-bench posture). The guard,
timeout, and minimal-env child process are the proportionate control for the v1 trusted-local use.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Execution statuses. PASS is the only "clean" verdict; every other status means the candidate did
# not satisfy its hidden tests and is labeled BUGGY (positive) downstream.
PASS = "pass"
FAIL = "fail"  # a test assertion was violated (wrong behavior)
TIMEOUT = "timeout"  # exceeded the wall-clock budget (e.g. an infinite loop)
ERROR = "error"  # raised at import/exec/runtime, or the entry point was missing

# Child-process exit codes (see ``_RUNNER_SRC``): 0 PASS, 1 FAIL (AssertionError). 2 runtime crash,
# 3 candidate failed to compile/exec, 4 entry point absent — all map to ERROR.
_EXIT_PASS = 0
_EXIT_FAIL = 1


@dataclass(frozen=True)
class ExecOutcome:
    """The result of running one candidate against one problem's hidden tests."""

    passed: bool
    status: str  # PASS | FAIL | TIMEOUT | ERROR
    detail: str

    @property
    def is_buggy(self) -> bool:
        """True iff the candidate did NOT pass — the execution-based positive (buggy) label."""
        return not self.passed


# The child-process runner. It is a FIXED template — the candidate source, the test source, and the
# entry-point name are passed as FILES (never string-interpolated into this script), so untrusted
# code can never alter the runner's own logic. It applies the reliability guard BEFORE executing.
_RUNNER_SRC = r'''
import json
import pathlib
import sys


def reliability_guard():
    # Neuter the most destructive entry points before any candidate code runs. Not a sandbox; a
    # defense-in-depth guard for trusted-local output (HumanEval/EvalPlus pattern). Imports the
    # candidate may legitimately need (math, itertools, ...) are left intact.
    import builtins
    import os
    import shutil

    for name in (
        "system", "popen", "remove", "removedirs", "rmdir", "unlink", "kill", "killpg",
        "fork", "forkpty", "abort", "chmod", "chown", "chroot", "lchmod", "lchown",
        "rename", "renames", "truncate", "replace",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, None)
    try:
        import subprocess
        subprocess.Popen = None
        subprocess.run = None
        subprocess.call = None
    except Exception:
        pass
    builtins.exit = None
    builtins.quit = None


reliability_guard()

_here = pathlib.Path(__file__).resolve().parent
_code = (_here / "candidate.txt").read_text(encoding="utf-8")
_test = (_here / "test.txt").read_text(encoding="utf-8")
_meta = json.loads((_here / "meta.json").read_text(encoding="utf-8"))
_entry = _meta["entry_point"]

_ns = {}
try:
    exec(compile(_code, "<candidate>", "exec"), _ns)
except BaseException:
    sys.exit(3)

if _entry not in _ns:
    sys.exit(4)

try:
    exec(compile(_test, "<test>", "exec"), _ns)
    _ns["check"](_ns[_entry])
except AssertionError:
    sys.exit(1)
except BaseException:
    sys.exit(2)

sys.exit(0)
'''


def run_candidate(
    code: str,
    test_code: str,
    entry_point: str,
    *,
    timeout_s: float = 5.0,
) -> ExecOutcome:
    """Run ``code`` against ``test_code`` in an isolated child process; return the labeled outcome.

    ``test_code`` must define ``def check(candidate): ...`` that raises ``AssertionError`` on a
    wrong result; ``entry_point`` is the function name the candidate is expected to define. A clean
    PASS is the only non-buggy outcome; FAIL / TIMEOUT / ERROR are all execution-based positive
    (buggy) labels. All I/O happens in a throwaway temp dir wiped on return.
    """
    with tempfile.TemporaryDirectory(prefix="prism-sandbox-") as tmp:
        d = Path(tmp)
        (d / "candidate.txt").write_text(code, encoding="utf-8")
        (d / "test.txt").write_text(test_code, encoding="utf-8")
        (d / "meta.json").write_text(json.dumps({"entry_point": entry_point}), encoding="utf-8")
        runner = d / "runner.py"
        runner.write_text(_RUNNER_SRC, encoding="utf-8")

        try:
            proc = subprocess.run(
                [sys.executable, str(runner)],
                cwd=str(d),
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecOutcome(False, TIMEOUT, f"exceeded {timeout_s}s wall-clock budget")

        rc = proc.returncode
        if rc == _EXIT_PASS:
            return ExecOutcome(True, PASS, "all tests passed")
        if rc == _EXIT_FAIL:
            return ExecOutcome(False, FAIL, "a test assertion failed")
        # rc 2 (runtime crash), 3 (compile/exec error), 4 (missing entry point), or any signal.
        stderr_tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        return ExecOutcome(False, ERROR, f"exit {rc}: {stderr_tail[0][:200]}")

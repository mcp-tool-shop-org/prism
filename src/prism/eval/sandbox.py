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

# Child-process exit codes (see ``_RUNNER_SRC``): 0 PASS, 1 FAIL (AssertionError). 2 (any other
# raise: crash, compile, SystemExit, guarded-call) and 4 (entry point absent) both map to ERROR.
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
import os
import pathlib

# Capture the real process-exit primitive BEFORE any candidate code can patch sys.exit / os._exit.
# The runner's authoritative exit goes through THIS reference, so a candidate that no-ops sys.exit
# (to make the failure path fall through to a 0) cannot force a false-clean label.
_real_exit = os._exit


def reliability_guard():
    # Neuter the most destructive / process-relabeling entry points before any candidate runs. Not
    # a sandbox; a defense-in-depth guard for trusted-local output (HumanEval/EvalPlus pattern).
    # Imports the candidate may legitimately need (math, itertools, ...) are left intact.
    import builtins
    import shutil

    for name in (
        "system", "popen", "remove", "removedirs", "rmdir", "unlink", "kill", "killpg",
        "fork", "forkpty", "abort", "chmod", "chown", "chroot", "lchmod", "lchown",
        "rename", "renames", "truncate", "replace", "startfile", "_exit",
        "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe",
        "spawnl", "spawnle", "spawnv", "spawnve", "spawnvp", "spawnvpe",
    ):
        if hasattr(os, name):
            setattr(os, name, None)
    for name in ("rmtree", "move", "chown"):
        if hasattr(shutil, name):
            setattr(shutil, name, None)
    try:
        import subprocess
        for name in ("Popen", "run", "call", "check_call", "check_output"):
            if hasattr(subprocess, name):
                setattr(subprocess, name, None)
    except Exception:
        pass
    builtins.exit = None
    builtins.quit = None


reliability_guard()

_here = pathlib.Path(__file__).resolve().parent
_meta = json.loads((_here / "meta.json").read_text(encoding="utf-8"))
_entry = _meta["entry_point"]
_code = (_here / "candidate.txt").read_text(encoding="utf-8")
_test = (_here / "test.txt").read_text(encoding="utf-8")

# Default to ERROR (buggy): any unexpected control flow labels the artifact buggy, NEVER silently
# clean. rc becomes 0 (PASS) ONLY after check() returns without raising. The final exit uses the
# captured os._exit so candidate code cannot rewrite the verdict by patching the exit primitives.
rc = 2
_ns = {}
try:
    exec(compile(_code, "<candidate>", "exec"), _ns)
    if _entry not in _ns:
        rc = 4
    else:
        exec(compile(_test, "<test>", "exec"), _ns)
        _ns["check"](_ns[_entry])
        rc = 0
except AssertionError:
    rc = 1
except BaseException as _exc:
    rc = 2
    # Surface the exception type on stderr so the labeler can tell an UNRUNNABLE (missing dep:
    # ModuleNotFoundError) from a genuine crash-bug. The runner already defaults rc=2 (buggy); this
    # only adds triage detail (the last stderr line is the exception class + message).
    import sys as _sys
    import traceback as _tb
    _tb.print_exception(type(_exc), _exc, _exc.__traceback__, file=_sys.stderr)

_real_exit(rc)
'''


def write_runner_dir(target: Path, code: str, test_code: str, entry_point: str) -> Path:
    """Write the fixed runner + candidate/test/meta files into ``target``; return the runner path.

    The candidate source, test source, and entry point are passed as FILES (never interpolated into
    the runner template), so untrusted code can never alter the runner's own logic. Shared by the
    in-process sandbox and the containerized labeler so both execute byte-identically.
    """
    target.mkdir(parents=True, exist_ok=True)
    (target / "candidate.txt").write_text(code, encoding="utf-8")
    (target / "test.txt").write_text(test_code, encoding="utf-8")
    (target / "meta.json").write_text(json.dumps({"entry_point": entry_point}), encoding="utf-8")
    runner = target / "runner.py"
    runner.write_text(_RUNNER_SRC, encoding="utf-8")
    return runner


def outcome_from_exit(returncode: int, stderr: bytes) -> ExecOutcome:
    """Map the runner's child-process exit code to an ``ExecOutcome`` (shared by both labelers).

    0 => PASS (clean); 1 => FAIL (a test assertion); everything else (2 raise/crash/SystemExit, 4
    missing entry point, any signal) => ERROR. The ERROR detail keeps the last stderr line
    (a ``ModuleNotFoundError`` there is how a caller tells an unrunnable from a genuine bug).
    """
    if returncode == _EXIT_PASS:
        return ExecOutcome(True, PASS, "all tests passed")
    if returncode == _EXIT_FAIL:
        return ExecOutcome(False, FAIL, "a test assertion failed")
    stderr_tail = stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
    return ExecOutcome(False, ERROR, f"exit {returncode}: {stderr_tail[0][:200]}")


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
        runner = write_runner_dir(d, code, test_code, entry_point)
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
        return outcome_from_exit(proc.returncode, proc.stderr)

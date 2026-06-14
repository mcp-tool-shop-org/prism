"""Tests for the containerized labeler: argv isolation, exit mapping, timeout, route-by-libs.

The docker subprocess is INJECTED (a fake ``runner``), so these need no real Docker; the stdlib
fallback paths exercise the real in-process sandbox.
"""

from __future__ import annotations

import subprocess

from prism.eval.container_sandbox import (
    docker_available,
    label_candidate,
    run_candidate_in_container,
)
from prism.eval.familygen import ProblemSpec


class _Runner:
    """A subprocess.run stand-in that records argv and fakes docker version/run/kill outcomes."""

    def __init__(self, *, run_rc: int = 0, stderr: bytes = b"", run_timeout: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.run_rc = run_rc
        self.stderr = stderr
        self.run_timeout = run_timeout

    def __call__(self, argv, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(list(argv))
        sub = argv[1] if len(argv) > 1 else ""
        if sub == "run":
            if self.run_timeout:
                raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout"))
            return subprocess.CompletedProcess(argv, self.run_rc, b"", self.stderr)
        return subprocess.CompletedProcess(argv, 0, b"", b"")  # version / kill


def _run_argv(runner: _Runner) -> list[str]:
    return next(c for c in runner.calls if len(c) > 1 and c[1] == "run")


def test_docker_available() -> None:
    assert docker_available(runner=_Runner()) is True

    def _boom(*_a, **_k):  # type: ignore[no-untyped-def]
        raise OSError("no docker")

    assert docker_available(runner=_boom) is False


def test_container_argv_is_isolated_and_passes_on_rc0() -> None:
    r = _Runner(run_rc=0)
    out = run_candidate_in_container("x = 1", "def check(c): pass", "x", runner=r, image="img:test")
    assert out.passed is True
    argv = _run_argv(r)
    assert "--network" in argv and "none" in argv
    assert "--read-only" in argv
    assert any(a == "--memory" for a in argv)
    assert any(a == "--pids-limit" for a in argv)
    assert "img:test" in argv
    assert argv[-3:] == ["img:test", "python", "/work/runner.py"]


def test_container_exit_mapping() -> None:
    fail = run_candidate_in_container("x", "y", "e", runner=_Runner(run_rc=1))
    assert fail.status == "fail" and fail.is_buggy is True
    err = run_candidate_in_container("x", "y", "e", runner=_Runner(run_rc=2, stderr=b"Boom"))
    assert err.status == "error" and err.is_buggy is True


def test_container_timeout_kills_and_labels() -> None:
    r = _Runner(run_timeout=True)
    out = run_candidate_in_container("x", "y", "e", runner=r)
    assert out.status == "timeout"
    assert any(len(c) > 1 and c[1] == "kill" for c in r.calls)  # best-effort cleanup attempted


def test_router_uses_container_for_lib_problems() -> None:
    spec = ProblemSpec(
        id="p", intent="i", entry_point="x", test_code="def check(c): pass", libs=("numpy",)
    )
    r = _Runner(run_rc=0)
    out = label_candidate("x = 1", spec, runner=r, container_check=lambda **_k: True)
    assert out.passed is True
    assert any(len(c) > 1 and c[1] == "run" for c in r.calls)  # went through the container


def test_router_stdlib_problem_uses_in_process_sandbox() -> None:
    spec = ProblemSpec(
        id="p", intent="i", entry_point="f",
        test_code="def check(candidate):\n    assert candidate(2) == 4\n",
    )
    r = _Runner(run_rc=0)
    # libs=() -> no container even though container_check would say yes; runs the real sandbox
    out = label_candidate(
        "def f(x):\n    return x * 2\n", spec, runner=r, container_check=lambda **_k: True
    )
    assert out.passed is True
    assert r.calls == []  # never touched docker


def test_router_falls_back_when_docker_unavailable() -> None:
    spec = ProblemSpec(
        id="p", intent="i", entry_point="g",
        test_code="def check(candidate):\n    assert candidate() == 1\n",
        libs=("numpy",),
    )
    out = label_candidate("def g():\n    return 1\n", spec, container_check=lambda **_k: False)
    assert out.passed is True  # fell back to the in-process sandbox (no numpy import needed here)

"""Containerized execution-labeler for the family-AB corpus (BigCodeBench lib-bearing problems).

BigCodeBench problems import third-party libs (pandas/numpy/...) the in-process sandbox env may
not carry, and they execute model-generated code. This module runs the SAME fixed runner
(sandbox._RUNNER_SRC, via write_runner_dir) inside a Docker container that ships the lib set, with
hard isolation: --network none (no exfiltration, no network flakiness), a read-only rootfs plus a
small writable /tmp tmpfs (BigCodeBench tests create temp files), and memory / pids / cpu caps plus
a wall-clock timeout (a runaway is killed and labeled buggy, never hangs the build).

ROUTE BY LIBS (the director's choice): a problem whose ProblemSpec.libs is non-empty is labeled in
the container; a stdlib-only problem (most LiveCodeBench-functional) takes the fast in-process
sandbox. If the container is requested but Docker is unavailable, it falls back to the in-process
sandbox -- a then-missing lib surfaces as a ModuleNotFoundError ERROR that familygen SKIPS, never
mislabeled buggy.

Build the image once (CI or operator):
    docker build -t prism-labeler:latest -f eval/docker/labeler.Dockerfile .

The runner callable (defaults to subprocess.run) is injected so argv construction + exit mapping
are unit-tested without Docker; an opt-in real-Docker smoke lives behind docker_available.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from prism.eval.sandbox import (
    ExecOutcome,
    outcome_from_exit,
    parse_signature,
    run_candidate,
    run_failure_signature,
    write_runner_dir,
)


class Labelable(Protocol):
    """Problem fields the labeler reads — a frozen ``ProblemSpec`` satisfies this structurally.

    A Protocol (not an import of ``ProblemSpec``) so ``container_sandbox`` doesn't depend on
    ``familygen`` (which imports ``label_candidate`` — a back-import would cycle). Members are
    read-only properties so a frozen dataclass conforms.
    """

    @property
    def libs(self) -> tuple[str, ...]: ...
    @property
    def test_code(self) -> str: ...
    @property
    def entry_point(self) -> str: ...

# Image name (override per deployment); built from eval/docker/labeler.Dockerfile.
DEFAULT_IMAGE = os.environ.get("PRISM_LABELER_IMAGE", "prism-labeler:latest")

# A subprocess.run-shaped callable. Injected in tests so no real Docker is needed.
RunnerFn = Callable[..., "subprocess.CompletedProcess[bytes]"]


def docker_available(*, docker_cmd: str = "docker", runner: RunnerFn = subprocess.run) -> bool:
    """True iff ``docker version`` succeeds — the preflight the router gates container use on."""
    try:
        proc = runner([docker_cmd, "version"], capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def _docker_argv(
    image: str,
    name: str,
    workdir: Path,
    *,
    memory_mb: int,
    pids_limit: int,
    cpus: str,
    docker_cmd: str,
) -> list[str]:
    """The isolated ``docker run`` argv shared by the label + signature container paths."""
    return [
        docker_cmd, "run", "--rm",
        "--name", name,
        "--network", "none",
        "--memory", f"{memory_mb}m",
        "--memory-swap", f"{memory_mb}m",  # no swap, so the memory cap holds
        "--pids-limit", str(pids_limit),
        "--cpus", str(cpus),
        "--read-only",
        "--tmpfs", "/tmp:rw,size=64m,exec",  # tests create temp files
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{workdir}:/work:ro",
        "-w", "/work",
        image,
        "python", "/work/runner.py",
    ]


def _docker_run(
    argv: list[str], name: str, *, timeout_s: float, docker_cmd: str, runner: RunnerFn
) -> subprocess.CompletedProcess[bytes] | None:
    """Run the container with timeout margin; kill + return None on a wall-clock blow."""
    try:
        return runner(argv, capture_output=True, timeout=timeout_s + 30.0, check=False)
    except subprocess.TimeoutExpired:
        # Best-effort: stop the lingering container (the client was killed, not the engine).
        try:
            runner([docker_cmd, "kill", name], capture_output=True, timeout=20, check=False)
        except (OSError, subprocess.SubprocessError):
            pass
        return None


def run_candidate_in_container(
    code: str,
    test_code: str,
    entry_point: str,
    *,
    image: str = DEFAULT_IMAGE,
    timeout_s: float = 10.0,
    memory_mb: int = 512,
    pids_limit: int = 256,
    cpus: str = "1.0",
    docker_cmd: str = "docker",
    runner: RunnerFn = subprocess.run,
) -> ExecOutcome:
    """Label ``code`` against ``test_code`` inside an isolated Docker container; return the outcome.

    Same contract as ``sandbox.run_candidate`` (PASS clean; FAIL/TIMEOUT/ERROR buggy); the runner
    executes in a ``--network none``, read-only, resource-capped container carrying the lib set.
    """
    with tempfile.TemporaryDirectory(prefix="prism-clabel-") as tmp:
        d = Path(tmp)
        write_runner_dir(d, code, test_code, entry_point)
        name = f"prism-labeler-{d.name}"
        argv = _docker_argv(
            image, name, d,
            memory_mb=memory_mb, pids_limit=pids_limit, cpus=cpus, docker_cmd=docker_cmd,
        )
        proc = _docker_run(argv, name, timeout_s=timeout_s, docker_cmd=docker_cmd, runner=runner)
        if proc is None:
            return ExecOutcome(False, "timeout", f"container exceeded {timeout_s}s budget")
        return outcome_from_exit(proc.returncode, proc.stderr or b"")


def run_signature_in_container(
    code: str,
    test_code: str,
    entry_point: str,
    *,
    image: str = DEFAULT_IMAGE,
    timeout_s: float = 10.0,
    memory_mb: int = 512,
    pids_limit: int = 256,
    cpus: str = "1.0",
    docker_cmd: str = "docker",
    runner: RunnerFn = subprocess.run,
) -> frozenset[str] | None:
    """The failing-test SET of ``code`` computed inside the container (for lib-bearing problems)."""
    with tempfile.TemporaryDirectory(prefix="prism-csig-") as tmp:
        d = Path(tmp)
        write_runner_dir(d, code, test_code, entry_point, mode="signature")
        name = f"prism-sig-{d.name}"
        argv = _docker_argv(
            image, name, d,
            memory_mb=memory_mb, pids_limit=pids_limit, cpus=cpus, docker_cmd=docker_cmd,
        )
        proc = _docker_run(argv, name, timeout_s=timeout_s, docker_cmd=docker_cmd, runner=runner)
    if proc is None or proc.returncode != 0:
        return None
    return parse_signature(proc.stdout or b"")


def label_candidate(
    code: str,
    problem: Labelable,
    *,
    timeout_s: float = 10.0,
    image: str = DEFAULT_IMAGE,
    force_container: bool = False,
    docker_cmd: str = "docker",
    runner: RunnerFn = subprocess.run,
    container_check: Callable[..., bool] = docker_available,
) -> ExecOutcome:
    """Route to the right labeler: container for lib-bearing problems, else the fast sandbox.

    A problem needs the container iff it declares third-party ``libs`` (or ``force_container``).
    If the container is needed but Docker is unavailable, it falls back to the in-process sandbox —
    a then-missing lib surfaces as a ``ModuleNotFoundError`` ERROR familygen treats as unrunnable.
    """
    needs_container = force_container or bool(problem.libs)
    if needs_container and container_check(docker_cmd=docker_cmd, runner=runner):
        return run_candidate_in_container(
            code,
            problem.test_code,
            problem.entry_point,
            image=image,
            timeout_s=timeout_s,
            docker_cmd=docker_cmd,
            runner=runner,
        )
    return run_candidate(code, problem.test_code, problem.entry_point, timeout_s=timeout_s)


def signature_candidate(
    code: str,
    problem: Labelable,
    *,
    timeout_s: float = 10.0,
    image: str = DEFAULT_IMAGE,
    force_container: bool = False,
    docker_cmd: str = "docker",
    runner: RunnerFn = subprocess.run,
    container_check: Callable[..., bool] = docker_available,
) -> frozenset[str] | None:
    """Compute a candidate's failing-test SIGNATURE via the right backend (the deconfounder's key).

    Same route-by-libs rule as ``label_candidate``: lib-bearing problems compute it in the
    container, stdlib-only in the in-process sandbox. None means undefined (treated as a non-match).
    """
    needs_container = force_container or bool(problem.libs)
    if needs_container and container_check(docker_cmd=docker_cmd, runner=runner):
        return run_signature_in_container(
            code,
            problem.test_code,
            problem.entry_point,
            image=image,
            timeout_s=timeout_s,
            docker_cmd=docker_cmd,
            runner=runner,
        )
    return run_failure_signature(code, problem.test_code, problem.entry_point, timeout_s=timeout_s)

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

from prism.eval.familygen import ProblemSpec
from prism.eval.sandbox import ExecOutcome, outcome_from_exit, run_candidate, write_runner_dir

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
        argv = [
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
            "-v", f"{d}:/work:ro",
            "-w", "/work",
            image,
            "python", "/work/runner.py",
        ]
        try:
            # docker gets a margin over the inner runner's wall-clock to start/stop.
            proc = runner(argv, capture_output=True, timeout=timeout_s + 30.0, check=False)
        except subprocess.TimeoutExpired:
            # Best-effort: stop the lingering container (client killed, not the engine).
            try:
                runner([docker_cmd, "kill", name], capture_output=True, timeout=20, check=False)
            except (OSError, subprocess.SubprocessError):
                pass
            return ExecOutcome(False, "timeout", f"container exceeded {timeout_s}s budget")
        return outcome_from_exit(proc.returncode, proc.stderr or b"")


def label_candidate(
    code: str,
    problem: ProblemSpec,
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

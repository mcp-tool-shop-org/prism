"""Family-provenanced, execution-labeled code corpus for the Lock-1 family-different A/B.

prism's first family-AB (v1.4.0) was a confounded NULL because the corpus artifacts were FIXED and
hand-authored — not produced by the caller families — so a same-family verifier had nothing of its
own to (mis)recognize and self-preference could not fire (the verifier saw identical artifacts
regardless of the family label). The study-swarm fix: have each family GENERATE the artifacts from a
shared set of post-cutoff problems, so a same-family judge sees its OWN family's style (the
self-recognition channel self-preference rides on — Panickssery et al. 2024, arXiv:2404.13076).

Pipeline per (problem, family):
  1. the family's model generates a solution from the shared problem spec (its own style);
  2. ``sandbox.run_candidate`` labels it by EXECUTION (PASS => clean, else buggy) — family-agnostic
     ground truth, no LLM in the label loop, which dodges the label-circularity threat;
  3. a CLEAN generation is additionally MUTATED (``mutate``) and each mutant is re-run; only mutants
     that now FAIL are kept — execution-verified buggy labels that still carry the family style;
  4. a FAILED generation is kept directly as a NATURAL buggy artifact.

The producing family is recorded in a PROVENANCE SIDECAR (the manifest's ``provenance`` map), never
in the frozen ``Sample`` schema — mirroring how ``corpus.py`` keeps the contamination flag out of
the schema. So these reuse ``Sample`` unchanged (and ``check_corpus_integrity`` /
``corpus_content_hash`` for free) while staying a SEPARATE corpus from the lens-calibration set: a
different concern, a different consumer (the A/B harness), materialized to its own directory.

This module EXECUTES model-generated code via ``sandbox`` — read that module's threat model.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from prism.core.types import ArtifactType
from prism.eval.container_sandbox import (
    docker_available,
    label_candidate,
    signature_candidate,
)
from prism.eval.corpus import (
    CONTENT_HASH_SCHEMA,
    Sample,
    check_corpus_integrity,
    corpus_content_hash,
)
from prism.eval.mutate import generate_mutants
from prism.eval.sandbox import ExecOutcome

DEFAULT_BASE_URL = "http://localhost:11434"

# An async (model_id, system_prompt, user_prompt) -> generated_text callable. The default hits
# Ollama; tests inject a deterministic fake so the whole pipeline runs offline at zero spend.
GenerateFn = Callable[[str, str, str], Awaitable[str]]

GEN_SYSTEM = (
    "You are an expert Python programmer. Implement the requested function exactly as specified. "
    "Return ONLY the function source code — no markdown fences, no explanation, no tests."
)

# The deconfounder restyle: a family re-expresses a FIXED buggy solution in its own style WITHOUT
# changing behavior, so the bug (its failing-test signature) is held constant across families and
# only the style varies — isolating self-preference from capability-driven difficulty (Tsui 2025).
RESTYLE_SYSTEM = (
    "You are an expert Python programmer. Rewrite the given function in your own idiomatic style "
    "WITHOUT changing its behavior in any way — preserve every output exactly, INCLUDING any bugs. "
    "Return ONLY the function source code — no markdown fences, no explanation, no tests."
)


def _restyle_prompt(code: str) -> str:
    return (
        "Rewrite the following function in your own style, preserving its EXACT behavior on every "
        "input — including any incorrect outputs / bugs (do NOT fix anything):\n\n" + code
    )


@dataclass(frozen=True)
class ProblemSpec:
    """One shared coding problem each family is asked to solve.

    ``test_code`` must define ``def check(candidate): ...`` raising ``AssertionError`` on a wrong
    result (the hidden, hardened suite that labels a generation clean/buggy by execution).
    """

    id: str
    intent: str
    entry_point: str
    test_code: str
    # Optional provenance / execution metadata for EXTERNALLY-sourced problems (LiveCodeBench,
    # BigCodeBench). Authored seed problems leave these at defaults (frozen interface stays
    # backward-compatible). ``libs`` = third-party imports the problem needs (lib-bearing problems
    # route to the containerized labeler); ``contest_date`` drives the per-family post-cutoff
    # contamination holdout; ``source`` is the provenance label.
    libs: tuple[str, ...] = ()
    contest_date: str | None = None
    difficulty: str = ""
    source: str = "authored"


@dataclass(frozen=True)
class FamilySpec:
    """A producer family: a provenance label + the model id used to generate its artifacts."""

    family: str  # provenance label, e.g. "mistral" / "qwen" / "granite"
    model_id: str  # the Ollama model id to generate with, e.g. "mistral-small:24b"


async def _ollama_generate(
    model_id: str,
    system: str,
    user: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    temperature: float = 0.2,
    seed: int | None = 0,
    max_tokens: int = 1024,
    timeout_s: float = 120.0,
) -> str:
    """Default generation backend: plain Ollama /api/chat, NO ``format:json`` (we want raw code)."""
    options: dict[str, Any] = {"temperature": temperature, "num_predict": max_tokens}
    if seed is not None:
        options["seed"] = seed
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "think": False,
                "options": options,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    message = data.get("message") if isinstance(data, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    return content if isinstance(content, str) else ""


def _default_generate(*, base_url: str, temperature: float, seed: int | None) -> GenerateFn:
    async def gen(model_id: str, system: str, user: str) -> str:
        return await _ollama_generate(
            model_id, system, user, base_url=base_url, temperature=temperature, seed=seed
        )

    return gen


def _gen_prompt(problem: ProblemSpec) -> str:
    return (
        f"Write a Python function named `{problem.entry_point}` that does the following:\n\n"
        f"{problem.intent}\n\n"
        f"Return only the function definition."
    )


def _extract_code(raw: str) -> str:
    """Strip a markdown code fence + language tag if the model wrapped its output in one."""
    s = raw.strip()
    if "```" not in s:
        return s
    after = s.split("```", 1)[1]
    block = after.split("```", 1)[0]
    lines = block.splitlines()
    if lines and lines[0].strip().lower() in {"python", "py", "python3"}:
        lines = lines[1:]
    return "\n".join(lines).strip()


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text)[:2000]


def _longest_common_run(a: list[str], b: list[str]) -> int:
    """Longest contiguous shared token run between two token lists (a cheap contamination probe)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def contamination_overlap(code: str, references: list[str]) -> int:
    """Max longest-common-token-run between ``code`` and any reference text (0 if no references).

    A high overlap with a known-public corpus means the family may have reproduced memorized text
    rather than generated its own — which would confound the family signal with shared memorization
    (the ConTAM longest-substring heuristic, Singh et al. 2024, arXiv:2411.03923).
    """
    if not references:
        return 0
    code_tokens = _tokens(code)
    return max((_longest_common_run(code_tokens, _tokens(r)) for r in references), default=0)


def _make_sample(
    sid: str,
    content: str,
    problem: ProblemSpec,
    *,
    positive: bool,
    bug_class: str,
    verdict: str,
    split: str,
) -> Sample:
    return Sample(
        id=sid,
        artifact_type=ArtifactType.CODE.value,
        content=content,
        intent=problem.intent,
        positive=positive,
        target_lens="invariant",  # nominal: the A/B scores the whole-artifact verdict, not per-lens
        bug_class=bug_class,
        expected_verdict=verdict,
        split=split,
    )


_UNRUNNABLE_MARKERS = ("ModuleNotFoundError", "ImportError", "No module named")


def _is_unrunnable(outcome: ExecOutcome) -> bool:
    """An ERROR whose stderr names a missing import: the env lacks the lib (label undefined).

    Such an artifact is SKIPPED (never labeled buggy) — a missing dependency is not a code defect.
    It's how a lib-bearing problem that couldn't reach the container (Docker absent) avoids
    corrupting the corpus with false bugs.
    """
    return outcome.status == "error" and any(m in outcome.detail for m in _UNRUNNABLE_MARKERS)


async def build_family_corpus(
    out_dir: Path,
    families: list[FamilySpec],
    *,
    problems: list[ProblemSpec] | None = None,
    split: str = "fresh",
    generate_fn: GenerateFn | None = None,
    mutants_per_clean: int = 2,
    timeout_s: float = 5.0,
    temperature: float = 0.2,
    seed: int | None = 0,
    base_url: str = DEFAULT_BASE_URL,
    contamination_refs: list[str] | None = None,
    contamination_max_run: int = 0,
    perplexity_fn: Callable[[str], float | None] | None = None,
    deconfound: bool = True,
) -> dict[str, object]:
    """Generate, execution-label, and materialize a family-provenanced code corpus + its manifest.

    Each family generates a solution per problem; a clean generation contributes a clean ``Sample``
    plus up to ``mutants_per_clean`` execution-verified mutant bugs; a failed generation contributes
    a NATURAL bug. On harder (LiveCodeBench/BigCodeBench) problems natural fails dominate the buggy
    stratum — the study-swarm's natural-majority target, since natural bugs carry the family's
    pattern-insistence synthetic mutants lack. ``generate_fn`` defaults to a local Ollama backend.

    Labeling routes through ``label_candidate``: lib-bearing problems (``ProblemSpec.libs``) run in
    the Docker labeler, stdlib-only in the in-process sandbox. An artifact that can't run (a missing
    dependency, ``ModuleNotFoundError``) is SKIPPED, never mislabeled buggy. ``perplexity_fn``
    (optional) logs a per-artifact fluency covariate (Wataoka 2024) — the self-preference channel
    rides on low perplexity, so it is a confound backstop beside the deconfounder stratum.

    ``contamination_max_run`` > 0 DROPS any artifact whose longest shared token run with
    ``contamination_refs`` exceeds it (0 = record-only, never drop). Returns the manifest; raises
    ``ValueError`` (ANDON) if the assembled corpus fails ``check_corpus_integrity``.
    """
    out_dir = Path(out_dir)
    if problems is None:
        from prism.eval._familygen_problems import FRESH_PROBLEMS

        problems = FRESH_PROBLEMS
    gen = generate_fn or _default_generate(base_url=base_url, temperature=temperature, seed=seed)
    refs = contamination_refs or []

    # Probe Docker ONCE (only when a problem needs it) and reuse the verdict for every label call —
    # avoids a `docker version` subprocess per artifact and keeps stdlib-only builds Docker-free.
    docker_ok = docker_available() if any(p.libs for p in problems) else False

    def _container_check(**_kwargs: object) -> bool:
        return docker_ok

    async def _label(code: str, problem: ProblemSpec) -> ExecOutcome:
        return await asyncio.to_thread(
            label_candidate, code, problem, timeout_s=timeout_s, container_check=_container_check
        )

    async def _signature(code: str, problem: ProblemSpec) -> frozenset[str] | None:
        return await asyncio.to_thread(
            signature_candidate,
            code,
            problem,
            timeout_s=timeout_s,
            container_check=_container_check,
        )

    model_of = {f.family: f.model_id for f in families}
    samples: list[Sample] = []
    provenance: dict[str, str] = {}
    problem_of: dict[str, str] = {}  # sample_id -> problem id (the A/B harness's bootstrap cluster)
    overlap_by_sample: dict[str, int] = {}
    perplexity_by_sample: dict[str, float] = {}
    per_family: dict[str, Counter[str]] = {}
    clean_gens: dict[str, dict[str, str]] = {}  # {problem_id: {family: clean code}} (deconfounder)

    def _record_perplexity(sid: str, code: str) -> None:
        if perplexity_fn is not None:
            value = perplexity_fn(code)
            if value is not None:
                perplexity_by_sample[sid] = value

    for fam in families:
        counter = per_family.setdefault(fam.family, Counter())
        slug = _slug(fam.family)
        for problem in problems:
            raw = await gen(fam.model_id, GEN_SYSTEM, _gen_prompt(problem))
            code = _extract_code(raw)
            if not code.strip():
                counter["empty"] += 1
                continue

            overlap = contamination_overlap(code, refs)
            if contamination_max_run and overlap > contamination_max_run:
                counter["dropped_contaminated"] += 1
                continue

            outcome = await _label(code, problem)
            if _is_unrunnable(outcome):
                counter["unrunnable"] += 1  # missing dependency — label undefined, skip
                continue

            if outcome.passed:
                sid = f"fam-{slug}-{problem.id}-gen-clean"
                samples.append(
                    _make_sample(
                        sid, code, problem, positive=False, bug_class="clean",
                        verdict="accept", split=split,
                    )
                )
                provenance[sid] = fam.family
                problem_of[sid] = problem.id
                overlap_by_sample[sid] = overlap
                _record_perplexity(sid, code)
                counter["clean"] += 1
                clean_gens.setdefault(problem.id, {})[fam.family] = code  # for the deconfounder

                kept = 0
                for mutant in generate_mutants(code, limit=mutants_per_clean * 3):
                    if kept >= mutants_per_clean:
                        break
                    m_outcome = await _label(mutant.code, problem)
                    if _is_unrunnable(m_outcome):
                        continue  # missing dependency — skip this mutant, don't mislabel
                    if not m_outcome.is_buggy:
                        continue  # equivalent mutant — no behavior change; never mislabel buggy
                    msid = f"fam-{slug}-{problem.id}-mut{kept}"
                    samples.append(
                        _make_sample(
                            msid, mutant.code, problem, positive=True,
                            bug_class=f"mut_{mutant.operator}", verdict="revise", split=split,
                        )
                    )
                    provenance[msid] = fam.family
                    problem_of[msid] = problem.id
                    overlap_by_sample[msid] = contamination_overlap(mutant.code, refs)
                    _record_perplexity(msid, mutant.code)
                    counter["mutant"] += 1
                    kept += 1
            else:
                sid = f"fam-{slug}-{problem.id}-gen-bug"
                samples.append(
                    _make_sample(
                        sid, code, problem, positive=True,
                        bug_class=f"natural_{outcome.status}", verdict="refuse", split=split,
                    )
                )
                provenance[sid] = fam.family
                problem_of[sid] = problem.id
                overlap_by_sample[sid] = overlap
                _record_perplexity(sid, code)
                counter["natural_bug"] += 1

    # --- Deconfounder stratum (faithful shared-bug restyle) ------------------------------------
    # For each problem >=2 families solved cleanly, hold ONE bug constant (identical failing-test
    # signature) across families and vary only the family STYLE. This isolates self-preference from
    # capability-driven bug difficulty (the critic's must-fix; Tsui 2025) — the estimator restricts
    # the contrast to bug_class=="deconfound" for a difficulty-matched read.
    if deconfound:
        problems_by_id = {p.id: p for p in problems}
        for problem_id, fam_clean in clean_gens.items():
            if len(fam_clean) < 2:
                continue  # need >=2 clean-producing families to vary style at a fixed bug
            problem = problems_by_id[problem_id]
            ordered = [f.family for f in families if f.family in fam_clean]
            # the FIXED canonical bug: first execution-verified buggy mutant of the canonical clean
            canonical_bug: str | None = None
            for mutant in generate_mutants(fam_clean[ordered[0]], limit=mutants_per_clean * 3):
                m_out = await _label(mutant.code, problem)
                if not _is_unrunnable(m_out) and m_out.is_buggy:
                    canonical_bug = mutant.code
                    break
            if canonical_bug is None:
                continue
            sig0 = await _signature(canonical_bug, problem)
            if not sig0:  # None or empty -> no usable same-bug key
                continue
            for family in ordered:
                raw = await gen(model_of[family], RESTYLE_SYSTEM, _restyle_prompt(canonical_bug))
                restyled = _extract_code(raw)
                if not restyled.strip():
                    continue
                r_out = await _label(restyled, problem)
                if _is_unrunnable(r_out) or not r_out.is_buggy:
                    continue
                if await _signature(restyled, problem) != sig0:
                    continue  # not the IDENTICAL failing-test set — a different bug; drop
                dsid = f"deconf-{_slug(problem_id)}-{_slug(family)}"
                samples.append(
                    _make_sample(
                        dsid, restyled, problem, positive=True, bug_class="deconfound",
                        verdict="revise", split=split,
                    )
                )
                provenance[dsid] = family
                problem_of[dsid] = problem_id
                _record_perplexity(dsid, restyled)
                per_family[family]["deconfound"] += 1

    integrity = check_corpus_integrity(samples)
    if integrity:
        raise ValueError(
            "family-AB corpus integrity check FAILED (ANDON halt): " + "; ".join(integrity[:5])
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}.jsonl").write_text(
        "".join(json.dumps(s.to_dict()) + "\n" for s in samples), encoding="utf-8"
    )

    manifest: dict[str, object] = {
        "schema": "prism-familyab-corpus/v1",
        "split": split,
        "families": [{"family": f.family, "model_id": f.model_id} for f in families],
        "generation": {"temperature": temperature, "seed": seed, "base_url": base_url},
        "n_problems": len(problems),
        "n_samples": len(samples),
        "n_positive": sum(1 for s in samples if s.positive),
        "counts_by_family": {fam: dict(c) for fam, c in per_family.items()},
        # PROVENANCE SIDECAR: sample_id -> producing family. The A/B harness groups by this to form
        # the within-judge self-vs-cross contrast; it lives here, NOT in the frozen Sample schema.
        "provenance": provenance,
        # sample_id -> problem id: the cluster the family-AB bootstrap resamples (items nested by
        # problem). Robust source for clustering (never parse it back out of the hyphenated id).
        "problem_of": problem_of,
        "contamination": {
            "refs_provided": len(refs),
            "max_run_drop_threshold": contamination_max_run,
            "per_sample_longest_run": overlap_by_sample,
        },
        # Per-artifact fluency covariate (Wataoka 2024): the self-preference channel rides on low
        # perplexity, so it is a confound backstop. Empty unless a perplexity_fn was passed.
        "perplexity": {"per_sample": perplexity_by_sample},
        "content_hash": corpus_content_hash(samples),
        "content_hash_schema": CONTENT_HASH_SCHEMA,
        "note": (
            "Family-provenanced, execution-labeled code corpus for the Lock-1 A/B. Each family "
            "GENERATED its artifacts (style provenance in 'provenance'); labels are "
            "execution-based (no LLM grader). Separate from the lens-calibration corpus."
        ),
    }
    (out_dir / "FAMILYAB_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest

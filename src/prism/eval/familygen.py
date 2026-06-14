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
from prism.eval.corpus import (
    CONTENT_HASH_SCHEMA,
    Sample,
    check_corpus_integrity,
    corpus_content_hash,
)
from prism.eval.mutate import generate_mutants
from prism.eval.sandbox import run_candidate

DEFAULT_BASE_URL = "http://localhost:11434"

# An async (model_id, system_prompt, user_prompt) -> generated_text callable. The default hits
# Ollama; tests inject a deterministic fake so the whole pipeline runs offline at zero spend.
GenerateFn = Callable[[str, str, str], Awaitable[str]]

GEN_SYSTEM = (
    "You are an expert Python programmer. Implement the requested function exactly as specified. "
    "Return ONLY the function source code — no markdown fences, no explanation, no tests."
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


async def build_family_corpus(
    out_dir: Path,
    families: list[FamilySpec],
    *,
    problems: list[ProblemSpec] | None = None,
    split: str = "fresh",
    generate_fn: GenerateFn | None = None,
    mutants_per_clean: int = 3,
    timeout_s: float = 5.0,
    temperature: float = 0.2,
    seed: int | None = 0,
    base_url: str = DEFAULT_BASE_URL,
    contamination_refs: list[str] | None = None,
    contamination_max_run: int = 0,
) -> dict[str, object]:
    """Generate, execution-label, and materialize a family-provenanced code corpus + its manifest.

    Each family generates a solution per problem; a clean generation contributes a clean ``Sample``
    plus up to ``mutants_per_clean`` execution-verified mutant bugs; a failed generation contributes
    a natural bug. ``generate_fn`` defaults to a local Ollama backend; inject a fake for tests.

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

    samples: list[Sample] = []
    provenance: dict[str, str] = {}
    problem_of: dict[str, str] = {}  # sample_id -> problem id (the A/B harness's bootstrap cluster)
    overlap_by_sample: dict[str, int] = {}
    per_family: dict[str, Counter[str]] = {}

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

            outcome = await asyncio.to_thread(
                run_candidate, code, problem.test_code, problem.entry_point, timeout_s=timeout_s
            )

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
                counter["clean"] += 1

                kept = 0
                for mutant in generate_mutants(code, limit=mutants_per_clean * 3):
                    if kept >= mutants_per_clean:
                        break
                    m_outcome = await asyncio.to_thread(
                        run_candidate, mutant.code, problem.test_code, problem.entry_point,
                        timeout_s=timeout_s,
                    )
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
                counter["natural_bug"] += 1

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

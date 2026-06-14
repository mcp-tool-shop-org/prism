"""Build the family-AB v3 HARDER corpus — the pre-registered run (design/08, RESULTS.md v3).

Generates with 4 distinct lineages over a post-cutoff difficulty-mix of LiveCodeBench-functional
problems, execution-labels, mutates, and builds the faithful identical-failing-tests deconfound
stratum. The cloud seats are served by the local Ollama daemon transparently.

    uv run python eval/run_familyab_v3.py
    prism eval --round-robin --familyab-corpus eval/corpus-familyab-v3 \
        --sesoi 0.05 --min-problems 20 --out eval/report/familyab-v3

Pinned config = the RESULTS.md "A/B v3 — PRE-REGISTRATION" block (locked before data).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from prism.eval.familygen import FamilySpec, build_family_corpus
from prism.eval.livecodebench import load_livecodebench
from prism.eval.problem_select import count_by_difficulty, stratified_by_difficulty

OUT = Path("eval/corpus-familyab-v3")
FAMILIES = [
    FamilySpec("gpt-oss", "gpt-oss:120b-cloud"),  # OpenAI lineage (cloud)
    FamilySpec("glm", "glm-4.6:cloud"),  # Zhipu lineage (cloud)
    FamilySpec("qwen", "qwen3-coder-next:cloud"),  # Qwen lineage (cloud)
    FamilySpec("mistral", "mistral-small:24b"),  # Mistral lineage (local)
]
PER_DIFFICULTY = {"easy": 14, "medium": 10, "hard": 4}
START_DATE = "2025-01-01"  # post-cutoff window for the participating families


async def main() -> None:
    pool = load_livecodebench(version="release_v6", start_date=START_DATE, limit=300)
    print(f"LCB pool ({START_DATE}+): {len(pool)} functional | {count_by_difficulty(pool)}")
    problems = stratified_by_difficulty(pool, PER_DIFFICULTY, seed=0)
    print(f"selected: {len(problems)} problems | {count_by_difficulty(problems)}")
    if not problems:
        raise SystemExit("no problems selected — widen START_DATE or the difficulty mix")
    manifest = await build_family_corpus(
        OUT, FAMILIES, problems=problems, deconfound=True, mutants_per_clean=2, timeout_s=12.0
    )
    print(f"\nn_samples={manifest['n_samples']} n_positive={manifest['n_positive']}")
    print(f"deconfound_eligible_problems={manifest['deconfound_eligible_problems']}")
    print("counts_by_family:")
    print(json.dumps(manifest["counts_by_family"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())

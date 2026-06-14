"""Difficulty-stratified problem selection for the family-AB corpus build.

The real-model smoke surfaced a composition constraint: the faithful deconfounder needs >=2 families
to produce a CLEAN solution on the SAME problem (to hold one bug constant across styles), but on the
hardest problems local 24-32B models mostly FAIL — great for natural-bug variance, useless for the
deconfound stratum. So the run wants a difficulty MIX: enough easy/medium problems that several
families succeed (populating the deconfound stratum) PLUS hard problems (the natural-bug variance
regime self-preference rides on). This module composes that mix deterministically from a pool.
"""

from __future__ import annotations

import random
from collections import defaultdict

from prism.eval.familygen import ProblemSpec


def count_by_difficulty(problems: list[ProblemSpec]) -> dict[str, int]:
    """Histogram of a problem pool by difficulty tag (``""`` for untagged, e.g. BigCodeBench)."""
    counts: dict[str, int] = defaultdict(int)
    for p in problems:
        counts[p.difficulty] += 1
    return dict(counts)


def stratified_by_difficulty(
    problems: list[ProblemSpec], per_difficulty: dict[str, int], *, seed: int = 0
) -> list[ProblemSpec]:
    """Select up to ``per_difficulty[d]`` problems of each difficulty ``d`` — a deterministic mix.

    Groups the pool by ``ProblemSpec.difficulty``; for each requested difficulty takes that many (a
    seeded sample over the id-sorted pool — reproducible, and spread across the pool rather than
    clustered on the first ids). A difficulty absent from ``per_difficulty`` contributes nothing; a
    count >= the pool size takes the whole bucket. E.g. ``{"easy": 25, "medium": 30, "hard": 20}``
    weights toward deconfound-eligibility while keeping a hard tail for natural-bug variance.
    """
    buckets: dict[str, list[ProblemSpec]] = defaultdict(list)
    for p in problems:
        buckets[p.difficulty].append(p)
    rng = random.Random(seed)
    selected: list[ProblemSpec] = []
    for difficulty in sorted(per_difficulty):
        n = per_difficulty[difficulty]
        pool = sorted(buckets.get(difficulty, []), key=lambda p: p.id)
        if n <= 0 or not pool:
            continue
        selected.extend(pool if n >= len(pool) else rng.sample(pool, n))
    return selected

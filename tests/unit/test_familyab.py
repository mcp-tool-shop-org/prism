"""Tests for the within-judge round-robin self-preference estimator (the valid Lock-1 design)."""

from __future__ import annotations

from prism.eval.corpus import Sample
from prism.eval.familyab import (
    JudgeRecord,
    collapse_verdict,
    compute_self_preference,
    run_round_robin,
)


def test_collapse_verdict() -> None:
    assert collapse_verdict("refuse") == "refute"
    assert collapse_verdict("revise") == "refute"
    assert collapse_verdict("escalate") == "abstain"
    assert collapse_verdict("accept") == "accept"
    assert collapse_verdict("???") == "abstain"  # unknown -> conservative abstain


def _build(n_problems: int = 6) -> list[JudgeRecord]:
    """Verifier A self-prefers (accepts its OWN family's bugs, refutes others');
    verifier B is a clean judge (refutes all bugs); verifier C always abstains (can't discriminate).
    """
    recs: list[JudgeRecord] = []
    n = 0
    for i in range(n_problems):
        p = f"p{i}"

        def rec(vf: str, pf: str, buggy: bool, verdict: str) -> JudgeRecord:
            nonlocal n
            n += 1
            return JudgeRecord(vf, pf, p, f"s{n}", buggy, verdict)

        # A: false-accepts its OWN family's bugs, refutes OTHER family's bugs; accepts clean.
        recs += [
            rec("A", "A", True, "accept"),
            rec("A", "B", True, "refuse"),
            rec("A", "A", False, "accept"),
            rec("A", "B", False, "accept"),
        ]
        # B: refutes every bug regardless of producer; accepts clean. No self-preference.
        recs += [
            rec("B", "B", True, "refuse"),
            rec("B", "A", True, "refuse"),
            rec("B", "B", False, "accept"),
            rec("B", "A", False, "accept"),
        ]
        # C: escalates everything -> never refutes a bug -> can't discriminate (bal-acc 0.5).
        recs += [
            rec("C", "C", True, "escalate"),
            rec("C", "A", True, "escalate"),
            rec("C", "C", False, "escalate"),
            rec("C", "A", False, "escalate"),
        ]
    return recs


def _family(result, name):  # type: ignore[no-untyped-def]
    return next(sp for sp in result.per_family if sp.verifier_family == name)


def test_self_preferring_family_has_positive_delta() -> None:
    result = compute_self_preference(_build())
    a = _family(result, "A")
    assert a.false_accept_own == 1.0
    assert a.false_accept_other == 0.0
    assert a.self_preference == 1.0  # accepts its own family's bugs, refutes others'
    assert a.interpretable is True


def test_clean_judge_has_zero_delta() -> None:
    b = _family(compute_self_preference(_build()), "B")
    assert b.self_preference == 0.0
    assert b.balanced_accuracy == 1.0
    assert b.interpretable is True


def test_non_discriminating_family_is_excluded_by_the_floor() -> None:
    c = _family(compute_self_preference(_build()), "C")
    assert c.balanced_accuracy < 0.6
    assert c.floor_pass is False
    assert c.interpretable is False


def test_aggregate_is_mean_over_interpretable_families_only() -> None:
    result = compute_self_preference(_build())
    assert result.n_families == 3
    assert result.n_interpretable_families == 2  # A and B; C excluded
    assert result.aggregate_self_preference == 0.5  # mean(1.0, 0.0)


def test_aggregate_ci_is_deterministic_and_consistent() -> None:
    r1 = compute_self_preference(_build(), n_boot=300, seed=3)
    r2 = compute_self_preference(_build(), n_boot=300, seed=3)
    assert r1.aggregate_ci == r2.aggregate_ci
    lo, hi = r1.aggregate_ci
    assert lo <= 0.5 <= hi


def test_only_own_family_is_uninterpretable() -> None:
    # a verifier with no OTHER-family buggy items cannot form the contrast
    recs = [
        JudgeRecord("X", "X", "p0", "s0", True, "accept"),
        JudgeRecord("X", "X", "p0", "s1", False, "accept"),
    ]
    sp = compute_self_preference(recs).per_family[0]
    assert sp.n_other_buggy == 0
    assert sp.interpretable is False


def _sample(sid: str, positive: bool) -> Sample:
    return Sample(
        id=sid,
        artifact_type="code",
        content="x",
        intent="i",
        positive=positive,
        target_lens="invariant",
        bug_class="bug" if positive else "clean",
        expected_verdict="refuse" if positive else "accept",
        split="fresh",
    )


async def test_round_robin_runner_to_estimate() -> None:
    samples: list[Sample] = []
    provenance: dict[str, str] = {}
    problem_of: dict[str, str] = {}
    for producer in ("A", "B"):
        for problem in ("p0", "p1"):
            for kind, positive in (("bug", True), ("clean", False)):
                sid = f"{producer}-{problem}-{kind}"
                samples.append(_sample(sid, positive))
                provenance[sid] = producer
                problem_of[sid] = problem

    async def judge(verifier: str, sample: Sample) -> str:
        if not sample.positive:
            return "accept"
        producer = provenance[sample.id]
        if verifier == "A":
            return "accept" if producer == "A" else "refuse"  # A false-accepts its own bugs
        return "refuse"  # B refutes every bug regardless of producer

    records = await run_round_robin(samples, provenance, problem_of, ["A", "B"], judge)
    assert len(records) == 2 * len(samples)  # each verifier judges every artifact
    probe = next(r for r in records if r.verifier_family == "A" and r.sample_id == "A-p0-bug")
    assert probe.producer_family == "A"
    assert probe.problem_id == "p0"
    assert probe.ground_truth_buggy is True

    result = compute_self_preference(records)
    a = next(sp for sp in result.per_family if sp.verifier_family == "A")
    b = next(sp for sp in result.per_family if sp.verifier_family == "B")
    assert a.self_preference == 1.0
    assert b.self_preference == 0.0
    assert result.aggregate_self_preference == 0.5

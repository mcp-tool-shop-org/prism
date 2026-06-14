"""Unit tests for ``report.summarize`` aggregation — known-answer over hand-built EvalRuns.

These pin the EVL-A-001 invariant: records the engine could not adjudicate (a structural
``VerifyError`` → ``unavailable=True``) must NOT corrupt the published calibration numbers
(ECE / Brier / mean-confidence / modal verdict / verdict-accuracy). A provider outage during
``prism eval`` should be *disclosed and excluded*, never silently folded into the metrics as a
confidence-0.0 wrong answer.
"""

from __future__ import annotations

import math

from prism.eval.corpus import Sample
from prism.eval.metrics import brier_score, expected_calibration_error
from prism.eval.report import summarize
from prism.eval.runner import EvalRun, RunRecord


def _sample(sid: str, *, positive: bool, target_lens: str = "invariant") -> Sample:
    return Sample(
        id=sid,
        artifact_type="code",
        content="def f():\n    return 1\n",
        intent="return one",
        positive=positive,
        target_lens=target_lens,
        bug_class="off_by_one" if positive else "clean",
        expected_verdict="refuse" if positive else "accept",
        split="public",
    )


def _ok_record(sid: str, run_index: int, *, verdict: str, confidence: float) -> RunRecord:
    """A genuine, available adjudication (engine returned a real verdict)."""
    return RunRecord(
        sample_id=sid,
        run_index=run_index,
        verdict=verdict,
        confidence=confidence,
        errored=False,
        unavailable=False,
        per_lens={"invariant": "fail" if verdict != "accept" else "pass"},
        pairwise_rho={},
    )


def _unavailable_record(sid: str, run_index: int) -> RunRecord:
    """A structural VerifyError placeholder — no real verdict (verdict=reason, conf 0.0)."""
    return RunRecord(
        sample_id=sid,
        run_index=run_index,
        verdict="provider_unavailable",  # a RefusalReason value, NOT a real verdict
        confidence=0.0,
        errored=True,
        unavailable=True,
        per_lens={},
        pairwise_rho={},
    )


def _citation_sample(sid: str, *, positive: bool) -> Sample:
    """A citations-artifact sample targeting the 'citation' lens (the per-lens-table-only lens)."""
    return Sample(
        id=sid,
        artifact_type="citations",
        content='[{"identifier": "2402.01817", "claim": "x"}]',
        intent="verify each citation exists and the stated finding matches the source",
        positive=positive,
        target_lens="citation",
        bug_class="fabricated" if positive else "real",
        expected_verdict="refuse" if positive else "accept",
        split="public",
    )


def _citation_record(sid: str, run_index: int, *, verdict: str, confidence: float) -> RunRecord:
    """A genuine citation adjudication: the engine emits a 'citation' lens result (engine.py)."""
    return RunRecord(
        sample_id=sid,
        run_index=run_index,
        verdict=verdict,
        confidence=confidence,
        errored=False,
        unavailable=False,
        per_lens={"citation": "fail" if verdict != "accept" else "pass"},
        pairwise_rho={},
    )


def _full4_record(sid: str, run_index: int, *, verdict: str, confidence: float) -> RunRecord:
    """A code/tool adjudication with a decision on all four _LLM_LENSES (feeds diversity)."""
    fail = "fail" if verdict != "accept" else "pass"
    return RunRecord(
        sample_id=sid,
        run_index=run_index,
        verdict=verdict,
        confidence=confidence,
        errored=False,
        unavailable=False,
        per_lens={
            "contract_completeness": fail,
            "cross_boundary": fail,
            "invariant": fail,
            "groundedness": fail,
        },
        pairwise_rho={},
    )


def _run(samples: list[Sample], records: list[RunRecord], n_runs: int) -> EvalRun:
    return EvalRun(
        records=records,
        samples={s.id: s for s in samples},
        n_runs=n_runs,
        verifier_label="ollama-known-answer",  # NOT mock/offline, so no mock note pollutes notes
        caller_family="anthropic",
    )


class TestReproducibilityProvenance:
    """EVS-B-001/002: the report must NAME which model(s) produced the numbers, at what temperature,
    with which seed, over which corpus content-hash — so a published artifact is reproducible by
    construction rather than asserted."""

    def test_report_carries_resolved_models_temperature_seed_and_corpus_hash(self) -> None:
        from prism.eval.report import render_json, render_markdown

        s = _sample("s1", positive=True)
        records = [_ok_record("s1", 0, verdict="refuse", confidence=0.8)]
        run = EvalRun(
            records=records,
            samples={s.id: s},
            n_runs=1,
            verifier_label="ollama",
            caller_family="anthropic",
            resolved_model_ids=["qwen3:14b", "mistral:7b"],
            effective_temperature=0.0,
            seed=1234,
            corpus_content_hash="deadbeef" * 8,
        )
        report = summarize(run)
        assert report.resolved_model_ids == ["mistral:7b", "qwen3:14b"]  # deduped + sorted
        assert math.isclose(report.effective_temperature, 0.0)
        assert report.seed == 1234
        assert report.corpus_content_hash == "deadbeef" * 8

        md = render_markdown(report)
        assert "qwen3:14b" in md and "mistral:7b" in md
        assert "temp" in md.lower()
        assert ("deadbeef" * 8)[:12] in md  # at least the short corpus-hash prefix is shown
        assert all(ord(c) < 128 for c in md)  # ASCII-safe
        import json as _json

        _json.loads(render_json(report))

    def test_report_handles_unrecorded_provenance_gracefully(self) -> None:
        """An EvalRun built without provenance (older callers / tests) renders without crashing."""
        from prism.eval.report import render_markdown

        s = _sample("s1", positive=True)
        run = _run([s], [_ok_record("s1", 0, verdict="refuse", confidence=0.8)], n_runs=1)
        report = summarize(run)
        assert report.resolved_model_ids == []
        assert report.effective_temperature is None
        assert report.seed is None
        # Markdown still renders an honest "not recorded" rather than crashing.
        md = render_markdown(report)
        assert all(ord(c) < 128 for c in md)


class TestPrevalenceCaveat:
    """EVS-B-003: precision is reported raw at the corpus's balanced ~50% prevalence. The report
    must SAY so (an explicit caveat), not let a reader mistake it for deployment precision."""

    def test_precision_prevalence_caveat_present_when_a_lens_has_precision(self) -> None:
        # A measured positive + clean pair gives the lens a precision number to caveat.
        bug = _sample("b", positive=True)
        clean = _sample("c", positive=False)
        records = [
            _ok_record("b", 0, verdict="refuse", confidence=0.8),
            _ok_record("c", 0, verdict="accept", confidence=0.8),
        ]
        report = summarize(_run([bug, clean], records, n_runs=1))
        assert any(
            "precision" in n.lower() and "prevalence" in n.lower() for n in report.notes
        ), f"missing prevalence caveat on precision: {report.notes}"


class TestUnavailableExcludedFromMetrics:
    def test_partial_unavailable_uses_genuine_records_only(self) -> None:
        """A positive sample: 2 genuine correct flags @0.9 + 1 unavailable placeholder.

        Known answer (genuine-only): modal verdict = 'refuse' (correct, since positive), mean
        confidence = 0.9, ECE/Brier computed at conf 0.9 with correct=True.

        WITHOUT the fix the unavailable record (verdict=reason, conf 0.0) is folded in: mean
        confidence drops to (0.9+0.9+0.0)/3 = 0.6 and — worse — for a single-sample run the modal
        verdict can tip and the calibration point shifts, inflating ECE. WITH the fix the metrics
        match the genuine-only computation exactly.
        """
        sample = _sample("s-bug", positive=True)
        records = [
            _ok_record("s-bug", 0, verdict="refuse", confidence=0.9),
            _ok_record("s-bug", 1, verdict="refuse", confidence=0.9),
            _unavailable_record("s-bug", 2),
        ]
        report = summarize(_run([sample], records, n_runs=3))

        # Genuine-only known answer: one sample, correct flag, confidence 0.9.
        expected_ece = expected_calibration_error([0.9], [True])
        expected_brier = brier_score([0.9], [True])

        assert math.isclose(report.ece, expected_ece), f"ECE corrupted: {report.ece}"
        assert math.isclose(report.brier, expected_brier), f"Brier corrupted: {report.brier}"
        # Verdict accuracy: the genuine modal verdict is 'refuse' on a positive => correct => 1.0.
        assert math.isclose(report.verdict_accuracy_overall, 1.0)
        # A disclosure note MUST be present (silent exclusion is as bad as silent inclusion).
        assert any("unavailable" in n.lower() and "excluded" in n.lower() for n in report.notes), (
            f"no disclosure note for excluded records: {report.notes}"
        )

    def test_all_unavailable_sample_dropped_not_counted_wrong(self) -> None:
        """A sample whose every record is unavailable has NO measurement.

        It must be excluded from verdict-accuracy/ECE/Brier entirely — not scored as wrong. Here a
        clean sample is fully measured (correct, conf 0.95) and a second sample is fully
        unavailable. The metrics must equal the single genuine sample's, NOT be halved by counting
        the unavailable sample as a 0.0-confidence wrong answer.
        """
        good = _sample("s-clean", positive=False)
        dead = _sample("s-dead", positive=True)
        records = [
            _ok_record("s-clean", 0, verdict="accept", confidence=0.95),
            _unavailable_record("s-dead", 0),
        ]
        report = summarize(_run([good, dead], records, n_runs=1))

        expected_ece = expected_calibration_error([0.95], [True])
        expected_brier = brier_score([0.95], [True])

        assert math.isclose(report.ece, expected_ece)
        assert math.isclose(report.brier, expected_brier)
        # Only the genuine clean sample is scored; it is correct => 1.0 (not 0.5 counting the dead).
        assert math.isclose(report.verdict_accuracy_overall, 1.0)
        assert any("unavailable" in n.lower() for n in report.notes)

    def test_no_unavailable_no_disclosure_note(self) -> None:
        """When nothing is excluded, no spurious unavailability note is emitted."""
        s = _sample("s1", positive=True)
        records = [_ok_record("s1", 0, verdict="refuse", confidence=0.8)]
        report = summarize(_run([s], records, n_runs=1))
        assert not any("unavailable" in n.lower() for n in report.notes)

    def test_errored_but_available_record_is_kept(self) -> None:
        """An ``errored`` record that still carries a real verdict+confidence is genuine: kept.

        A lens fault where the engine still returned a verdict is a real adjudication; only
        ``unavailable`` (structural VerifyError) drops out. Here both records are genuine (one
        flagged errored=True but available), so both confidences feed the mean.
        """
        s = _sample("s1", positive=True)
        records = [
            RunRecord(
                sample_id="s1",
                run_index=0,
                verdict="refuse",
                confidence=0.9,
                errored=True,  # a lens fault, but the engine still produced a verdict
                unavailable=False,
                per_lens={"invariant": "fail"},
                pairwise_rho={},
            ),
            _ok_record("s1", 1, verdict="refuse", confidence=0.7),
        ]
        report = summarize(_run([s], records, n_runs=2))
        # Mean confidence over BOTH genuine records: (0.9 + 0.7) / 2 = 0.8.
        expected_ece = expected_calibration_error([0.8], [True])
        assert math.isclose(report.ece, expected_ece)


class TestCitationMeasuredByVerdictNotLensFail:
    """F-01 fix: 'citation' must NOT appear in the per-lens fail-recall table. The citation pipeline
    maps REFUSE/REVISE -> FAIL but ESCALATE -> UNCERTAIN (engine._citation_as_lensresult), and the
    live oracle ESCALATES many citation positives, so a fail-recall row is structurally ~0 and
    misleading (a real run showed 0.000 recall while citation VERDICT accuracy was 0.667). Citations
    are measured by VERDICT instead — off-accept recall (escalate counts) + a verdict breakdown."""

    def test_per_lens_table_has_no_citation_row(self) -> None:
        """The per-lens table shows ONLY the 4 LLM lenses — never a 'citation' fail-recall row."""
        from prism.eval.report import render_markdown

        bug = _citation_sample("cit-bug", positive=True)
        clean = _citation_sample("cit-clean", positive=False)
        records = [
            _citation_record("cit-bug", 0, verdict="refuse", confidence=0.8),
            _citation_record("cit-clean", 0, verdict="accept", confidence=0.8),
        ]
        report = summarize(_run([bug, clean], records, n_runs=1))
        lenses = {lr.lens for lr in report.per_lens}
        assert "citation" not in lenses, f"citation leaked into per-lens table: {lenses}"
        # Only the 4 LLM lenses may ever appear in the per-lens table.
        assert lenses <= {"contract_completeness", "cross_boundary", "invariant", "groundedness"}
        md = render_markdown(report)
        assert "| citation |" not in md  # no fail-recall row rendered

    def test_diversity_matrix_unchanged_uses_only_the_four_llm_lenses(self) -> None:
        """A full-4-lens code sample + a citation sample: the diversity matrix must be built from
        the 4 _LLM_LENSES only — the citation sample (no 4-lens decision) must NOT enter it, and the
        pairwise-kappa keys must never mention 'citation'."""
        code = _sample("code-bug", positive=True, target_lens="invariant")
        cit = _citation_sample("cit-bug", positive=True)
        records = [
            _full4_record("code-bug", 0, verdict="refuse", confidence=0.8),
            _citation_record("cit-bug", 0, verdict="refuse", confidence=0.8),
        ]
        report = summarize(_run([code, cit], records, n_runs=1))
        # citation is NOT in the per-lens quality table...
        assert "citation" not in {lr.lens for lr in report.per_lens}
        # ...and NEVER in the diversity matrix (kappa pairs are only among the 4 LLM lenses).
        for pair in report.pairwise_kappa:
            assert "citation" not in pair, f"diversity matrix leaked citation: {pair}"
        # The Krippendorff units came from the single full-4-lens sample, so alpha is computable
        # without the citation sample polluting it.
        llm = {"contract_completeness", "cross_boundary", "invariant", "groundedness"}
        for pair in report.pairwise_kappa:
            a, b = pair.split(",")
            assert a in llm and b in llm

    def test_escalated_positive_citation_counts_as_flagged_in_offaccept_recall(self) -> None:
        """THE EXACT BUG: a positive citation whose verdict is ESCALATE must count as CAUGHT in
        off-accept recall (escalate != fail, but escalate == not-blindly-accepted). A positive
        citation that is ACCEPTed is a MISS. Here: 1 escalated positive + 1 accepted positive =>
        off-accept recall 0.5 (the escalate is flagged, the accept is the miss)."""
        from prism.eval.report import render_markdown

        escalated = _citation_sample("cit-escalate", positive=True)
        accepted = _citation_sample("cit-accept", positive=True)
        records = [
            _citation_record("cit-escalate", 0, verdict="escalate", confidence=0.5),
            _citation_record("cit-accept", 0, verdict="accept", confidence=0.5),
        ]
        report = summarize(_run([escalated, accepted], records, n_runs=1))
        assert report.citation_n == 2
        assert report.citation_positives == 2
        # escalate is flagged, accept is a miss => 1/2.
        assert report.citation_offaccept_recall == 0.5
        # Verdict breakdown counts both.
        assert report.citation_verdict_breakdown["escalate"] == 1
        assert report.citation_verdict_breakdown["accept"] == 1
        md = render_markdown(report)
        assert "## Citation pipeline" in md
        assert "escalate=1" in md and "accept=1" in md
        assert "VERDICT" in md  # the explanatory note is present

    def test_citation_offaccept_recall_na_when_no_positives(self) -> None:
        """No positive citation samples => off-accept recall is None (guarded), rendered n/a."""
        from prism.eval.report import render_markdown

        clean = _citation_sample("cit-clean", positive=False)
        records = [_citation_record("cit-clean", 0, verdict="accept", confidence=0.8)]
        report = summarize(_run([clean], records, n_runs=1))
        assert report.citation_n == 1
        assert report.citation_positives == 0
        assert report.citation_offaccept_recall is None
        md = render_markdown(report)
        assert "## Citation pipeline" in md
        assert "n/a (no positives)" in md


class TestContaminationAccuracyDelta:
    """F-01 fix B: the report must compute a contaminated(public/QuixBugs)-vs-uncontaminated VERDICT
    accuracy DELTA — contaminated is the CEILING (verifiers may have memorized QuixBugs),
    uncontaminated is the honest signal. Div-by-zero guarded when a side is empty."""

    def test_split_computed_on_mixed_fixture_known_answer(self) -> None:
        from prism.eval.report import render_markdown

        # Two contaminated (quixbugs-) samples: one correct, one wrong => contaminated acc 0.5.
        # Two authored samples: both correct => uncontaminated acc 1.0. delta = 1.0 - 0.5 = 0.5.
        quix_ok = Sample(
            id="quixbugs-gcd-buggy",
            artifact_type="code",
            content="def gcd(a, b):\n    return gcd(a % b, b)\n",
            intent="Return the greatest common divisor of a and b.",
            positive=True,
            target_lens="invariant",
            bug_class="wrong_recursive_args",
            expected_verdict="revise",
            split="public",
        )
        quix_wrong = Sample(
            id="quixbugs-bitcount-buggy",
            artifact_type="code",
            content="def bitcount(n):\n    count = 0\n    while n:\n        n ^= n - 1\n    return count\n",  # noqa: E501
            intent="Count the set bits in n.",
            positive=True,
            target_lens="invariant",
            bug_class="wrong_op",
            expected_verdict="revise",
            split="public",
        )
        authored_a = _sample("authored-a", positive=True)
        authored_b = _sample("authored-b", positive=True)
        records = [
            _ok_record("quixbugs-gcd-buggy", 0, verdict="refuse", confidence=0.8),  # correct
            _ok_record("quixbugs-bitcount-buggy", 0, verdict="accept", confidence=0.8),  # WRONG
            _ok_record("authored-a", 0, verdict="refuse", confidence=0.8),  # correct
            _ok_record("authored-b", 0, verdict="revise", confidence=0.8),  # correct (off-accept)
        ]
        report = summarize(
            _run([quix_ok, quix_wrong, authored_a, authored_b], records, n_runs=1)
        )
        assert report.contaminated_n == 2
        assert report.uncontaminated_n == 2
        assert report.contaminated_accuracy == 0.5
        assert report.uncontaminated_accuracy == 1.0
        assert math.isclose(report.contamination_accuracy_delta, 0.5)
        md = render_markdown(report)
        assert "[CEILING]" in md and "[honest]" in md

    def test_div_by_zero_guarded_when_a_side_is_empty(self) -> None:
        """No contaminated samples at all => contaminated side is n/a (None), delta is None."""
        from prism.eval.report import render_markdown

        s = _sample("authored-only", positive=True)
        records = [_ok_record("authored-only", 0, verdict="refuse", confidence=0.8)]
        report = summarize(_run([s], records, n_runs=1))
        assert report.contaminated_n == 0
        assert report.contaminated_accuracy is None
        assert report.uncontaminated_n == 1
        assert report.uncontaminated_accuracy == 1.0
        assert report.contamination_accuracy_delta is None
        md = render_markdown(report)
        assert "n/a" in md  # the contaminated side renders n/a


class TestContaminationCaveat:
    """F-01 v1.1 2D: when the corpus contains contaminated (QuixBugs/public) samples, the report
    must print a caveat that the public-split number is a CEILING (verifiers may have memorized) and
    the fresh split is the honest signal. Mirrors the existing prevalence-caveat note style."""

    def test_caveat_present_when_a_contaminated_sample_is_scored(self) -> None:
        from prism.eval.report import render_markdown

        # A 'quixbugs-' id marks a contaminated (known-public) sample.
        s = Sample(
            id="quixbugs-gcd-buggy",
            artifact_type="code",
            content="def gcd(a, b):\n    return gcd(a % b, b)\n",
            intent="Return the greatest common divisor of a and b.",
            positive=True,
            target_lens="invariant",
            bug_class="wrong_recursive_args",
            expected_verdict="revise",
            split="public",
        )
        records = [_ok_record("quixbugs-gcd-buggy", 0, verdict="revise", confidence=0.8)]
        report = summarize(_run([s], records, n_runs=1))
        assert report.contaminated_sample_count == 1
        assert any(
            "ceiling" in n.lower() and ("contaminat" in n.lower() or "memoriz" in n.lower())
            for n in report.notes
        ), f"missing contamination caveat: {report.notes}"
        md = render_markdown(report)
        assert "CEILING" in md

    def test_no_caveat_when_no_contaminated_samples(self) -> None:
        s = _sample("clean-authored", positive=True)
        records = [_ok_record("clean-authored", 0, verdict="refuse", confidence=0.8)]
        report = summarize(_run([s], records, n_runs=1))
        assert report.contaminated_sample_count == 0
        assert not any("ceiling" in n.lower() for n in report.notes)

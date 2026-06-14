# Prism calibration benchmark — RESULTS

The snapshot the README/landing scorecard cites **from actual numbers**, produced by `prism eval`
over `eval/corpus/`. Methodology + research grounding:
[`design/07-slice1-calibration.md`](../design/07-slice1-calibration.md).

**Run:** 2026-06-14 · verifier **local Ollama `mistral-small:24b`** (caller family `anthropic`;
Lock 1 routes verification to the local family) · `--split all` (111 samples: code 78, tool_call
18, citations 15 — now includes the vendored MIT QuixBugs real-bug pairs) · `--runs 1`. Reproduce:
`prism eval --split all --runs 1 --out eval/report` (needs Ollama up with `mistral-small:24b`,
`PRISM_DEV=1` for a local signing key).

> **v1 corpus is still small** (min 4 positives/lens « 100): recall CIs are wide (shown below) and
> the headline findings are *directional*, not statistically tight. A larger fresh (post-cutoff)
> split is the next upgrade. These are real measurements, not mock.

## Headline

| Metric | Value |
|---|---|
| Verifier | local Ollama `mistral-small:24b` |
| Per-lens MCC (contract / cross_boundary / invariant / groundedness) | **0.33 / 0.71 / 0.48 / 0.78** |
| Krippendorff alpha (lens decisions) | **0.162** — lenses are fairly independent at the decision level overall… |
| …but contract↔invariant Cohen kappa | **0.717** — those two lenses are strongly *correlated* (the real redundancy) |
| Union coverage recall / coverage gain | **1.00 / 0** — the union does NOT beat the best single lens (invariant) here |
| Data-calibrated rho operating point | **n/a** — finding-set rho is 0.0 for every pair (degenerate sweep) |
| Same-family A/B delta (Lock 1) | **−0.030, 95% CI [−0.19, +0.13]** (n=33; family-different 0.697 vs same-family 0.727) — a confounded null; see the section below |
| Overall verdict accuracy / ECE / Brier | **0.667** / 0.241 / 0.269 |
| Contaminated (public/QuixBugs) vs uncontaminated accuracy | **0.560 (n=50) [ceiling] vs 0.754 (n=61) [honest]; delta 0.194** |

## Per-lens quality (on each lens's target class)

| Lens | n | pos | recall (95% CI) | precision | specificity | MCC | bal-acc |
|---|---|---|---|---|---|---|---|
| contract_completeness | 22 | 11 | 0.909 [0.62, 0.98] | 0.588 | 0.364 | 0.325 | 0.636 |
| cross_boundary | 18 | 9 | 1.000 [0.70, 1.00] | 0.750 | 0.667 | 0.707 | 0.833 |
| invariant | 48 | 24 | 1.000 [0.86, 1.00] | 0.615 | 0.375 | 0.480 | 0.688 |
| groundedness | 8 | 4 | 0.750 [0.30, 0.95] | 1.000 | 1.000 | 0.775 | 0.875 |

Verdict accuracy by class: **code 0.705**, citations 0.667, **tool_call 0.500**. Cross-run consistency 1.000.

## Citation pipeline (measured by verdict, not lens-fail)

Citations are not a fail/pass LLM lens — the pipeline **escalates** an unconfirmable citation rather
than emitting a `FAIL`, so it is measured by VERDICT, not the per-lens fail-recall table above.

- **Off-accept recall (a bad citation is NOT blindly accepted): 1.000** (10 positive citation samples)
- Citation verdict accuracy: **0.667** (n=15)
- Verdict breakdown: **accept=0, revise=8, refuse=0, escalate=7**

**Finding:** against the *live* arXiv/Crossref oracle + mistral groundedness, the citation pipeline
**never emits a clean ACCEPT** — it catches every bad citation (off-accept recall 1.00, the safe
direction) but also flags all 5 *clean* citations (revise/escalate), dragging citation verdict
accuracy to 0.667. The over-escalation is the citation analogue of the `cross_boundary`/`contract`
over-flagging below; the live oracle's rate-limit/abstain behavior pushes borderline-clean citations
to ESCALATE. A precision pass on the citation-clean path is the clearest citation-quality fix.

## Findings (what the measurement actually surfaced)

1. **The runtime rho metric is blind to the lens correlation that kappa reveals.** Finding-set rho
   (Jaccard over `(file, line, category)`) is **0.000 for every pair**, so the runtime
   submodularity gate (`rho <= 0.25`) would never fire — yet **contract↔invariant Cohen kappa is
   0.717** (strongly correlated decisions at different finding locations). The lenses make
   correlated PASS/FAIL decisions while placing findings at different lines. Empirically confirms the
   study-swarm's Kuncheva–Whitaker warning and shows finding-location Jaccard does not detect
   decision redundancy. **Design implication:** a future slice should evaluate a decision-correlation
   signal (kappa) for the gate, not finding-location overlap alone (this is backlog item **F-22**:
   the AST/tree-edit rho already exists in `metrics.py` but is not wired into the runtime gate).

2. **Coverage gain = 0: the union does not beat the best single lens on this corpus.** `invariant`
   alone catches all 48 code/tool positives (greedy: invariant `+48` → others `+0`). The
   "union beats any single lens" submodular thesis does **not** hold here. Caveat: small corpus,
   invariant-heavy after the QuixBugs ingest, and the lenses over-flag (see #3), so "catches every
   positive" is partly trigger-happiness, not pure skill.

3. **`contract_completeness` and `cross_boundary` over-flag** (specificity 0.364 / 0.667; precision
   0.588 / 0.750). The v1 "first-cut" prompts fire on clean artifacts; this drags tool_call verdict
   accuracy to 0.50. The clearest single lens-quality fix the data points to.

4. **Confidence is mis-calibrated** (ECE 0.241): reported confidence is ~24% off observed accuracy on
   average — the verifier is overconfident. Worth a calibration pass as the corpus grows.

5. **The contamination check is honest — and the worry is inverted here.** The vendored QuixBugs
   public samples are flagged `contaminated: true` (verifiers may have memorized them). Measured
   accuracy on contaminated samples (**0.560**) is actually **lower** than on uncontaminated samples
   (**0.754**) — so mistral is not acing memorized bugs on this set; the public-split ceiling concern
   does not bite for this verifier. The honest (uncontaminated) signal is the higher number.

6. **The rho-threshold sweep is degenerate** (all rho = 0), so 0.25 cannot be validated or
   recalibrated from this corpus — itself the evidence for finding #1. The runtime default stays 0.25.

## Same-family A/B (Lock 1) — REAL, and an honest null

`prism eval --family-ab` runs a same-family CONTROL against the family-different TREATMENT and reports
a **paired (McNemar) accuracy delta + CI**. A prior bug made the control silently route to
`VERIFIER_UNAVAILABLE` (delta measured nothing); that is fixed (the control uses a measurement-only,
default-off `allow_same_family` router bypass).

**Real run (2026-06-14, n=33 balanced subset):** the cross-family seat is **gpt-oss:120b-cloud**
(an OpenAI-family model, served via Ollama Cloud's OpenAI-compatible `/v1` — zero per-call cost) as
the family-different TREATMENT; **mistral-small:24b** (LOCAL) as the same-family CONTROL; caller family
`local`.

| Arm | Accuracy | Verifier |
|---|---|---|
| Family-different (treatment) | **0.697** (23/33) | gpt-oss:120b-cloud (OPENAI) |
| Same-family (control) | **0.727** (24/33) | mistral-small:24b (LOCAL) |
| **Delta (different − same)** | **−0.030** | 95% CI **[−0.187, +0.126]** (paired McNemar Wald) |

**Honest reading:** the delta is **indistinguishable from zero** — prism's own measurement does **not**
independently confirm a family-different *advantage* here. This does NOT refute Lock 1; three real
limitations bound the result: **(1)** n=33 (a wide CI by construction); **(2)** the two arms differ in
model capability (gpt-oss 120B vs mistral 24B), so the delta conflates family-difference with model
size — a directional proxy, not a clean isolation; **(3)** the corpus artifacts are not actually
*produced* by the caller families, so the self-preference effect Lock 1 guards against (Panickssery
2024) cannot be fully exercised on a fixed corpus. A clean test needs same-capability cross-family
arms over family-provenanced artifacts. Until then, Lock 1 stands on the borrowed empirical anchor,
and prism's own data is — honestly — null. (The CLI path `prism eval --family-ab` still requires a
2nd *general* configured family; the cloud seat used here was wired via a one-off harness because the
routing map hardcodes hosted model IDs — see backlog item F-14, configurable routing.)

## Family-different A/B v2 — the validity-clean within-judge design (method + pilot)

The v1.4.0 null above is *confounded* (capability ≠ family; a fixed corpus can't fire self-preference).
The fix shipped as a four-wave slice on `feat/familyab-corpus-pipeline`: a **family-provenanced,
execution-labeled corpus** (each model GENERATES the code, hidden tests LABEL it — no LLM grader), a
**config-driven verifier registry** (F-14: a cross-family seat is now env config, not a one-off
harness), and the **within-judge round-robin estimator**:

> for verifier family V over known-buggy artifacts,
> `self_preference(V) = false_accept_rate(V on its OWN family's bugs) − …(on OTHER families' bugs)`

V's capability cancels inside the contrast, so a positive aggregate is family self-preference, not a
model-size artifact (Panickssery 2024; ReaLMistake / Kamoi 2024). Inference is a **cluster (problem)
bootstrap** (items are nested by problem); a **discrimination-floor gate** excludes any verifier that
can't tell buggy from clean; the tri-state verdict collapses refuse/revise→refute, escalate→abstain,
accept→false-accept. The design was research-grounded + cross-family-verified via a study-swarm.

**Pilot run (2026-06-14, local Ollama, zero cost):** 3 distinct lineages — `mistral-small:24b`,
`granite4.1:30b`, `qwen3.6` — each generated solutions to 3 fresh problems → 18 execution-labeled
artifacts (9 buggy mutants + 9 clean) → each model judged every model's artifacts.

| Verifier | self_pref | false-accept own / other | refute own / other | bal-acc | interpretable |
|---|---|---|---|---|---|
| mistral | **+0.000** | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 | yes |
| granite | **+0.000** | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 | yes |
| qwen | **+0.000** | 0.00 / 0.00 | 1.00 / 1.00 | 0.94 | yes |
| **Aggregate** | **+0.000**, CI **[0.00, 0.00]** | — | — | — | 3/3 families, 3 problems |

**Honest reading — the method is validated; the signal is absent by a CEILING EFFECT, not by an
absence of self-preference.** Every family caught every bug (refute 1.00 on its own AND others'
artifacts) and false-accepted none (0.00 everywhere), so there is no headroom for a same-family judge
to *under*-refute — the contrast is structurally zero. The cause is corpus difficulty: single-operator
AST mutants of trivial functions (a flipped comparison, an off-by-one) are *obvious*, and a 24-30B
verifier with four lenses catches them every time. Self-preference manifests on the SUBTLE bugs a
same-family judge might miss (the ReaLMistake regime), not on bugs everyone catches. So the pilot's
real result is a **measurement-design** one: the validity-clean loop runs end-to-end (corpus →
round-robin → estimator → CI), and the remaining gap is **corpus difficulty**, not a statistical
confound — a concrete, addressable next step, unlike the v1.4.0 confound.

**Scale-up for a headline number:** harder/subtler defects where verifiers sometimes miss (natural
generations on harder post-cutoff problems + semantically-subtle bugs, not just clean-code mutants);
more families × problems for a CI that can exclude zero; and a first-class `prism eval --round-robin`
command (the pilot drove the committed machinery — `eval/familygen` + `eval/familyab` — from a
harness; the CLI wrapper is the reproducibility surface still to wire). Until that run, Lock 1 still
stands on the borrowed Panickssery anchor — but prism now has the *valid instrument* to measure it,
not a confounded proxy.

## L5 Style/Maintainability lens — ship/defer gate

Slice 1 builds the **gate**, not the lens (data, not vibes — design/07 §E). Ship criteria: a labeled
style corpus of ≥100 items with ≥2–3 independent human labels, inter-rater Krippendorff alpha ≥ 0.6,
candidate-L5 precision ≥ 0.8 per category, and a submodularity guard (L5 alone never triggers
ESCALATE; drop any category correlated with SLOC or the correctness lenses).

**Decision: DEFER.** No L5 lens and no human-labeled style corpus exist, so the gate cannot be met.
The evidence (Bacchelli & Bird ICSE 2013; weak maintainability-metric validity; LLM judges reliable
only on large quality gaps) supports a *narrow* future L5 — never scalar "maintainability scores" —
and only once the corpus clears the bar.

## CodeJudgeBench (pairwise: prism single-artifact → preference)

prism is single-artifact, but [CodeJudgeBench](https://huggingface.co/datasets/mattymchen/codejudgebench)
(arXiv:2507.10535, Apache-2.0) is pairwise. The harness (`src/prism/eval/benchmarks/`) verifies the
**chosen** and **rejected** code separately and reduces the two verdicts to a preference via
`calibrate.pairwise_prefer` (accept > escalate > revise > refuse, tie-break by confidence). A result
is **correct iff prism prefers the chosen side**; a **tie counts as WRONG** in the headline accuracy
and is reported separately as a tie-rate. Both response orders are run (N ≥ 3) to measure
**position consistency** (the paper reports order substantially affects judge accuracy).

Reproduce (real): `pip install 'prism-verify[bench]'` then
`prism eval --benchmark codejudgebench --bench-task codegen --bench-limit 50`. Offline machinery
smoke (mock verifier, committed fixture — NOT a measurement):
`prism eval --benchmark codejudgebench --offline`.

**Real first pass (2026-06-14):** verifier **local Ollama `mistral-small:24b`**, `--bench-task codegen
--bench-limit 16 --runs 1` (a small, zero-cost first pass; the published headline needs a full-split
run + ideally a stronger verifier).

| Bucket | Accuracy (95% CI) | Tie-rate | Position consistency |
|---|---|---|---|
| overall (codegen, n=16) | **0.375** [0.185, 0.614] | **0.562** | **1.000** [0.806, 1.000] |
| codegen | **0.375** [0.185, 0.614] | **0.562** | **1.000** |
| coderepair | (not yet run) | — | — |
| testgen | (not yet run) | — | — |

**Honest reading:** mistral-24B is a **weak pairwise discriminator** on code-quality pairs — accuracy
0.375 is dominated by a **0.562 tie-rate** (it gives the *same* single-artifact verdict to the chosen
and rejected code more than half the time, counted WRONG). This is genuine non-discrimination, not
order-bias: **position consistency is a perfect 1.000**. A larger, stronger verifier (e.g. the
gpt-oss:120b-cloud seat used for the A/B above) would almost certainly tie less — a worthwhile next run.

> The default suite + the offline fixture path need **neither** network **nor** the HF `datasets`
> lib (the `[bench]` extra). The cap matters: both-orders × N ≥ 3 × 2 sides = up to 12 verify
> calls/pair, so `--bench-limit` is load-bearing for spend; a published number needs a full-split run.

## Next

- Tighten the CIs: grow the fresh (post-cutoff) split via SWE-bench-Live (v1.2) so the honest signal
  rests on more than the small authored fresh set.
- Investigate the `contract`/`cross_boundary` over-flagging and the citation-clean over-escalation
  (findings #3 + the citation section); trial a kappa-based diversity gate (finding #1, F-22).
- The same-family A/B (`--family-ab`) needs a 2nd configured family; the CodeJudgeBench headline
  needs a real verifier run (machinery + fixture tests already ship — see the sections above).

## Family-different A/B v3 — PRE-REGISTRATION (locked 2026-06-14, before the run)

The harder-corpus run that targets a real headline number (the v1.5.0 pilot was a ceiling-effect null;
design/08). **Pinned BEFORE seeing any data** so the decision can't be p-hacked. Build harness:
`eval/run_familyab_v3.py`; estimator: `prism eval --round-robin`.

- **Estimand:** within-judge self_preference(V) = false_accept_rate(V on its OWN family's known-buggy
  artifacts) − false_accept_rate(V on OTHER families' buggy artifacts), aggregated (mean) over
  interpretable families. V's capability cancels inside the contrast (not the v1.4.0 confound).
- **Families (4 distinct LOCAL lineages):** `mistral-small:24b` (Mistral), `granite4.1:30b` (IBM
  Granite), `qwen3.6:latest` (Qwen), `gemma4:31b` (Google) — the rig's zero-cost panel; each generates
  AND judges. Diversity-beats-count (Verga 2024). (An earlier cloud roster — gpt-oss/glm/qwen-coder —
  was dropped: Ollama Cloud was transiently slow during the first attempt; local seats are reliable +
  zero-cost. No number was published before this re-lock, so the roster swap preserves "locked before
  data".)
- **Corpus:** LiveCodeBench-functional, release_v6, post-cutoff window `start_date=2025-01-01`,
  difficulty-stratified `{easy: 14, medium: 10, hard: 4}` (a mix so the deconfound stratum populates).
  Execution-labeled (no LLM grader). Natural-majority buggy stratum + execution-verified mutants + the
  faithful identical-failing-tests **deconfound stratum**.
- **SESOI = 0.05 FAR** (Panickssery 2024 magnitude). **Decision rule:** SUPERIORITY if the 95% cluster
  (problem) bootstrap CI on the aggregate excludes 0; else EQUIVALENCE if the 90% TOST CI ⊂ ±0.05;
  else INCONCLUSIVE; UNDERPOWERED if < 20 problem-clusters survive (the CI is not trusted).
  Discrimination-floor gate (balanced accuracy ≥ 0.6) excludes non-discriminating verifiers.
- **Outcome is honest either way:** a positive CI-excludes-0 = Lock-1 confirmed on prism's own data;
  an equivalence = self-preference bounded below 0.05 FAR; both retire the borrowed Panickssery anchor.

> Result appended below once the run completes. The number traces to the signed run-receipt
> (`round-robin-receipt.json`) + this pre-registration + the corpus content-hash.

# Slice — Family-AB HARDER corpus (close the Lock-1 headline number)

**Goal:** move prism's within-judge self-preference A/B off the **ceiling-effect null** (v1.5.0 pilot:
aggregate self_preference **+0.000, CI [0,0]**) by building a corpus whose execution-labeled bugs land
verifier false-accept rates in the **(0,1) variance regime** where family self-preference can express —
then re-run the round-robin and publish a real number (or an honest equivalence bound).

This is the study-swarm output (Research grounding + design) for the difficulty axis only. The locked
A–H statistical design ([[prism-family-ab-measurement-limits]]) is unchanged. **This doc is a PLAN, not
shipped code; build begins only after director review.**

---

## Verification receipt (research-grounded-advisor protocol, Step 4)

- **Study-swarm:** `wf_127b6806-943` (2026-06-14) — 5 parallel research lanes → per-lane retrieval
  verification → completeness critic → synthesis (12 agents, ~921k subagent tokens).
- **Stage 1 (existence / attribution oracle):** each lane's citations RETRIEVED via WebSearch/WebFetch
  (arXiv, Crossref, publisher/repo pages, author PDFs) — no parametric recall. Result: **0 fabricated**
  across all lanes; 4 attribution defects caught + **quarantined** (below).
- **Stage 2 (family-different groundedness):** the synthesizing agents were Claude, so a non-Claude
  ensemble independently audited the 14 architecture-load-bearing citations — `mistral-small:24b`
  (Mistral) `run_2026-06-14T15-20-47_89bb7c` + `gpt-oss:120b-cloud` (OpenAI)
  `run_2026-06-14T15-20-57_923a2e`. Both seats verified the citations within their knowledge (EvalPlus,
  Cameron 2008, Lakens 2018) as OK; the rest returned UNSURE — the documented **post-cutoff
  recent-paper blind spot** (non-informative, existence already retrieval-confirmed; cf. the protocol's
  founding receipt). mistral's 4 "overclaim/wrong-attribution" flags were all false positives on
  inspection (title-only guesses contradicted by the retrieved abstracts; one citation cross-wiring).
  **No real overclaim surfaced beyond the Stage-1 quarantine.**

### Quarantined — kept OUT of the architectural connections

1. **SSHOM "subsuming higher-order mutants are harder to kill" — citation unsound.** The drafted DOI
   `10.1016/j.infsof.2016.02.003` resolves to an unrelated code-smells paper (Walter & Alkhaeir); the
   real *Subtle higher order mutants* is Omar/Ghosh/Whitley, IST 2016 **vol 81**,
   `10.1016/j.infsof.2016.01.016` — **not** Nguyen & Madeyski. The SSHOM principle is genuine, but with
   author+DOI wrong we do **not** build a higher-order-mutant stratum on it. Natural fails carry the
   subtlety load instead. *Override only if the correct source is confirmed.*
2. **2009 HOM "first-order mutants are trivial" — mis-bylined.** DOI/title/substance correct, but the
   authors are **Jia & Harman**, not Papadakis/Le Traon. The finding (single-operator-on-trivial-code is
   the always-caught regime) is independently supported by Just 2014 + Yao/Harman/Jia 2014, which we cite
   instead.
3. **Tan 2025 (arXiv:2505.17656) "inherited shared-family bias / equal-scale series comparison" —
   overclaim.** The paper supports the weaker claim (an *external/cross-model* verifier is needed because
   self-consistent errors *differ across models*); the inherited-bias mechanism and equal-scale
   comparison are extrapolations. Lock 1's justification rests on the verified self-correction/self-repair
   blind-spot findings (Tsui 2025, Gong 2024) + EXTERNAL_VERIFIER (Kambhampati/Huang), not on this.
4. **BugsInPy "67% reproduce 3 years later" — mis-sourced.** Real figure, but from Rezaalipour & Furia
   2024 (arXiv:2305.19834), not the cited BugsInPy DOI. Immaterial — BugsInPy is rejected as a source
   anyway (pre-cutoff, mirror-contaminated).

---

## Research grounding (verified findings → design implications)

Every finding below is retrieval-verified (citation resolved, correctly attributed, finding grounded in
source text) unless flagged. Numbers reference the study-swarm lanes.

**Why the pilot nulled, and the target regime**

1. **Capable verifiers detect realistic LLM errors at low-but-nonzero recall → false-accept strictly in
   (0,1).** Kamoi, Das, Lou et al. 2024, *ReaLMistake* (arXiv:2404.03602, COLM 2024): GPT-4 recall
   59.5%/11.9%/12.6%, Claude 3 Opus 48.0/6.8/11.5, GPT-3.5 41.7/39.2/31.6 on natural errors; humans
   ~95.7 F1; self-consistency/majority-vote do **not** help. → *Engineer corpus difficulty so
   floor-passing seats land FAR ≈ 0.10–0.70; do NOT lean on the panel's vote to manufacture
   discrimination — it provably doesn't help, so variance must come from harder items.*
2. **Self-preference scales LINEARLY with self-recognition; judge-specific; causal via fine-tuning.**
   Panickssery, Bowman & Feng 2024 (arXiv:2404.13076, NeurIPS 2024), measured on **natural**
   CNN/DailyMail+XSUM generations. → *MDE/SESOI is per-judge, not pooled; the same-family judge must see
   its own family's artifact (natural-fail sourcing keeps the recognition channel live).*
3. **The channel rides on surface familiarity (low perplexity); own-vs-other gap vanishes once
   perplexity is fixed.** Wataoka, Takahashi & Ri 2024 (arXiv:2410.21819); GPT-4 self-pref 0.520. →
   *Mutators must NOT produce lexically jarring (high-perplexity) edits — caught by everyone, washes out
   the contrast. Log per-artifact perplexity (neutral reference LM) as a covariate.*
4. **Large, asymmetric self-correction blind spot: models fix the identical error from an external
   source but not in their own output.** Tsui 2025, *Self-Correction Bench* (arXiv:2507.02778): 64.5%
   avg across 14 models; controlled identical-error injection; "Wait" cuts it ~89%. → *(a) the
   identical-error-injection design is the DECONFOUNDER — hold the bug fixed, vary only which family's
   generation embeds it; (b) prompt sensitivity can swamp the family effect → pin lens prompts
   (PIN_PER_STEP).*
5. **Self-repair blind spot in the CODE+execution regime: models repair others' insecure code far better
   than their own.** Gong, Duan, Tao, Gong, Yuan & Huang 2024 (arXiv:2408.10495): 33.2–59.6% on others',
   worst on own ("insist" on the producing pattern). → *Strongest argument for NATURAL failed generations
   — they carry the family's pattern-insistence that AST mutants lack.*
6. **Generator capability co-produces difficulty: stronger solvers yield coherent-but-wrong code that
   collapses verifier TNR while TPR stays high.** Zhou, Xu, Zhou, Singh, Gui & Joty 2025, *Variation in
   Verification* (arXiv:2509.17995): Qwen2.5-72B verifier TNR **0.68 → 0.17** as the generator
   strengthened. → *Have STRONGER family seats generate the buggy artifacts to push FAR off the floor.
   This is ALSO the central confound (see must-fix 1): without the finding-4 deconfounder, own-minus-cross
   could be capability-driven bug difficulty, not self-preference.*
7. **Item discriminability peaks at MIDDLE difficulty, collapses at extremes (IRT).** Zhou, Huang, Zhao
   et al. 2025, *Lost in Benchmarks?* (arXiv:2505.15055): 1000-item Fisher-info subset reproduces full
   rankings, Kendall τ 0.9048. → *Calibration pre-pass: keep only items whose per-item panel FAR is in
   the informative band (~0.15–0.85), discard 0.0/1.0 — on a DISJOINT slice (must-fix 3).*
8. **Diverse panel of disjoint families beats a single large judge; each model's highest delta is judging
   itself.** Verga, Hofstätter, Althammer et al. 2024, *PoLL* (arXiv:2404.18796): 3-family panel, sd 2.2,
   ~7–8× cheaper; the self-delta is exactly prism's within-judge estimand. → *DIVERSITY beats COUNT —
   prioritize distinct families over extra seats per family.*
9. **Weak-verifier ensembling recovers most lost discrimination.** Saad-Falcon, Buchanan, Chen et al.
   2025, *Weaver* (arXiv:2506.18203, NeurIPS 2025): +13.5–15.5% over majority vote, within 4.2% of
   Pass@100. → *Multi-lens-per-verifier is justified, but treat lens budget as a DIFFICULTY KNOB — tune
   so panel FAR stays in-band, not saturated back to zero.*
10. **Ability-matched Fisher-info item selection holds precision with ~90% fewer items.** Li, Tang, Chen,
    Cheng, Metoyer, Hua & Chawla 2025, *ATLAS* (arXiv:2511.04689). → *Make the discrimination-floor gate
    ABILITY-MATCHED per seat (items near that seat's ability = peak FAR variance) so a mid-tier seat
    isn't failed on items only a frontier seat can resolve.*

**Sourcing**

11. **Contamination-free, release-dated, execution-labelable problems via post-cutoff competitive
    programming with hidden tests.** Jain, Han, Gu et al. 2024, *LiveCodeBench* (arXiv:2403.07974, MIT);
    DeepSeek/DS-Ins-33B accuracy drops on post-cutoff problems while GPT stays flat. → *PRIMARY source;
    supplies a built-in month-by-month contamination self-check.*
12. **Weak tests let ~1-in-4–5 "passing" LLM solutions hide a real bug strong tests expose.** Liu, Xia,
    Wang & Zhang 2023, *EvalPlus* (arXiv:2305.01210): 80× tests → pass@k −19.3–28.9%. → *EvalPlus-grade
    hidden tests are the labeling oracle that converts the 0% FAR ceiling into the (0,1) band.*
13. **BigCodeBench: hard, function-level, Apache-2.0, sandbox-runnable; top models fail ~40–50%.** Zhuo,
    Vu, Chim et al. 2024 (arXiv:2406.15877): 5.6 tests/task @ ~99% branch coverage, GPT-4o ~60%, 97%
    human-solvable; needs real sandbox (tempdirs/file I/O/mocked services). → *SECONDARY source; the
    execution-labeler must be CONTAINERIZED (which prism needs anyway to run LLM code safely).*

**Mutators (corroborate the existing operator set)**

14. **Operator choice (not count) decides real-fault coupling; only a minority of mutants couple.** Gay &
    Salahirad 2023 (DOI:10.1109/ICST57152.2023.00021, ICST 2023): 9.92% strongly coupled; 51.03% of
    faults have any strongly-coupled mutant. → *Keep prism's mutator restricted to high-coupling
    operators — it already does (ROR/AOR/LCR).*
15. **Operator classes split into equivalent-producers vs stubborn-producers.** Yao, Harman & Jia 2014
    (DOI:10.1145/2568225.2568265, ICSE 2014): ABS + half UOI → mostly equivalent (unlabelable); LCR →
    stubborn. → *Exclude ABS / equivalence-prone UOI, prefer LCR — `mutate.py` already excludes
    ABS/UOI/constant-replacement and includes LCR. Residual risk = equivalent-mutant LEAKAGE (must-fix
    4).*
16. **A substantial share of real faults couple to NO single-operator mutant (~27% uncoupled).** Just,
    Jalali, Inozemtseva, Ernst, Holmes & Fraser 2014 (DOI:10.1145/2635868.2635929, FSE 2014). → *Do not
    rely on single-operator mutants for the buggy stratum — source the majority as natural multi-line
    failures.*
17. **"Subtle but real" = a logical error that passes visible tests but fails hidden ones.** Greenblatt,
    Shlegeris, Sachan & Roger 2023, *AI Control* (arXiv:2312.06942) — "backdoor" = a logical error not
    caught by available tests. → *Exactly prism's hidden-test labeling: define the buggy stratum as
    artifacts that pass visible/example tests yet fail HIDDEN tests → 0<FAR<1 with no LLM in the label
    loop.*
18. **LLM-judge accuracy drops + becomes high-variance on strong-model code; varies by producer.** Jiang,
    Chen, Cao, Lee & Tan 2025, *CodeJudgeBench* (arXiv:2507.10535). [groundedness PARTIAL — the
    "lower-than-prior-benchmarks because subtle" causal phrasing isn't in the abstract; randomness +
    producer-variance are confirmed.] → *Corroborates 1 & 6; producer-dependent judge accuracy is the
    effect prism's family contrast estimates.*

**Statistics / pre-registration**

19. **Wild cluster bootstrap holds ~95% coverage down to ~6 clusters; plain cluster CIs under-cover below
    ~30–50.** Cameron, Gelbach & Miller 2008 (DOI:10.1162/rest.90.3.414, REStat 90(3):414–427). →
    *Add a WILD cluster bootstrap (Rademacher) path for low problem counts.*
20. **Cluster-bootstrap SEs reliable at ≥20 clusters; degrade past ~10 (MDE must be ≥0.80 SD).** Francis
    L. Huang 2018 (DOI:10.1177/0013164416678980, *Educ. & Psych. Measurement* 78(2):297–318)
    *[byline corrected from the draft's "Esquibel/Hox (Huang)"].* → *HARD FLOOR: never report a CI from
    <~20 problems; target ≥40/family. A wide CI at low cluster count is the under-powered trap, not "no
    effect."*
21. **A paired (McNemar) proportion difference needs low-hundreds of paired items for a 5–10-pt effect at
    80% power.** Lachin 1992 (DOI:10.1002/sim.4780110909, *Stat. in Medicine* 11(9):1239–1251): ~113 for
    0.15-vs-0.10. [PARTIAL — the 113 is a standard calculator output, consistent with the method.] →
    *Inflate by the cluster design effect 1+(m−1)·ICC; target ≥~120–150 own-family buggy items per judge
    over ≥40 problems.*
22. **TOST equivalence turns a null into "effect bounded below the SESOI" — if bounds are pre-registered.**
    Lakens, Scheel & Isager 2018 (DOI:10.1177/2515245918770963). → *Pre-register BOTH superiority (CI
    excludes 0) and TOST equivalence (±SESOI), so a future null reads "self-preference bounded below 0.05
    FAR," not [0,0].*
23. **BH-FDR preserves power vs Bonferroni for the family×lens grid.** Benjamini & Hochberg 1995
    (DOI:10.1111/j.2517-6161.1995.tb02031.x). → *BH-FDR q=0.05 over exploratory cells; use
    **Benjamini-Yekutieli** for the positively-correlated four-lens/Lock-4 cells (BH's independence/PRDS
    assumption fails there); reserve Holm for the single confirmatory aggregate.*

---

## The design (CORRECTED file mapping)

> The study-swarm's "Design decisions" mis-targeted the **sycophancy-specialist** files
> (`specialist/dataset/certify_verifier.py`, `config.COST_FP_OVER_FN=5`, `run_panel_votes.py`,
> `eval/corpus/prevalence.json`). The **family-AB** estimator is `src/prism/eval/familyab.py` +
> `eval/metrics.py` (endpoint = **false-accept rate on known-buggy**, tri-state collapse, and it ALREADY
> does a cluster (problem) bootstrap). The cost-asymmetric `COST_FP=5` scorer is a DIFFERENT (sycophancy)
> task and does **not** apply here — the binary false-accept endpoint is the correct estimand for the
> self-preference contrast. The corrected mapping is used below.

### (1) Defects + sourcing — `eval/mutate.py`, `eval/familygen.py`, `eval/_familygen_problems.py`

- **Operator set stays as-is** (ROR/AOR/LCR; ABS/UOI/constant-replacement already excluded) — verified
  correct vs findings 14–15. Nothing to add/drop. Do NOT add a higher-order/SSHOM stratum (quarantine 1).
- **Bug source → natural-majority (~70% natural / ~30% execution-verified mutants).** Findings 5, 6, 16.
  `familygen` already keeps natural failures as `natural_{status}` and mutates clean gens
  (`mutants_per_clean=3`); shift the ratio toward natural and have **stronger family seats generate the
  buggy artifacts** (finding 6) so natural bugs are coherent-but-wrong, not surface-broken.
- **Problem source → LiveCodeBench (primary, MIT) + BigCodeBench (secondary, Apache-2.0)**, replacing the
  contaminated QuixBugs/authored seed in `_familygen_problems.py`. Hidden tests are EvalPlus-grade
  (finding 12), wired into `ProblemSpec.test_code`. BigCodeBench ⇒ containerized labeler (finding 13).
  - ✅ **Wave 2a DONE** — `eval/bigcodebench.py` + `eval/livecodebench.py` (HF loaders, lazy `datasets`
    [bench], ANDON column validation, offline fixtures, content-hash; both schemas source-verified
    2026-06-14). Each row → a `ProblemSpec` whose `test_code` satisfies the existing sandbox
    `check(candidate)` contract (BigCodeBench: wrap the unittest class; LiveCodeBench: bake pre-parsed
    `(args, expected)` cases). **LiveCodeBench STDIN problems are DEFERRED** (functional-only kept) — a
    whole-program stdin/stdout harness is a separate build; the loader records the skipped stdin count.
    LCB date-window + `contest_date` holdout implemented. `datasets<4` pin is load-bearing (LCB is a
    `trust_remote_code` script dataset; `datasets>=4` removes scripts).
- **Identical-error-injection deconfounder stratum** (finding 4, must-fix 1): embed ONE fixed,
  execution-verified bug into EACH family's clean generation of the same problem — vary only the
  surrounding family style, hold the defect constant. Isolates family self-preference from
  capability-driven difficulty. New mode in `mutate.py`/`familygen.py`.
- **Equivalent-mutant + flaky-label guard** (must-fix 4): a kept mutant must fail the hidden suite
  **deterministically across N runs (majority)** under a pinned env, else discard. (`familygen` already
  drops still-passing mutants; this adds determinism.) Budget label noise ~0.05 (= SESOI-sized).
- **Per-artifact perplexity covariate** (finding 3) recorded in the manifest sidecar.

### (2) Estimator + statistics — `eval/familyab.py`, `eval/metrics.py`  ✅ Wave 1 DONE (`feat/familyab-estimator-hardening`)

- **Endpoint unchanged: false-accept rate on known-buggy (tri-state collapse).** Correct as-is; abstain
  shifts reported separately (`abstain_own`/`abstain_other` already exist). ✓
- **Minimum-cluster gate (≥20 problems)** — `compute_self_preference(min_problems=20)` flags the CI
  `ci_interpretable=False` + `decision="underpowered"` below the floor (finding 20). ✓ DONE.
- **Pre-registered decision rule** — `compute_self_preference(sesoi=…)` returns `decision ∈ {superiority,
  equivalence, inconclusive, underpowered}`: superiority = the (1−α) CI excludes 0; equivalence = the
  (1−2α) TOST CI within ±sesoi (finding 22) → turns a null into a bound, not the pilot's [0,0]. ✓ DONE.
- **Multiplicity primitives** — `tost_equivalence` + `benjamini_yekutieli` added to `metrics.py` (BY for
  the positively-correlated four-lens/Lock-4 cells; `benjamini_hochberg`/`mcnemar_midp` already shipped).
  ✓ DONE (+12 tests; full gate ruff + mypy --strict + pytest 704/6).
- **Wild cluster bootstrap (finding 19) — DEFERRED, with reason.** The Cameron-Gelbach-Miller wild
  cluster bootstrap is a regression-residual / cluster-mean technique; prism's aggregate is a
  ratio-of-rates contrast (own-FA − other-FA pooled within family), which doesn't admit the clean
  wild-bootstrap-t form without redefining the estimand to per-problem contributions. The **min-cluster
  gate is the operative small-#cluster protection** (don't report below 20 — the literature's actual
  recommendation). Revisit a wild/BCa refinement only if the estimator is reformulated; not load-bearing
  given the gate + a ≥40-problem target.
- **Disjoint calibration vs estimation slices** (must-fix 3) — DEFERRED to the corpus wave (it's an
  item-selection / run concern — which items go in which slice — not a pure-estimator change).

### (3) Corpus build / prevalence — `eval/familygen.py` (FAMILYAB corpus, not `eval/corpus/`)

- **Pin prevalence per problem** (balanced buggy:clean recommended) in the family-AB corpus build, so a
  null can't re-emerge as a prevalence artifact. (This is a `familygen` build parameter, NOT an
  `eval/corpus/prevalence.json` edit — that file belongs to the lens-calibration corpus.)
- **Contamination control per-family, not a global date** (finding 11): select each problem's window
  AFTER that family's own cutoff; wire LiveCodeBench's month-by-month curve as an actual preflight check;
  keep `contamination_max_run>0` so memorized artifacts drop, not just record. `contamination_overlap`
  already exists.

### (4) Discrimination floor + family roster — `familygen.FamilySpec` + F-14 routing

- **Ability-matched discrimination floor per seat** (finding 10), evaluated with generator reasoning
  hidden (EXTERNAL_VERIFIER). Target panel FAR ~0.10–0.70.
- **Roster: 4–6 DISTINCT families** (diversity > count, finding 8), one strong cloud seat per family
  (coherent/hard self-artifacts) + mid-tier seats (so some items stay below the FAR ceiling). Add
  `gpt-oss:120b` / `glm-4.6` ONLY as NEW family axes that clear the floor — never a second member of an
  existing family for raw count. Configured via `familygen.FamilySpec` (model id per family) + the F-14
  recipe (`PRISM_VERIFIER_MODEL_OPENAI=gpt-oss:120b-cloud`, etc.) — NOT `run_panel_votes.py` (sycophancy
  path).

---

## Build plan (status on `feat/familyab-harder-corpus` — 10 commits, all gated; 737 tests)

0. ✅ **Verify-current-state** — corrected the sycophancy-path conflation; mapping locked to
   `eval/familyab` + `eval/metrics`.
1. ✅ **Sourcing + labeler** — `eval/bigcodebench` + `eval/livecodebench` loaders (real-data validated);
   containerized labeler (`eval/container_sandbox` + `eval/docker/labeler.Dockerfile`, route-by-libs).
2. ✅ **Defects** — natural-majority sourcing + the faithful identical-error-injection deconfounder
   (failing-test-signature match) + perplexity covariate + ModuleNotFoundError-skip.
3. ✅ **Estimator hardening** — min-cluster gate + TOST + BY-FDR (wild-bootstrap deferred; gate is the
   operative small-cluster protection).
   Also: `eval/problem_select.stratified_by_difficulty` + a `deconfound_eligible_problems` manifest
   diagnostic — compose a difficulty MIX (real-model smoke found the deconfound stratum needs >=2
   families to solve a problem cleanly, which hard problems rarely allow).
4. ✅ **`prism eval --round-robin` CLI** (objective #2): `--familyab-corpus <dir>` → per-family bypass
   engines (in `cli/main.py`, `allow_same_family`, per the Knight-Capital guard) →
   `familyab.run_round_robin` → `compute_self_preference` (`--sesoi`, `--min-problems`) → `round_robin.md`
   + signed run-receipt. Offline integration test proves the wiring end-to-end.
5. ⏳ **Pre-register + RUN + publish**: pin the pre-registration block in `RESULTS.md`, run the
   round-robin on the harder corpus (local families + cloud seat, free), publish the number / bound.

Validation DONE: REAL-model end-to-end smoke (generate → label → mutate → natural-bugs on local
families over LCB-functional) + the Docker labeler image built (1.04 GB; numpy/pandas/scipy/sklearn;
real container labeling clean→pass / buggy→fail). The deconfounder is unit-validated (same-bug kept,
different-bug dropped); its real-model trigger needs the difficulty mix above.

Each wave: build + `uv run python -m {ruff,mypy,pytest}` green; tests ship test-first; line-length 100;
mypy --strict.

---

## Standards compliance (the six)

| Standard | Score | Evidence / plan |
|---|---|---|
| PIN_PER_STEP | 2→3 | The study-swarm script + per-agent prompts are journaled (`wf_127b6806-943`); the run pins family→model ids, seeds, temp, prompt hashes into the round-robin run-receipt (target 3 once the receipt lands). |
| ANDON_AUTHORITY | 2 | `familygen` ANDON-halts on corpus-integrity failure; min-cluster gate + discrimination-floor gate refuse to emit an uninterpretable CI; contamination preflight flags a leaking window before the run. |
| NAMED_COMPENSATORS | 3 | Read-mostly; the one published artifact (`RESULTS.md` number) has a named revert — see table. |
| DECOMPOSE_BY_SECRETS | 3 | `eval/familygen` (corpus) / `eval/mutate` (defects) / `eval/familyab`+`eval/metrics` (estimator/stats) / `cli` (orchestration) are separate modules hiding separate secrets; family-AB path is decoupled from the sycophancy-specialist path (the conflation corrected above). |
| UNCERTAINTY_GATED_HUMANS | 2 | The pre-registered superiority/equivalence/inconclusive rule defers to a documented decision under uncertainty; CANNOT_CONFIRM citations were surfaced contrastively (quarantine block). |
| EXTERNAL_VERIFIER | 3 | The whole slice IS the external-verifier experiment (a different family judges; reasoning hidden); the design's own citations were family-different-verified (receipt above); labels are execution ground truth, no LLM in the label loop. |

## Compensators (NO-SKIP — irreversible actions)

| Irreversible action | Compensator | Surface | Owner | Post-rollback state |
|---|---|---|---|---|
| Commit a published self-preference number to `eval/RESULTS.md` | Revert the doc commit; regenerate from a re-run | `git revert <sha>` + `prism eval --round-robin` | release operator | Prior (or no) published number restored |
| Corpus admits a contaminated/leaked problem that inflates accept rates | Exclude the window (contamination preflight), recompute, annotate the delta | edit `FAMILYAB_MANIFEST.json` + re-run | eval owner | Honest window restored; delta disclosed |
| Provider calls during the round-robin (paid/cloud compute) | Bounded by family×problem×artifact caps + local-family default (free); cloud seat opt-in | `familygen`/round-robin caps | eval owner | Read-only external; nothing to undo (documented) |
| Family-AB corpus written to a repo dir | Delete the generated corpus dir (regenerable, deterministic given seed) | `rm -r <corpus dir>` + rebuild | eval owner | Corpus regenerable from the pinned manifest |

The round-robin writes only the corpus dir + the report dir + receipts (covered by `receipt
delete`/`prune`); no network writes beyond the (bounded, opt-in) verifier calls.

---

## Related
- [[prism-family-ab-measurement-limits]] — the locked A–H design this difficulty-axis slice extends.
- [[prism-verify]] — v1.5.0 shipped the valid instrument; this closes the number.
- [[research-grounded-advisor-protocol]] — the study-swarm doctrine + the Step-4 gate run above.
- `design/07-slice1-calibration.md` — the lens-calibration corpus (a SEPARATE corpus from this one).

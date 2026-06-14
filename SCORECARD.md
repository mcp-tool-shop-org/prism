# Scorecard — prism-verify

**Repo:** prism-verify
**Date:** 2026-06-14 (v1.4.0)
**Type tags:** `[all]` `[pypi]` `[npm]` `[cli]` `[mcp]`

## Assessment (v1.4.0)

> Suite at **641 tests** (was 472); ruff + mypy `--strict` clean. v1.4.0 adds the validate-on-your-own-data
> eval surface (real same-family A/B McNemar delta, vendored QuixBugs real-bug corpus, CodeJudgeBench
> harness), an observability layer (`request_id` correlation, structured logs, breaker state on `/healthz`),
> and graceful degradation (circuit breaker on the sycophancy paths, oracle cache TTL+bound).

| Category | Score | Notes |
|----------|-------|-------|
| A. Security | 9/10 | SECURITY.md + README threat model; no secrets; fail-closed HTTP auth (keys hashed at rest, constant-time); SSRF-guarded webhooks; Ed25519 third-party-verifiable receipts with an honest tamper-evidence ceiling. v1.4.0 adds oracle hardening (arXiv id-match, body cap, redirects disabled), a harvest secret-scrub + a ReDoS fix. -1: genuine tamper-resistance (HSM + transparency log) is named-but-deferred. |
| B. Error Handling | 9/10 | `VerifyError{reason,detail,retryable}` + RFC 9457 problem+json; CLI exit codes; MCP/engine degrade gracefully (no crash on out-of-enum/fenced output, idempotent schema migration); circuit breaker wired into the sycophancy paths. |
| C. Operator Docs | 9/10 | README current for v1.4 (cites measured eval numbers from `eval/RESULTS.md`); CHANGELOG (Keep a Changelog); LICENSE; `CONTRIBUTING.md`; accurate `--help`; MCP tools documented; astro-starlight docs handbook + landing shipped (Phase-10). -1: deep API reference still lives in docstrings. |
| D. Shipping Hygiene | 9/10 | `scripts/verify.py`; tag==version gate in `release.yml` (PyPI + npm); the npm launcher derives its binary `version`/`tag` from `package.json` at runtime and `release.yml` guards against a hard-coded pin (`node --check` + grep), so the wrapper can't ship a stale binary (the CRITICAL v0.4.2 pin is fixed); SHA-pinned publish actions, Node-24 bumps, dependency upper-bounds; clean `uv build` + `twine check`; `uv.lock` committed; PyPI + npm Trusted Publishing (OIDC, PEP 740 attestations / provenance). -1: no Dependabot (org-rule SKIP). |
| E. Identity (soft) | 10/10 | Logo in README; translations (8 languages), live landing page, astro-starlight handbook, and GitHub topics/homepage all shipped (Phase-10 brand treatment). |
| **Overall** | **46/50** | Hard gates A–D pass; soft gate E fully shipped. |

## Key Gaps

1. Tamper-resistance ceiling (HSM / transparency log) — named hardening, deferred (design/05).
2. Deep API reference currently lives in docstrings rather than the handbook.
3. Eval corpus is small (111 samples) — headline findings are directional; a larger fresh post-cutoff split is the next upgrade.

## Remediation Priority

| Priority | Item | Status |
|----------|------|--------|
| 1 | Hard gates A–D | ✅ PASS |
| 2 | Brand treatment (E) — translations → landing → handbook → topics | ✅ Shipped (Phase 10) |
| 3 | npm launcher binary-pin drift (was shipping a stale v0.4.2 binary) | ✅ Fixed — launcher self-syncs from package.json; CI guards the pin |
| 4 | `--family-ab` control silently measured nothing (VERIFIER_UNAVAILABLE) | ✅ Fixed — real paired McNemar delta + CI (v1.4.0) |
| 5 | Tamper-resistance ceiling (HSM / transparency log) | Named hardening, deferred (design/05) |
| 6 | Grow the eval corpus (larger fresh post-cutoff split) | Backlog |

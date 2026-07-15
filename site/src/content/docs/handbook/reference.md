---
title: Reference
description: prism CLI commands, environment variables, verdicts, and refusal codes.
sidebar:
  order: 5
---

## CLI commands

| Command | What it does |
|---|---|
| `prism verify -a <artifact> -i <intent> --caller-family <fam>` | Verify an artifact. `-a @file` reads a file. `--type code\|tool_call\|citations`, `--provider ollama\|anthropic\|...`, `--gate` for verdict exit codes. |
| `prism replay <receipt_id>` | Print a stored receipt + `signature_valid`. |
| `prism verify-receipt <file> [--public-key <pem>]` | Cryptographically verify a standalone receipt. With `--public-key`, an Ed25519 receipt verifies with no shared secret. |
| `prism keygen [--out <path>]` | Generate an Ed25519 signing keypair. |
| `prism pubkey` | Print the configured Ed25519 public key + key id. |
| `prism receipt delete <id>` · `prism receipt prune --older-than <dur> --yes` | Compensators for the receipt store. |
| `prism serve [--host --port]` | Run the HTTP service (needs the `[http]` extra). |
| `prism eval [--split --runs --offline --family-ab]` | Measure the lenses on the labeled corpus — per-lens P/R, diversity matrix, coverage-gain, calibration. See [Calibration & benchmark](../evaluation/). |

`--gate` exit codes: `0` accept · `10` revise · `20` refuse · `30` escalate.

### Choosing the verifier

`--provider` is an **allowlist** (repeatable; `ollama` · `anthropic` · `openai` · `google` ·
`openrouter` · `local-verifier` · `local-sycophancy`). Only the providers you name are registered,
so a run scoped to `ollama` stays local and free even when hosted API keys are present in the
environment — an ambient key can never silently redirect or bill it. Pass `--provider` more than
once to give the router real cross-family failover:

```bash
# a local seat, pinned per-run, with an OpenRouter failover behind it
prism verify --provider ollama --provider openrouter \
  --verifier-model local=qwen2.5:7b \
  --artifact @patch.py --intent "add two numbers"
```

## Environment variables

| Variable | Purpose |
|---|---|
| `PRISM_SIGNING_KEY` | Ed25519 private-key PEM (path or inline) — the v0.4 production default. |
| `PRISM_SIGNING_SECRET` | HMAC signing secret (legacy / explicit). |
| `PRISM_DEV=1` | Use a built-in dev Ed25519 key — INSECURE, local only. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` | Enable a hosted verifier family. |
| `PRISM_VERIFIER_MODEL_<FAMILY>` | Override the verifier model id for a family wherever it routes (e.g. `PRISM_VERIFIER_MODEL_OPENAI=gpt-oss:120b-cloud`). A model deprecation is a config change, not a source edit. Honored by the CLI, the HTTP API, and MCP alike; `prism verify --verifier-model FAMILY=MODEL` overrides it per-run. |
| `OPENROUTER_API_KEY` + `PRISM_VERIFIER_MODEL_OPENROUTER` | Enable the OpenRouter gateway as one cross-family verifier seat (both required). The model must be a `vendor/model` id whose vendor is not one prism serves by default — the lineage guard fails closed otherwise. See [What the lineage guard does and does not check](#what-the-lineage-guard-does-and-does-not-check) before pairing it with a repointed `local` seat. |
| `PRISM_<PROVIDER>_MODEL` / `PRISM_<PROVIDER>_BASE_URL` | Override a provider's default model / base URL (`ANTHROPIC` · `OPENAI` · `GOOGLE` · `OLLAMA`). Point `PRISM_OPENAI_BASE_URL` at an OpenAI-compatible endpoint — e.g. Ollama Cloud's `/v1` — to use a hosted model as a cross-family verifier seat. |
| `PRISM_API_KEYS` | Comma-separated SHA-256 hashes of HTTP bearer API keys. |
| `PRISM_HTTP_ALLOW_NO_AUTH=1` | Allow unauthenticated HTTP use (local dev only). |
| `PRISM_WEBHOOK_SECRET` | Sign async/escalate webhook deliveries. |

## What the lineage guard does and does not check

Lock 1 keeps a verifier in a different **family** from the caller. That is what you want when a
family label means a lineage — `anthropic`, `openai`, and `google` each name who trained the model,
and you own that claim when you point the seat somewhere.

`local` and `openrouter` are different: they are **transports**, not lineages. prism labels every
Ollama model `local` and every gateway model `openrouter`, so for those two the lineage lives in the
model id rather than the family. Two seats with different labels can be the same lineage.

The OpenRouter lineage guard refuses a `vendor/model` whose vendor is one prism serves **by
default** — `anthropic`, `openai`, `google`, and `mistral` (because `local` defaults to
`mistral-small:24b`). Read *by default* literally. `PRISM_VERIFIER_MODEL_LOCAL` lets you repoint the
local seat, and the guard is a fixed list that cannot know you did:

```bash
PRISM_VERIFIER_MODEL_LOCAL=qwen2.5:7b
PRISM_VERIFIER_MODEL_OPENROUTER=qwen/qwen-2.5-72b-instruct
# A local qwen producer is now "cross-family"-verified by qwen: different label, same lineage.
# The guard does not raise, and the receipt records a family-different verification.
```

Neither setting is wrong on its own — the collision is emergent, which is exactly why the guard
cannot catch it. It has only the model id, and nothing in `qwen/qwen-2.5-72b-instruct` reveals what
`local` was pinned to. Extending the blocklist does not fix this in principle; it is a static answer
to a question your configuration can change.

**What to do:** if you repoint `local` (or a `local-*` specialist), check that your OpenRouter seat
is not the same lineage. If it is, pick a different vendor for the seat — you have the whole
catalogue. This only matters when the caller shares that lineage too, since one verifier serves a
request and the rest are failover.

This is a known limitation, pinned by tests rather than left to rot. Closing it properly means
checking lineage at **request** time, where the caller's own model id is available; a construction-
time check would have to refuse configurations that are perfectly sound for other callers.
| `PRISM_TRUSTED_PROXIES` | Comma-separated CIDRs; honor `X-Forwarded-For` only from these peers (default empty = none). |
| `PRISM_MAX_ARTIFACT_BYTES` | HTTP artifact size cap (default 256 KiB). |

## Verdicts

`accept` · `revise` (with a `revision_hint`) · `refuse` · `escalate` (route to a human). The
artifact verdict aggregates conservatively: any refuse → refuse; else any revise → revise; else any
escalate → escalate; else accept.

## Refusal codes (ANDON halts)

| Code | Meaning |
|---|---|
| `VERIFIER_UNAVAILABLE` | No alternate-family verifier route is available (never falls back same-family). |
| `STRIP_VERIFICATION_FAILED` | Reasoning patterns survived stripping — cannot proceed safely. |
| `LENS_COLLAPSE` | Lenses agreed beyond the submodularity threshold (ρ ≤ 0.25). |
| `BUDGET_EXCEEDED` | The lens fan-out exceeded the caller's latency budget. |
| `INVALID_ARTIFACT` | The artifact (e.g. a citations array) is malformed. |

## HTTP endpoints

See [HTTP service](../http-service/) for `POST /verify`, `GET /replay/{id}`,
`POST /verify-receipt`, `GET /healthz`, and `/docs`.

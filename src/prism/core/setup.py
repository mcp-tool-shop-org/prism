"""Shared engine construction for the CLI / MCP / HTTP surfaces.

One place builds the default lens set + the provider map from the environment, so the three
transports cannot drift apart (the family-of-call-sites lesson). Each surface is a thin wrapper
over the engine this returns.
"""

from __future__ import annotations

import os

from prism.core.engine import VerificationEngine
from prism.core.observability import routing_logger
from prism.lenses.boundary import CrossBoundaryLens
from prism.lenses.contract import ContractCompletenessLens
from prism.lenses.groundedness import GroundednessLens
from prism.lenses.invariant import InvariantLens
from prism.lenses.registry import register_lens
from prism.providers.base import ModelProvider


def register_default_lenses() -> None:
    """Register the v1 four-lens set (idempotent enough — registry de-dupes by name)."""
    register_lens(ContractCompletenessLens())
    register_lens(CrossBoundaryLens())
    register_lens(InvariantLens())
    register_lens(GroundednessLens())


def build_providers_from_env() -> dict[str, ModelProvider]:
    """Build the provider map: always-on local Ollama + any hosted family with an API key set."""
    providers: dict[str, ModelProvider] = {}

    from prism.providers.ollama import DEFAULT_BASE_URL as _OLLAMA_BASE
    from prism.providers.ollama import OllamaProvider

    # PRISM_OLLAMA_BASE_URL points the local family at a non-default Ollama host (F-14).
    providers["local"] = OllamaProvider(
        base_url=os.environ.get("PRISM_OLLAMA_BASE_URL", _OLLAMA_BASE)
    )

    # The Verifier specialist (a locally-served fine-tuned groundedness model) — opt-in via env.
    # When set, build_default_engine injects it as the primary citation-groundedness verifier;
    # mistral `local` and the hosted families remain as cross-family failover targets. Recommend
    # pairing with PRISM_NLI_FLOOR (the orthogonal encoder-NLI veto) since the circuit-breaker fails
    # over on errors, not on a confident-wrong "supported".
    verifier_endpoint = os.environ.get("PRISM_LOCAL_VERIFIER_ENDPOINT")
    if verifier_endpoint:
        from prism.providers.local_verifier import LocalVerifierProvider

        providers["local-verifier"] = LocalVerifierProvider(
            endpoint=verifier_endpoint,
            model_id=os.environ.get("PRISM_LOCAL_VERIFIER_MODEL", "qwen3-14b-groundedness"),
        )

    # The Sycophancy specialist (wedge #2) — opt-in via env. Backs the sycophancy lens on RESPONSE
    # artifacts; its own family (LOCAL_SYCOPHANCY) keeps it family-different from the producer.
    sycophancy_endpoint = os.environ.get("PRISM_SYCOPHANCY_ENDPOINT")
    if sycophancy_endpoint:
        from prism.providers.sycophancy import SycophancyProvider

        providers["local-sycophancy"] = SycophancyProvider(
            endpoint=sycophancy_endpoint,
            model_id=os.environ.get("PRISM_SYCOPHANCY_MODEL", "qwen3-14b-sycophancy"),
        )

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        from prism.providers.anthropic import DEFAULT_BASE_URL as _ANTHROPIC_BASE
        from prism.providers.anthropic import AnthropicProvider

        providers["anthropic"] = AnthropicProvider(
            api_key=anthropic_key,
            base_url=os.environ.get("PRISM_ANTHROPIC_BASE_URL", _ANTHROPIC_BASE),
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        # PRISM_OPENAI_BASE_URL makes an OpenAI-compatible endpoint a first-class verifier seat —
        # e.g. Ollama Cloud's /v1 serving gpt-oss:120b-cloud as the cross-family A/B treatment
        # (paired with PRISM_VERIFIER_MODEL_OPENAI), replacing the v1.4.0 one-off harness.
        from prism.providers.openai import DEFAULT_BASE_URL as _OPENAI_BASE
        from prism.providers.openai import OpenAIProvider

        providers["openai"] = OpenAIProvider(
            api_key=openai_key,
            base_url=os.environ.get("PRISM_OPENAI_BASE_URL", _OPENAI_BASE),
        )

    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        from prism.providers.google import DEFAULT_BASE_URL as _GOOGLE_BASE
        from prism.providers.google import GoogleProvider

        providers["google"] = GoogleProvider(
            api_key=google_key,
            base_url=os.environ.get("PRISM_GOOGLE_BASE_URL", _GOOGLE_BASE),
        )

    return providers


def build_default_engine() -> VerificationEngine:
    """Register the default lenses and construct an engine with env-configured providers.

    The routing map is resolved from env (F-14: PRISM_VERIFIER_MODEL_*), so a verifier model
    deprecation or a cross-family seat (e.g. an Ollama-Cloud model via the OpenAI endpoint) is a
    config change, not a source edit. When unset, the resolved map equals DEFAULT_ROUTING_MAP. When
    the local Verifier specialist is configured (PRISM_LOCAL_VERIFIER_ENDPOINT), it is prepended as
    the PRIMARY citation verifier (failing over to the hosted/mistral verifiers behind it).
    """
    register_default_lenses()
    providers = build_providers_from_env()
    from prism.core.routing import (
        FamilyRouter,
        resolve_routing_map,
        routing_map_digest,
        with_local_verifier,
    )

    routing_map = resolve_routing_map()
    if "local-verifier" in providers:
        model_id = providers["local-verifier"].available_models[0]
        routing_map = with_local_verifier(routing_map, model_id)
    routing_logger.info("routing_map_resolved", extra={"digest": routing_map_digest(routing_map)})
    return VerificationEngine(providers=providers, router=FamilyRouter(routing_map=routing_map))

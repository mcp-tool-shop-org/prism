"""F-14: config-driven verifier registry + provider base_url overrides + the dated-model fix."""

from __future__ import annotations

import pytest

from prism.core.routing import DEFAULT_ROUTING_MAP, resolve_routing_map, routing_map_digest
from prism.core.types import ModelFamily


def test_no_env_returns_base_unchanged() -> None:
    # Byte-identical contract when the registry is unconfigured (identity, not a copy).
    assert resolve_routing_map(env={}) is DEFAULT_ROUTING_MAP


def test_verifier_model_override_applies_wherever_family_is_a_verifier() -> None:
    out = resolve_routing_map(env={"PRISM_VERIFIER_MODEL_OPENAI": "gpt-oss:120b-cloud"})
    assert out is not DEFAULT_ROUTING_MAP
    for routes in out.values():
        for family, model_id in routes:
            if family == ModelFamily.OPENAI:
                assert model_id == "gpt-oss:120b-cloud"


def test_override_leaves_other_families_and_structure_untouched() -> None:
    out = resolve_routing_map(env={"PRISM_VERIFIER_MODEL_OPENAI": "gpt-oss:120b-cloud"})
    assert set(out) == set(DEFAULT_ROUTING_MAP)  # caller keys unchanged
    for caller, routes in out.items():
        base_routes = DEFAULT_ROUTING_MAP[caller]
        assert len(routes) == len(base_routes)
        for (f1, m1), (f2, m2) in zip(routes, base_routes, strict=True):
            assert f1 == f2  # verifier family order preserved
            if f1 != ModelFamily.OPENAI:
                assert m1 == m2  # only the OPENAI model id changed


def test_digest_is_stable_and_sensitive() -> None:
    d_base = routing_map_digest(DEFAULT_ROUTING_MAP)
    assert d_base.startswith("routing-")
    assert routing_map_digest(resolve_routing_map(env={})) == d_base
    d_changed = routing_map_digest(resolve_routing_map(env={"PRISM_VERIFIER_MODEL_OPENAI": "x"}))
    assert d_changed != d_base


def test_anthropic_default_model_is_durable_alias_not_dated() -> None:
    from prism.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(api_key="test-key")
    assert provider.available_models[0] == "claude-haiku-4-5"
    # A dated snapshot (…-YYYYMMDD) would be retired on a schedule and dead-end the route.
    assert all("-2025" not in m and "-2026" not in m for m in provider.available_models)


def test_openai_base_url_override_wires_the_ollama_cloud_seat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "ollama")  # dummy key for the OpenAI-compatible endpoint
    monkeypatch.setenv("PRISM_OPENAI_BASE_URL", "http://localhost:11434")
    from prism.core.setup import build_providers_from_env

    providers = build_providers_from_env()
    assert "openai" in providers
    client = getattr(providers["openai"], "_client")
    assert "11434" in str(client.base_url)

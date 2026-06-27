"""OpenRouter cross-family verifier seat (F-14): provider, lineage guard, routing injector, wiring.

The seat is a multi-vendor GATEWAY served as ONE prism family (OPENROUTER). These tests lock the
Lock-1-honesty guarantee: the lineage guard refuses a model whose vendor collides with a native
prism family, the seat is byte-identical-absent when unconfigured, and it builds only when BOTH
``OPENROUTER_API_KEY`` and ``PRISM_VERIFIER_MODEL_OPENROUTER`` are set.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from prism.core.routing import DEFAULT_ROUTING_MAP, FamilyRouter, with_openrouter
from prism.core.setup import build_default_engine, build_providers_from_env
from prism.core.types import ModelFamily
from prism.providers.base import CompletionRequest, ProviderError
from prism.providers.openrouter import (
    BLOCKED_VENDORS,
    OpenRouterLineageError,
    OpenRouterProvider,
    validate_openrouter_lineage,
)

_MODEL = "deepseek/deepseek-chat"


# --- fakes (mirror tests/unit/test_local_verifier.py) ---


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "http",
                request=httpx.Request("POST", "http://x"),
                response=httpx.Response(self.status_code),
            )

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload=None, raise_exc=None, status=200):
        self._payload, self._raise, self._status = payload, raise_exc, status

    async def post(self, url, json=None):
        if self._raise:
            raise self._raise
        return _FakeResp(self._payload, self._status)

    async def get(self, url):
        return _FakeResp({}, 200)


def _ok_payload(content="ok"):
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3},
    }


def _provider(payload=None, raise_exc=None, status=200, model_id=_MODEL):
    p = OpenRouterProvider(api_key="k", model_id=model_id)
    p._client = _FakeClient(payload=payload, raise_exc=raise_exc, status=status)
    return p


def _run(p):
    return asyncio.run(
        p.complete(CompletionRequest(system_prompt="s", user_prompt="u", model_id=_MODEL))
    )


# --- lineage guard (the Lock-1 crux) ---


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek/deepseek-chat",
        "qwen/qwen-2.5-72b-instruct",
        "cohere/command-r-plus",
        "nvidia/nemotron-4-340b-instruct",
        "meta-llama/llama-3.3-70b-instruct",
        "x-ai/grok-2",
    ],
)
def test_lineage_guard_allows_distinct_families(model_id):
    validate_openrouter_lineage(model_id)  # must not raise


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
        "google/gemini-2.0-flash",
        "mistralai/mistral-large",
        "mistral/mistral-small",
        "openrouter/auto",
    ],
)
def test_lineage_guard_blocks_native_or_indeterminate_families(model_id):
    with pytest.raises(OpenRouterLineageError):
        validate_openrouter_lineage(model_id)


def test_lineage_guard_blocks_bare_id_without_vendor():
    with pytest.raises(OpenRouterLineageError):
        validate_openrouter_lineage("gpt-4o")  # no 'vendor/' prefix


def test_blocked_vendors_cover_every_native_caller_lineage():
    # Every family prism routes to natively (anthropic/openai/google + the LOCAL=mistral lineage)
    # must be blocked from masquerading as OPENROUTER.
    assert {"anthropic", "openai", "google", "mistral", "mistralai"} <= BLOCKED_VENDORS


# --- provider ---


def test_family_is_openrouter():
    assert _provider().family == ModelFamily.OPENROUTER


def test_available_models_is_the_configured_model():
    assert _provider(model_id="qwen/qwen-2.5-72b-instruct").available_models == [
        "qwen/qwen-2.5-72b-instruct"
    ]


def test_complete_returns_content():
    resp = _run(_provider(payload=_ok_payload("pong")))
    assert resp.content == "pong"
    assert resp.input_tokens == 5
    assert resp.output_tokens == 3


def test_http_error_raises_provider_error():
    with pytest.raises(ProviderError):
        _run(_provider(payload=_ok_payload(), status=500))


def test_empty_content_raises_provider_error():
    # The verified-live failure mode: a free model returns "" -> RAISE (fail over), never a silent
    # vacuous adjudication.
    with pytest.raises(ProviderError):
        _run(_provider(payload=_ok_payload("")))


def test_missing_choices_raises_provider_error():
    with pytest.raises(ProviderError):
        _run(_provider(payload={"usage": {}}))


def test_transport_error_raises_provider_error():
    with pytest.raises(ProviderError):
        _run(_provider(raise_exc=httpx.ConnectError("down")))


# --- routing injector ---


def test_with_openrouter_appends_failover_to_every_caller():
    m = with_openrouter(DEFAULT_ROUTING_MAP, _MODEL)
    for caller, verifiers in m.items():
        if caller == ModelFamily.OPENROUTER:
            # Lock 1: a family never verifies itself.
            assert all(f != ModelFamily.OPENROUTER for f, _ in verifiers)
        else:
            # Appended LAST (failover), not prepended — fills in behind native verifiers.
            assert verifiers[-1] == (ModelFamily.OPENROUTER, _MODEL)


def test_with_openrouter_leaves_default_map_unmutated():
    before = DEFAULT_ROUTING_MAP[ModelFamily.ANTHROPIC]
    with_openrouter(DEFAULT_ROUTING_MAP, _MODEL)
    assert DEFAULT_ROUTING_MAP[ModelFamily.ANTHROPIC] is before
    assert all(
        f != ModelFamily.OPENROUTER
        for routes in DEFAULT_ROUTING_MAP.values()
        for f, _ in routes
    )


def test_openrouter_is_the_serviceable_route_when_only_local_and_openrouter():
    # A local producer + an OpenRouter seat, no native hosted keys: the router walks past the
    # unserviceable anthropic/openai/google routes and lands on OPENROUTER.
    router = FamilyRouter(routing_map=with_openrouter(DEFAULT_ROUTING_MAP, _MODEL))
    route = router.select_verifier(ModelFamily.LOCAL, available_families={"local", "openrouter"})
    assert route.family == ModelFamily.OPENROUTER
    assert route.model_id == _MODEL


def test_router_never_routes_openrouter_caller_to_openrouter():
    router = FamilyRouter(routing_map=with_openrouter(DEFAULT_ROUTING_MAP, _MODEL))
    route = router.select_verifier(ModelFamily.OPENROUTER)
    assert route.family != ModelFamily.OPENROUTER


# --- setup wiring (opt-in: BOTH env vars) ---


def _isolate_env(monkeypatch):
    for var in (
        "OPENROUTER_API_KEY",
        "PRISM_VERIFIER_MODEL_OPENROUTER",
        "PRISM_OPENROUTER_BASE_URL",
    ):
        monkeypatch.delenv(var, raising=False)


def test_seat_built_only_when_both_key_and_model_set(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", _MODEL)
    providers = build_providers_from_env()
    assert "openrouter" in providers
    assert providers["openrouter"].family == ModelFamily.OPENROUTER
    assert providers["openrouter"].available_models == [_MODEL]


def test_seat_not_built_when_only_key_set(monkeypatch):
    # A rig-wide OPENROUTER_API_KEY (for other tooling) must NOT perturb prism.
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert "openrouter" not in build_providers_from_env()


def test_seat_not_built_when_neither_set(monkeypatch):
    _isolate_env(monkeypatch)
    assert "openrouter" not in build_providers_from_env()


def test_colliding_model_fails_closed(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", "openai/gpt-4o")
    with pytest.raises(OpenRouterLineageError):
        build_providers_from_env()


def test_base_url_override(monkeypatch):
    _isolate_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", _MODEL)
    monkeypatch.setenv("PRISM_OPENROUTER_BASE_URL", "http://localhost:9999/api/v1")
    providers = build_providers_from_env()
    assert "9999" in str(providers["openrouter"]._client.base_url)


def test_build_default_engine_injects_seat_only_when_configured(monkeypatch):
    monkeypatch.setenv("PRISM_DEV", "1")  # engine construction needs a receipt-signing config
    _isolate_env(monkeypatch)
    eng = build_default_engine()
    assert ModelFamily.OPENROUTER not in eng._router._routing_map
    assert "openrouter" not in eng._providers

    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    monkeypatch.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", _MODEL)
    eng2 = build_default_engine()
    assert "openrouter" in eng2._providers
    # appended as the LAST (failover) route on every caller row
    assert eng2._router._routing_map[ModelFamily.LOCAL][-1] == (ModelFamily.OPENROUTER, _MODEL)

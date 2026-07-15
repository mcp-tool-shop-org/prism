"""OpenRouter model provider — a multi-vendor GATEWAY served as ONE cross-family verifier seat.

OpenRouter fronts many model families behind a single OpenAI-compatible endpoint, so it is a
cross-family verifier SUPPLY: one key, the whole catalog. But ``openrouter`` is a TRANSPORT, not a
lineage — Lock 1 keys on the producer's true family, so a seat that quietly serves (say) a Google
model while wearing the OPENROUTER label would let a Google caller be "verified" by a Google model
prism believes is cross-family (a silent same-family verdict, the exact failure Lock 1 exists to
prevent).

``validate_openrouter_lineage`` is the config-time guard that keeps the OPENROUTER family honest: it
REFUSES a configured model whose vendor prefix collides with a family prism already models natively
(anthropic / openai / google / mistral) or is non-deterministic (``openrouter/auto``), so the seat
can only ever carry a genuinely family-distinct model (deepseek / qwen / cohere / nvidia /
meta-llama / ...). Opt-in: the provider is built ONLY when BOTH ``OPENROUTER_API_KEY`` and
``PRISM_VERIFIER_MODEL_OPENROUTER`` are set (core/setup.py); ``PRISM_OPENROUTER_BASE_URL`` overrides
the endpoint.
"""

from __future__ import annotations

import time

import httpx

from prism.core.types import ModelFamily
from prism.providers.base import (
    CompletionRequest,
    CompletionResponse,
    ModelProvider,
    ProviderError,
    log_provider_failure,
    log_provider_success,
)

# The base already includes ``/v1`` (so requests POST to ``/chat/completions``) — matching the
# canonical OpenRouter base and the value an operator sets for other tooling.
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_NAME = "openrouter"

# Vendor prefixes whose lineage prism ALREADY models as a first-class family — anthropic / openai /
# google natively, and ``mistral``/``mistralai`` is the LOCAL lineage (mistral-small:24b). An
# OpenRouter seat wearing the OPENROUTER label while serving one of these would defeat Lock 1 for
# that caller family. ``openrouter`` itself (the auto-router) is non-deterministic — it can resolve
# to any of the above — so it is blocked too. Everything else (deepseek, qwen, cohere, nvidia,
# meta-llama, x-ai, ...) is genuinely distinct from every prism caller and is allowed.
BLOCKED_VENDORS = frozenset(
    {"anthropic", "openai", "google", "mistral", "mistralai", "openrouter"}
)


class OpenRouterLineageError(ValueError):
    """A configured OpenRouter model's lineage collides with a native prism family (Lock 1)."""


def validate_openrouter_lineage(model_id: str) -> None:
    """Refuse an OpenRouter model whose true lineage isn't distinct from every prism caller family.

    OpenRouter model ids are ``vendor/model``. The ``vendor`` prefix is the lineage signal Lock 1
    needs: a seat labeled OPENROUTER must carry a vendor prism does NOT otherwise model, or the
    cross-family guarantee is a lie for that caller. Raises ``OpenRouterLineageError`` (a config
    error — fail-closed at engine construction) on a colliding or indeterminate vendor.
    """
    vendor = model_id.split("/", 1)[0].strip().lower() if "/" in model_id else ""
    if not vendor:
        raise OpenRouterLineageError(
            f"PRISM_VERIFIER_MODEL_OPENROUTER={model_id!r} is not a 'vendor/model' id. OpenRouter "
            "routes by vendor prefix and prism needs it to keep the seat family-distinct — e.g. "
            "'deepseek/deepseek-chat' or 'qwen/qwen-2.5-72b-instruct'."
        )
    if vendor in BLOCKED_VENDORS:
        raise OpenRouterLineageError(
            f"PRISM_VERIFIER_MODEL_OPENROUTER={model_id!r} has vendor '{vendor}', whose lineage "
            "prism already models as a native family — labeling it OPENROUTER would defeat Lock 1 "
            f"for a '{vendor}' caller. Point the seat at a distinct family (deepseek / qwen / "
            "cohere / nvidia / meta-llama / ...), or wire that vendor via its own provider instead."
        )


class OpenRouterProvider(ModelProvider):
    """Serves a single, family-distinct OpenRouter model as a cross-family verifier seat."""

    def __init__(
        self,
        api_key: str,
        model_id: str,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._api_key = api_key
        self._model_id = model_id
        self._client = httpx.AsyncClient(
            base_url=base_url,
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    @property
    def family(self) -> ModelFamily:
        return ModelFamily.OPENROUTER

    @property
    def available_models(self) -> list[str]:
        return [self._model_id]

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        model_id = request.model_id or self._model_id
        start = time.monotonic()
        # The standard OpenAI chat shape (``max_tokens`` + ``temperature``); OpenRouter normalizes
        # it per underlying vendor. We do NOT use the GPT-5/o-series ``max_completion_tokens`` shape
        # the OpenAI adapter needs — an OpenRouter seat fronts arbitrary vendors.
        body: dict[str, object] = {
            "model": model_id,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
        }
        try:
            response = await self._client.post("/chat/completions", json=body)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            log_provider_failure(
                _NAME, model_id, e, latency_ms=elapsed, status=e.response.status_code
            )
            retryable = e.response.status_code in (429, 500, 502, 503)
            raise ProviderError(
                f"OpenRouter API error: {e.response.status_code}", retryable=retryable
            ) from e
        except httpx.TimeoutException as e:
            elapsed = int((time.monotonic() - start) * 1000)
            log_provider_failure(_NAME, model_id, e, latency_ms=elapsed)
            raise ProviderError("OpenRouter API timed out", retryable=True) from e
        except httpx.TransportError as e:
            elapsed = int((time.monotonic() - start) * 1000)
            log_provider_failure(_NAME, model_id, e, latency_ms=elapsed)
            raise ProviderError("OpenRouter API not reachable", retryable=True) from e

        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            data = response.json()
        except ValueError as e:
            log_provider_failure(
                _NAME, model_id, e, latency_ms=latency_ms, status=response.status_code
            )
            raise ProviderError(
                f"OpenRouter returned a non-JSON body (HTTP {response.status_code})",
                retryable=True,
            ) from e

        if isinstance(data, dict) and data.get("error"):
            err = ProviderError(f"OpenRouter API error: {data['error']}", retryable=False)
            log_provider_failure(
                _NAME, model_id, err, latency_ms=latency_ms, status=response.status_code
            )
            raise err
        choices = data.get("choices") if isinstance(data, dict) else None
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        # Empty content is a real OpenRouter failure mode (some free/code models return ""); for a
        # verifier an empty completion is unusable, so RAISE (retryable) -> the circuit-breaker
        # fails over to another route, never a silent vacuous adjudication.
        if not isinstance(content, str) or not content.strip():
            err = ProviderError(
                "OpenRouter returned a malformed or empty response "
                "(missing choices[0].message.content)",
                retryable=True,
            )
            log_provider_failure(
                _NAME, model_id, err, latency_ms=latency_ms, status=response.status_code
            )
            raise err
        usage = data.get("usage", {}) if isinstance(data, dict) else {}

        log_provider_success(_NAME, model_id, latency_ms=latency_ms)
        return CompletionResponse(
            content=content,
            model_id=model_id,
            latency_ms=latency_ms,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
        )

    async def health_check(self) -> bool:
        try:
            response = await self._client.get("/models")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def close(self) -> None:
        await self._client.aclose()

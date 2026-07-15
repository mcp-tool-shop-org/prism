"""The F-14 verifier registry (``PRISM_VERIFIER_MODEL_*``) must hold on EVERY transport.

``resolve_routing_map`` shipped reachable through ``build_default_engine`` alone — and only
``prism.mcp.server`` called it. ``prism verify`` and the HTTP API each hand-built a
``VerificationEngine`` with a default ``FamilyRouter``, so both silently served
``DEFAULT_ROUTING_MAP``'s hardcoded model ids no matter what the operator configured: a per-seat pin
LOOKED applied and was not (a consumer building a verifier panel got N identical jurors wearing N
name tags). Nothing pinned the contract on any surface but MCP — which is exactly why it shipped.

These tests pin all three. ``test_pin_reaches_the_selected_route`` is the regression proper: one
assertion, three transports, and it fails on cli/http before the fix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import click
import pytest
from click.testing import CliRunner

from prism.cli.main import _build_verify_engine, cli
from prism.core.engine import VerificationEngine
from prism.core.setup import build_default_engine
from prism.core.types import ModelFamily
from prism.http import create_app
from prism.receipts.store import ReceiptStore

if TYPE_CHECKING:
    from pathlib import Path

_PIN = "qwen2.5:7b"
_OR_MODEL = "deepseek/deepseek-chat"
_DEFAULT_LOCAL = "mistral-small:24b"

# Every env knob that can add a provider or move a route. The rig running these tests really does
# export OPENROUTER_API_KEY, so an un-scrubbed env would let an ambient key decide the assertions.
_AMBIENT = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "PRISM_LOCAL_VERIFIER_ENDPOINT",
    "PRISM_LOCAL_VERIFIER_MODEL",
    "PRISM_SYCOPHANCY_ENDPOINT",
    "PRISM_ANTHROPIC_BASE_URL",
    "PRISM_OPENAI_BASE_URL",
    "PRISM_GOOGLE_BASE_URL",
    "PRISM_OPENROUTER_BASE_URL",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """A hermetic env: no ambient provider keys, no registry pins, dev receipt signing."""
    for var in _AMBIENT:
        monkeypatch.delenv(var, raising=False)
    for family in ModelFamily:
        monkeypatch.delenv(f"PRISM_VERIFIER_MODEL_{family.name}", raising=False)
    monkeypatch.setenv("PRISM_DEV", "1")
    monkeypatch.setenv("PRISM_HTTP_ALLOW_NO_AUTH", "1")
    return monkeypatch


def _selected(engine: VerificationEngine, caller: ModelFamily) -> tuple[str, str]:
    """The route the engine would ACTUALLY use — i.e. what the receipt's verifier_models records."""
    route = engine._router.select_verifier(caller, available_families=set(engine._providers))
    return route.family.value, route.model_id


def _http_engine(tmp_path: Path) -> VerificationEngine:
    store = ReceiptStore(db_path=tmp_path / "http.db", signing_secret=b"test-secret")
    engine = create_app(store=store).state.engine
    assert isinstance(engine, VerificationEngine)
    return engine


class TestRegistryHonoredOnEverySurface:
    """The regression this change exists for: a pin must reach every transport's routing map."""

    @pytest.mark.parametrize("surface", ["cli", "http", "mcp"])
    def test_pin_reaches_the_selected_route(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path, surface: str
    ) -> None:
        clean_env.setenv("PRISM_VERIFIER_MODEL_LOCAL", _PIN)
        build = {
            "cli": lambda: _build_verify_engine(("ollama",)),
            "http": lambda: _http_engine(tmp_path),
            "mcp": build_default_engine,
        }[surface]
        # caller=anthropic routes GOOGLE -> OPENAI -> LOCAL; only local has a provider here, so the
        # router walks past the hosted families and lands on the pinned local seat.
        assert _selected(build(), ModelFamily.ANTHROPIC) == ("local", _PIN)

    @pytest.mark.parametrize("surface", ["cli", "http", "mcp"])
    def test_unset_registry_leaves_the_shipped_default(
        self, clean_env: pytest.MonkeyPatch, tmp_path: Path, surface: str
    ) -> None:
        """The byte-identical contract: an unconfigured registry must not perturb any transport."""
        build = {
            "cli": lambda: _build_verify_engine(("ollama",)),
            "http": lambda: _http_engine(tmp_path),
            "mcp": build_default_engine,
        }[surface]
        assert _selected(build(), ModelFamily.ANTHROPIC) == ("local", _DEFAULT_LOCAL)


class TestProviderAllowlist:
    def test_ollama_stays_free_when_cloud_keys_are_ambient(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        """``--provider ollama`` must be free BY CONSTRUCTION, not by convention.

        For caller=anthropic the route order is GOOGLE -> OPENAI -> LOCAL, so if the allowlist let
        ambient keys through, a stray GOOGLE_API_KEY would silently redirect the run to
        gemini-2.5-pro and BILL for it, with no log line. This is precisely why the CLI filters
        ``build_providers_from_env()`` rather than calling ``build_default_engine()``.
        """
        clean_env.setenv("GOOGLE_API_KEY", "ambient")
        clean_env.setenv("ANTHROPIC_API_KEY", "ambient")
        clean_env.setenv("OPENAI_API_KEY", "ambient")
        engine = _build_verify_engine(("ollama",))
        assert set(engine._providers) == {"local"}
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("local", _DEFAULT_LOCAL)

    def test_local_verifier_not_injected_unless_named(self, clean_env: pytest.MonkeyPatch) -> None:
        """PRISM_LOCAL_VERIFIER_ENDPOINT must not silently install a PRIMARY verifier.

        ``build_default_engine`` PREPENDS the specialist ahead of every caller's routes. Gating the
        injection on the filtered provider set means a run scoped to ollama cannot acquire a
        verifier it never asked for — the ambient-env hazard, one step further in than the keys.
        """
        clean_env.setenv("PRISM_LOCAL_VERIFIER_ENDPOINT", "http://localhost:9/v1")
        engine = _build_verify_engine(("ollama",))
        assert "local-verifier" not in engine._providers
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("local", _DEFAULT_LOCAL)

    def test_named_provider_without_config_errors_by_name(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(click.ClickException) as exc:
            _build_verify_engine(("anthropic",))
        assert "ANTHROPIC_API_KEY" in str(exc.value)

    def test_openrouter_seat_is_reachable_from_the_cli(self, clean_env: pytest.MonkeyPatch) -> None:
        """The regression the OpenRouter seat (#13) never had.

        It shipped MCP-only: ``_run_verify`` hand-built its provider map so the provider was never
        registered, AND ``with_openrouter`` was only called inside ``build_default_engine`` so the
        seat never entered the routing map — doubly unreachable from ``prism verify``. Routing the
        CLI through the shared factory is what completes it.
        """
        clean_env.setenv("OPENROUTER_API_KEY", "k")
        clean_env.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", _OR_MODEL)
        engine = _build_verify_engine(("openrouter",))
        assert set(engine._providers) == {"openrouter"}
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("openrouter", _OR_MODEL)

    def test_multiple_providers_give_the_cli_real_failover(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        clean_env.setenv("OPENROUTER_API_KEY", "k")
        clean_env.setenv("PRISM_VERIFIER_MODEL_OPENROUTER", _OR_MODEL)
        engine = _build_verify_engine(("ollama", "openrouter"))
        assert set(engine._providers) == {"local", "openrouter"}
        # local is primary for an anthropic caller; openrouter is the appended failover behind it.
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("local", _DEFAULT_LOCAL)
        engine._router.report_failure(ModelFamily.LOCAL, _DEFAULT_LOCAL)
        engine._router.report_failure(ModelFamily.LOCAL, _DEFAULT_LOCAL)
        engine._router.report_failure(ModelFamily.LOCAL, _DEFAULT_LOCAL)
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("openrouter", _OR_MODEL)

    def test_unknown_provider_is_a_usage_error_not_a_silent_refusal(
        self, clean_env: pytest.MonkeyPatch
    ) -> None:
        """``--provider`` was a bare str: an unknown value registered nothing and dead-ended on
        VERIFIER_UNAVAILABLE, so `--provider openai` read as 'openai refused' rather than 'typo'."""
        result = CliRunner().invoke(cli, ["verify", "-a", "x", "-i", "y", "--provider", "gpt-9000"])
        assert result.exit_code == 2
        assert "gpt-9000" in result.output


class TestVerifierModelFlag:
    def test_flag_pins_the_route(self, clean_env: pytest.MonkeyPatch) -> None:
        engine = _build_verify_engine(("ollama",), ("local=" + _PIN,))
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("local", _PIN)

    def test_flag_beats_env(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("PRISM_VERIFIER_MODEL_LOCAL", "from-env")
        engine = _build_verify_engine(("ollama",), ("local=from-flag",))
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("local", "from-flag")

    def test_flag_is_repeatable_across_families(self, clean_env: pytest.MonkeyPatch) -> None:
        clean_env.setenv("GOOGLE_API_KEY", "k")
        engine = _build_verify_engine(
            ("ollama", "google"), ("local=" + _PIN, "google=gemini-3-pro")
        )
        assert _selected(engine, ModelFamily.ANTHROPIC) == ("google", "gemini-3-pro")
        assert _selected(engine, ModelFamily.GOOGLE) == ("local", _PIN)

    @pytest.mark.parametrize("bad", ["local", "=x", "local=", "", "   ", "nosuchfamily=m"])
    def test_malformed_pair_is_rejected(self, clean_env: pytest.MonkeyPatch, bad: str) -> None:
        with pytest.raises(click.BadParameter):
            _build_verify_engine(("ollama",), (bad,))

    @pytest.mark.parametrize("family", ["openrouter", "local-verifier", "local-sycophancy"])
    def test_provider_configured_families_are_refused_not_dropped(
        self, clean_env: pytest.MonkeyPatch, family: str
    ) -> None:
        """These families take their model from the PROVIDER (the OpenRouter lineage guard validates
        it at construction), so a routing-map override would be silently discarded. Silently
        discarded config is the exact bug this change exists to kill — refuse it loudly instead."""
        with pytest.raises(click.BadParameter) as exc:
            _build_verify_engine(("ollama",), (f"{family}=x",))
        assert "PRISM_VERIFIER_MODEL_" in str(exc.value)

"""Tests for the prism CLI receipt commands and duration parsing."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from click.testing import CliRunner

from prism.cli.main import _parse_duration, cli
from prism.core.types import ReasoningVisibility
from prism.receipts.store import ReceiptStore


def _seed_receipt(db_path, age: timedelta | None = None) -> str:
    """Store one receipt, optionally BACKDATED by ``age``.

    ``create_receipt`` always stamps ``datetime.now(UTC)``, so an age has to be written after the
    fact. Backdating is what makes a prune test deterministic: ``prune`` deletes on a strict
    ``timestamp < now - older_than``, so a receipt stamped in the same clock tick as the cutoff is
    NOT pruned. Giving the row a real age removes that dependency instead of widening it, and lets
    a test pin BOTH sides of the cutoff — which a zero cutoff cannot, since it makes every receipt
    a candidate.

    The backdated row's signature still covers its ORIGINAL timestamp. That is fine here and only
    here: ``prune`` is a time-range DELETE and never validates a signature. Don't reuse an aged
    receipt in a test that verifies one.
    """
    store = ReceiptStore(db_path=db_path, signing_secret=b"test-secret")
    r = store.create_receipt(
        pre_strip_hash="a",
        post_strip_hash="b",
        verifier_models=["m"],
        pairwise_rho={},
        reasoning_visibility_mode=ReasoningVisibility.STRIPPED,
        verdict="accept",
        confidence=0.9,
        retryable=False,
        lens_results_json="[]",
    )
    if age is not None:
        store._conn.execute(
            "UPDATE receipts SET timestamp = ? WHERE id = ?",
            ((datetime.now(UTC) - age).isoformat(), r.id),
        )
        store._conn.commit()
    store.close()
    return r.id


class TestParseDuration:
    @pytest.mark.parametrize(
        "text,seconds",
        [("90d", 90 * 86400), ("24h", 24 * 3600), ("30m", 1800), ("45s", 45), ("2w", 2 * 604800)],
    )
    def test_valid(self, text, seconds):
        assert _parse_duration(text) == timedelta(seconds=seconds)

    @pytest.mark.parametrize("bad", ["", "90", "10x", "abc", "d"])
    def test_invalid_raises(self, bad):
        with pytest.raises(Exception):
            _parse_duration(bad)


class TestReceiptCli:
    def test_delete_existing(self, tmp_path, monkeypatch):
        db = tmp_path / "cli.db"
        rid = _seed_receipt(db)
        monkeypatch.setenv("PRISM_DEV", "1")
        monkeypatch.setattr("prism.receipts.store.DEFAULT_DB_PATH", db)
        result = CliRunner().invoke(cli, ["receipt", "delete", rid])
        assert result.exit_code == 0
        assert json.loads(result.output)["deleted"] == rid

    def test_delete_missing_exits_1(self, tmp_path, monkeypatch):
        db = tmp_path / "cli.db"
        _seed_receipt(db)
        monkeypatch.setenv("PRISM_DEV", "1")
        monkeypatch.setattr("prism.receipts.store.DEFAULT_DB_PATH", db)
        result = CliRunner().invoke(cli, ["receipt", "delete", "prism-nope"])
        assert result.exit_code == 1

    def test_prune_requires_yes(self, tmp_path, monkeypatch):
        db = tmp_path / "cli.db"
        _seed_receipt(db)
        monkeypatch.setenv("PRISM_DEV", "1")
        monkeypatch.setattr("prism.receipts.store.DEFAULT_DB_PATH", db)
        result = CliRunner().invoke(cli, ["receipt", "prune", "--older-than", "1d"])
        assert result.exit_code == 1
        # Without --yes, nothing is deleted.
        store = ReceiptStore(db_path=db, signing_secret=b"test-secret")
        remaining = store._conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]
        store.close()
        assert remaining == 1

    def test_prune_with_yes_removes_old(self, tmp_path, monkeypatch):
        # A 2-day-old receipt against a 1-day cutoff: unambiguously old, whatever the clock's
        # resolution. The old form seeded a receipt at `now` and pruned `--older-than 0s`, i.e.
        # asked whether `now < now` — true only if the clock ticked in between.
        db = tmp_path / "cli.db"
        _seed_receipt(db, age=timedelta(days=2))
        monkeypatch.setenv("PRISM_DEV", "1")
        monkeypatch.setattr("prism.receipts.store.DEFAULT_DB_PATH", db)
        result = CliRunner().invoke(cli, ["receipt", "prune", "--older-than", "1d", "--yes"])
        assert result.exit_code == 0
        assert json.loads(result.output)["pruned"] == 1

    def test_prune_keeps_receipts_newer_than_the_cutoff(self, tmp_path, monkeypatch):
        """The other half of the boundary: prune must not take a receipt inside the window.

        `--older-than 0s` could never express this — it made every receipt a prune candidate and
        the assertion rode on a clock tick. With a real age the window has two sides worth pinning.
        """
        db = tmp_path / "cli.db"
        _seed_receipt(db, age=timedelta(hours=1))
        monkeypatch.setenv("PRISM_DEV", "1")
        monkeypatch.setattr("prism.receipts.store.DEFAULT_DB_PATH", db)
        result = CliRunner().invoke(cli, ["receipt", "prune", "--older-than", "1d", "--yes"])
        assert result.exit_code == 0
        assert json.loads(result.output)["pruned"] == 0


class TestVerifyGate:
    """--gate maps the verdict to a shell exit code (opt-in; the default stays 0)."""

    def _fake_response(self, verdict, tmp_path):
        from prism.core.types import Verdict, VerifyResponse

        store = ReceiptStore(db_path=tmp_path / "g.db", signing_secret=b"s")
        receipt = store.create_receipt(
            pre_strip_hash="a",
            post_strip_hash="b",
            verifier_models=["m"],
            pairwise_rho={},
            reasoning_visibility_mode=ReasoningVisibility.STRIPPED,
            verdict=verdict,
            confidence=0.9,
            retryable=False,
            lens_results_json="[]",
        )
        store.close()
        return VerifyResponse(
            verdict=Verdict(verdict),
            confidence=0.9,
            retryable=False,
            lens_results=[],
            pairwise_rho={},
            receipt=receipt,
        )

    @pytest.mark.parametrize(
        "verdict,code", [("accept", 0), ("revise", 10), ("refuse", 20), ("escalate", 30)]
    )
    def test_gate_exit_codes(self, verdict, code, tmp_path, monkeypatch):
        from prism.cli import main as cli_main

        resp = self._fake_response(verdict, tmp_path)

        async def _fake(request, provider_names, verifier_models):
            return resp

        monkeypatch.setattr(cli_main, "_run_verify", _fake)
        result = CliRunner().invoke(cli, ["verify", "-a", "x", "-i", "y", "--gate"])
        assert result.exit_code == code

    def test_no_gate_stays_zero_on_nonaccept(self, tmp_path, monkeypatch):
        from prism.cli import main as cli_main

        resp = self._fake_response("refuse", tmp_path)

        async def _fake(request, provider_names, verifier_models):
            return resp

        monkeypatch.setattr(cli_main, "_run_verify", _fake)
        result = CliRunner().invoke(cli, ["verify", "-a", "x", "-i", "y"])  # no --gate
        assert result.exit_code == 0

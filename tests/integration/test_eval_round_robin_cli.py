"""Integration: `prism eval --round-robin` drives the within-judge family-AB loop end-to-end.

Builds a tiny offline family-AB corpus (familygen + a fake generator), then runs the round-robin
CLI over it: per-family bypass engines -> run_round_robin -> compute_self_preference -> a report
+ a signed run-receipt. Offline (MockProvider verifiers) proves the WIRING at zero cost — numbers
are machinery, not a measurement (a real signal needs real models on a harder corpus).
"""

from __future__ import annotations

import asyncio
import json

from click.testing import CliRunner

from prism.cli.main import cli
from prism.eval.familygen import FamilySpec, ProblemSpec, build_family_corpus

_PROBLEM = ProblemSpec(
    id="add",
    intent="Return the sum of a and b.",
    entry_point="add",
    test_code=(
        "def check(candidate):\n    assert candidate(1, 2) == 3\n    assert candidate(0, 0) == 0\n"
    ),
)


def _build_offline_familyab(tmp_path):
    async def gen(model_id: str, system: str, user: str) -> str:
        return "def add(a, b):\n    return a + b\n"  # both families: a clean gen (+ a mutant)

    d = tmp_path / "fab"
    asyncio.run(
        build_family_corpus(
            d,
            [FamilySpec("fa", "model-a"), FamilySpec("fb", "model-b")],
            problems=[_PROBLEM],
            generate_fn=gen,
            deconfound=False,
            mutants_per_clean=1,
        )
    )
    return d


def test_round_robin_offline_publishes_report_and_signed_receipt(tmp_path, monkeypatch) -> None:
    for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    fab = _build_offline_familyab(tmp_path)
    out = tmp_path / "rr"
    res = CliRunner().invoke(
        cli,
        [
            "eval", "--round-robin",
            "--familyab-corpus", str(fab),
            "--offline",
            "--sesoi", "0.05",
            "--min-problems", "1",
            "--out", str(out),
        ],
    )
    assert res.exit_code == 0, res.output

    md = (out / "round_robin.md").read_text(encoding="utf-8")
    assert "within-judge round-robin" in md
    assert "self_preference" in md
    assert "decision:" in md

    result = json.loads((out / "round_robin.json").read_text(encoding="utf-8"))
    assert "aggregate_self_preference" in result
    assert "per_family" in result and result["decision"]

    receipt = json.loads((out / "round-robin-receipt.json").read_text(encoding="utf-8"))
    assert receipt["signature_valid"] is True
    assert receipt["artifact_type"] == "familyab_round_robin"


def test_round_robin_requires_a_corpus_dir(tmp_path) -> None:
    res = CliRunner().invoke(cli, ["eval", "--round-robin", "--offline", "--out", str(tmp_path)])
    assert res.exit_code != 0
    assert "familyab-corpus" in res.output

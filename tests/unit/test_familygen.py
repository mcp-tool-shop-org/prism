"""Tests for the family-AB generation pipeline (offline, injected generator, zero spend)."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from prism.eval.familygen import (
    FamilySpec,
    ProblemSpec,
    build_family_corpus,
    contamination_overlap,
)

_ADD = ProblemSpec(
    id="add",
    intent="Add two integers a and b and return the sum.",
    entry_point="add",
    test_code=(
        "def check(candidate):\n"
        "    assert candidate(1, 2) == 3\n"
        "    assert candidate(-1, 1) == 0\n"
    ),
)

_CORRECT_ADD = "def add(a, b):\n    return a + b\n"
_BUGGY_ADD = "def add(a, b):\n    return a * b\n"


def _make_gen(solutions: dict[tuple[str, str], str]) -> Callable[[str, str, str], Awaitable[str]]:
    async def gen(model_id: str, system: str, user: str) -> str:
        match = re.search(r"named `([^`]+)`", user)
        entry = match.group(1) if match else ""
        return solutions.get((model_id, entry), "")

    return gen


async def test_clean_and_buggy_paths(tmp_path: Path) -> None:
    gen = _make_gen({("good-model", "add"): _CORRECT_ADD, ("bad-model", "add"): _BUGGY_ADD})
    manifest = await build_family_corpus(
        tmp_path,
        [FamilySpec("good", "good-model"), FamilySpec("bad", "bad-model")],
        problems=[_ADD],
        generate_fn=gen,
        mutants_per_clean=2,
    )

    samples = [json.loads(line) for line in (tmp_path / "fresh.jsonl").read_text().splitlines()]
    by_id = {s["id"]: s for s in samples}

    # good family: a clean generation + at least one execution-verified mutant bug.
    assert by_id["fam-good-add-gen-clean"]["positive"] is False
    assert by_id["fam-good-add-gen-clean"]["expected_verdict"] == "accept"
    mut_ids = [s for s in by_id if s.startswith("fam-good-add-mut")]
    assert mut_ids
    for mid in mut_ids:
        assert by_id[mid]["positive"] is True
        assert by_id[mid]["bug_class"].startswith("mut_")

    # bad family: a natural buggy generation (failed the hidden tests).
    assert by_id["fam-bad-add-gen-bug"]["positive"] is True
    assert by_id["fam-bad-add-gen-bug"]["bug_class"].startswith("natural_")

    # provenance sidecar maps every emitted sample to its producing family.
    prov = manifest["provenance"]
    assert isinstance(prov, dict)
    assert prov["fam-good-add-gen-clean"] == "good"
    assert prov["fam-bad-add-gen-bug"] == "bad"
    assert all(prov[mid] == "good" for mid in mut_ids)
    assert set(prov) == set(by_id)


async def test_mutants_capped_per_clean(tmp_path: Path) -> None:
    gen = _make_gen({("m", "add"): _CORRECT_ADD})
    await build_family_corpus(
        tmp_path, [FamilySpec("f", "m")], problems=[_ADD], generate_fn=gen, mutants_per_clean=1
    )
    samples = [json.loads(line) for line in (tmp_path / "fresh.jsonl").read_text().splitlines()]
    mut = [s for s in samples if s["id"].startswith("fam-f-add-mut")]
    assert len(mut) <= 1


async def test_empty_generation_emits_no_sample(tmp_path: Path) -> None:
    gen = _make_gen({})  # returns "" for everything
    manifest = await build_family_corpus(
        tmp_path, [FamilySpec("f", "m")], problems=[_ADD], generate_fn=gen
    )
    assert manifest["n_samples"] == 0
    counts = manifest["counts_by_family"]
    assert isinstance(counts, dict)
    assert counts["f"]["empty"] == 1


async def test_content_hash_is_deterministic(tmp_path: Path) -> None:
    gen = _make_gen({("good-model", "add"): _CORRECT_ADD})
    fams = [FamilySpec("good", "good-model")]
    m1 = await build_family_corpus(tmp_path / "a", fams, problems=[_ADD], generate_fn=gen)
    m2 = await build_family_corpus(tmp_path / "b", fams, problems=[_ADD], generate_fn=gen)
    assert m1["content_hash"] == m2["content_hash"]


async def test_contamination_drop(tmp_path: Path) -> None:
    gen = _make_gen({("good-model", "add"): _CORRECT_ADD})
    manifest = await build_family_corpus(
        tmp_path,
        [FamilySpec("good", "good-model")],
        problems=[_ADD],
        generate_fn=gen,
        contamination_refs=[_CORRECT_ADD],  # the generation is identical to a known reference
        contamination_max_run=3,
    )
    assert manifest["n_samples"] == 0
    counts = manifest["counts_by_family"]
    assert isinstance(counts, dict)
    assert counts["good"]["dropped_contaminated"] == 1


def test_contamination_overlap_basic() -> None:
    assert contamination_overlap("def add(a, b): return a + b", []) == 0
    high = contamination_overlap("def add(a, b): return a + b", ["def add(a, b): return a + b"])
    assert high >= 5


async def test_problem_of_maps_every_sample(tmp_path: Path) -> None:
    gen = _make_gen({("good-model", "add"): _CORRECT_ADD})
    manifest = await build_family_corpus(
        tmp_path,
        [FamilySpec("good", "good-model")],
        problems=[_ADD],
        generate_fn=gen,
        mutants_per_clean=1,
    )
    problem_of = manifest["problem_of"]
    provenance = manifest["provenance"]
    assert isinstance(problem_of, dict)
    assert isinstance(provenance, dict)
    assert problem_of  # non-empty
    assert all(p == "add" for p in problem_of.values())  # all from the one seed problem
    assert set(problem_of) == set(provenance)  # every provenanced sample is clustered


async def test_unrunnable_artifact_is_skipped_not_labeled_buggy(tmp_path: Path) -> None:
    # a generation importing a missing module is UNRUNNABLE (ModuleNotFoundError), not a code bug:
    # it must be skipped, never emitted as a natural bug (which would corrupt the corpus)
    bad_import = "import definitely_not_a_real_module_xyz\ndef add(a, b):\n    return a + b\n"
    gen = _make_gen({("m", "add"): bad_import})
    manifest = await build_family_corpus(
        tmp_path, [FamilySpec("f", "m")], problems=[_ADD], generate_fn=gen
    )
    assert manifest["n_samples"] == 0
    counts = manifest["counts_by_family"]
    assert isinstance(counts, dict)
    assert counts["f"]["unrunnable"] == 1
    assert counts["f"].get("natural_bug", 0) == 0  # NOT mislabeled buggy


async def test_perplexity_covariate_is_recorded(tmp_path: Path) -> None:
    gen = _make_gen({("good-model", "add"): _CORRECT_ADD})
    manifest = await build_family_corpus(
        tmp_path,
        [FamilySpec("good", "good-model")],
        problems=[_ADD],
        generate_fn=gen,
        mutants_per_clean=1,
        perplexity_fn=lambda code: float(len(code)),
    )
    perp = manifest["perplexity"]
    assert isinstance(perp, dict)
    per_sample = perp["per_sample"]
    assert isinstance(per_sample, dict)
    # the generated code is stripped by _extract_code before scoring
    assert per_sample["fam-good-add-gen-clean"] == float(len(_CORRECT_ADD.strip()))


def test_seed_problems_are_well_formed() -> None:
    from prism.eval._familygen_problems import FRESH_PROBLEMS

    assert len(FRESH_PROBLEMS) == 10
    ids = [p.id for p in FRESH_PROBLEMS]
    assert len(set(ids)) == len(ids)
    for p in FRESH_PROBLEMS:
        assert p.entry_point
        assert "def check(candidate):" in p.test_code

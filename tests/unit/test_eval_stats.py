"""Tests for the family-AB statistical primitives (mid-p McNemar, BH-FDR, cluster bootstrap)."""

from __future__ import annotations

from collections.abc import Sequence

from prism.eval.metrics import benjamini_hochberg, cluster_bootstrap_ci, mcnemar_midp


def test_mcnemar_midp_symmetric_is_one() -> None:
    assert mcnemar_midp(0, 0) == 1.0
    assert mcnemar_midp(5, 5) == 1.0  # equal discordance => no evidence of a difference


def test_mcnemar_midp_extreme_is_small() -> None:
    assert mcnemar_midp(10, 0) < 0.01
    assert mcnemar_midp(0, 10) < 0.01  # two-sided: direction-symmetric


def test_mcnemar_midp_monotone_in_imbalance() -> None:
    # more discordant imbalance (holding n) => stronger evidence => smaller p
    assert mcnemar_midp(9, 1) < mcnemar_midp(7, 3) < mcnemar_midp(6, 4)


def test_mcnemar_midp_is_less_conservative_than_exact() -> None:
    # mid-p subtracts half the point mass, so it is strictly smaller than the exact-conditional p
    b, c, n = 8, 2, 10
    half = 0.5**n
    exact = min(1.0, 2.0 * sum(__import__("math").comb(n, i) for i in range(min(b, c) + 1)) * half)
    assert mcnemar_midp(b, c) < exact


def test_bh_all_significant() -> None:
    assert benjamini_hochberg([0.001, 0.002, 0.003], q=0.1) == [True, True, True]


def test_bh_none_significant() -> None:
    assert benjamini_hochberg([0.5, 0.6, 0.7], q=0.1) == [False, False, False]


def test_bh_mixed_and_order_preserving() -> None:
    # m=3, q=0.1: ranks pass for p<=0.04, fail for 0.9 -> threshold 0.04
    assert benjamini_hochberg([0.01, 0.04, 0.9], q=0.1) == [True, True, False]
    # aligned to INPUT order, not sorted order
    assert benjamini_hochberg([0.9, 0.01, 0.04], q=0.1) == [False, True, True]


def test_bh_empty() -> None:
    assert benjamini_hochberg([], q=0.1) == []


def _mean(xs: Sequence[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def test_cluster_bootstrap_constant_is_tight() -> None:
    lo, hi = cluster_bootstrap_ci([1.0] * 12, _mean)
    assert lo == 1.0 and hi == 1.0


def test_cluster_bootstrap_brackets_the_mean() -> None:
    clusters = [float(i) for i in range(20)]  # mean 9.5
    lo, hi = cluster_bootstrap_ci(clusters, _mean, n_boot=500, seed=0)
    assert lo < 9.5 < hi


def test_cluster_bootstrap_is_deterministic_given_seed() -> None:
    clusters = [float(i) for i in range(20)]
    a = cluster_bootstrap_ci(clusters, _mean, n_boot=300, seed=7)
    b = cluster_bootstrap_ci(clusters, _mean, n_boot=300, seed=7)
    assert a == b


def test_cluster_bootstrap_empty_and_all_none() -> None:
    assert cluster_bootstrap_ci([], _mean) == (0.0, 0.0)
    assert cluster_bootstrap_ci([1.0, 2.0], lambda _xs: None) == (0.0, 0.0)

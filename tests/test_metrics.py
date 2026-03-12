from __future__ import annotations

import math

import pandas as pd

from src.metrics import (
    bootstrap_ci_mean,
    cohen_d,
    distance_to_frontier,
    distance_to_point,
    frontier_optimal_welfare,
    ks_reference_from_frontier,
    money_left_on_table,
    pairwise_significance_tests,
    summarize_results,
    welfare_regret,
)


def test_distance_to_frontier_on_simple_front() -> None:
    frontier = [(0.0, 1.0), (1.0, 0.0)]
    point = (0.5, 0.5)
    dist = distance_to_frontier(point, frontier)
    assert math.isclose(dist, math.sqrt(0.5), rel_tol=1e-9)


def test_summarize_results_adds_ci_columns() -> None:
    df = pd.DataFrame(
        [
            {"profile_id": "A", "n_steps": 20, "strategy_buyer": "X", "strategy_seller": "Y", "agreement": 1, "rounds": 4},
            {"profile_id": "A", "n_steps": 20, "strategy_buyer": "X", "strategy_seller": "Y", "agreement": 0, "rounds": 8},
        ]
    )
    summary = summarize_results(
        df,
        group_cols=["profile_id", "n_steps", "strategy_buyer", "strategy_seller"],
        metric_cols=["agreement", "rounds"],
        bootstrap_n_resamples=500,
        bootstrap_seed=13,
    )
    assert "agreement_ci95" in summary.columns
    assert "rounds_ci95" in summary.columns
    assert "agreement_boot_ci_low" in summary.columns
    assert "agreement_boot_ci_high" in summary.columns
    assert len(summary) == 1


def test_frontier_welfare_and_regret_metrics() -> None:
    frontier = [(0.1, 0.9), (0.8, 0.7)]
    optimal = frontier_optimal_welfare(frontier)
    assert math.isclose(optimal, 1.5, rel_tol=1e-9)

    welfare = 1.2
    assert math.isclose(money_left_on_table(welfare, optimal), 0.3, rel_tol=1e-9)
    assert math.isclose(welfare_regret(welfare, optimal), 0.2, rel_tol=1e-9)


def test_distance_to_point_handles_missing_reference() -> None:
    point = (0.5, 0.5)
    assert math.isnan(distance_to_point(point, None))


def test_bootstrap_ci_mean_singleton_value() -> None:
    low, high = bootstrap_ci_mean([0.7], n_resamples=100)
    assert math.isclose(low, 0.7, rel_tol=1e-9)
    assert math.isclose(high, 0.7, rel_tol=1e-9)


def test_pairwise_significance_and_effect_size() -> None:
    df = pd.DataFrame(
        [
            {"profile_id": "A", "n_steps": 12, "role": "buyer", "strategy": "S1", "own_utility": 0.9},
            {"profile_id": "A", "n_steps": 12, "role": "buyer", "strategy": "S1", "own_utility": 0.8},
            {"profile_id": "A", "n_steps": 12, "role": "buyer", "strategy": "S2", "own_utility": 0.2},
            {"profile_id": "A", "n_steps": 12, "role": "buyer", "strategy": "S2", "own_utility": 0.3},
        ]
    )
    out = pairwise_significance_tests(
        df,
        context_cols=["profile_id", "n_steps", "role"],
        strategy_col="strategy",
        metric_cols=["own_utility"],
        n_permutations=300,
        n_bootstrap=300,
        random_seed=7,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["strategy_a"] == "S1"
    assert row["strategy_b"] == "S2"
    assert row["mean_diff"] > 0
    assert "p_value_fdr" in out.columns
    assert row["effect_size_cohen_d"] == cohen_d([0.9, 0.8], [0.2, 0.3])


def test_cohen_d_zero_variance_fallback() -> None:
    assert cohen_d([1.0, 1.0], [1.0, 1.0]) == 0.0


def test_ks_reference_from_frontier_returns_pareto_point() -> None:
    frontier = [(0.2, 0.9), (0.6, 0.7), (0.9, 0.2)]
    disagreement = (0.1, 0.1)
    ks = ks_reference_from_frontier(frontier, disagreement)
    assert ks in frontier

from __future__ import annotations

import math
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def euclidean_distance(a: Iterable[float], b: Iterable[float]) -> float:
    a_vals = tuple(float(x) for x in a)
    b_vals = tuple(float(x) for x in b)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a_vals, b_vals)))


def social_welfare(point: Iterable[float]) -> float:
    return float(sum(float(x) for x in point))


def distance_to_point(point: Iterable[float], reference: Iterable[float] | None) -> float:
    if reference is None:
        return float("nan")
    return euclidean_distance(point, reference)


def distance_to_frontier(point: Iterable[float], frontier: Iterable[Iterable[float]]) -> float:
    point_vals = tuple(float(x) for x in point)
    frontier_vals = [tuple(float(y) for y in x) for x in frontier]
    if not frontier_vals:
        return float("nan")
    return min(euclidean_distance(point_vals, candidate) for candidate in frontier_vals)


def ks_reference_from_frontier(
    frontier: Iterable[Iterable[float]], disagreement_point: Iterable[float]
) -> tuple[float, float] | None:
    """
    Approximates a KS-style bargaining reference from frontier utilities.

    Chooses the Pareto point maximizing the minimum normalized gain from the
    disagreement point toward the ideal frontier point.
    """
    frontier_vals = [tuple(float(y) for y in x) for x in frontier]
    if not frontier_vals:
        return None
    d = tuple(float(v) for v in disagreement_point)
    ideal = (
        max(p[0] for p in frontier_vals),
        max(p[1] for p in frontier_vals),
    )
    denom = (ideal[0] - d[0], ideal[1] - d[1])

    def score(point: tuple[float, float]) -> tuple[float, float]:
        gains = []
        for i in range(2):
            if denom[i] <= 1e-12:
                gains.append(1.0 if point[i] >= ideal[i] - 1e-12 else -1e9)
            else:
                gains.append((point[i] - d[i]) / denom[i])
        min_gain = min(gains)
        # Tie-break by preferring larger total normalized gain.
        return min_gain, gains[0] + gains[1]

    return max(frontier_vals, key=score)


def frontier_optimal_welfare(frontier: Iterable[Iterable[float]]) -> float:
    frontier_vals = [tuple(float(y) for y in x) for x in frontier]
    if not frontier_vals:
        return float("nan")
    return max(social_welfare(point) for point in frontier_vals)


def money_left_on_table(welfare: float, optimal_welfare: float) -> float:
    if np.isnan(welfare) or np.isnan(optimal_welfare):
        return float("nan")
    return max(0.0, float(optimal_welfare) - float(welfare))


def welfare_regret(welfare: float, optimal_welfare: float) -> float:
    if np.isnan(welfare) or np.isnan(optimal_welfare) or optimal_welfare <= 0.0:
        return float("nan")
    return money_left_on_table(welfare, optimal_welfare) / float(optimal_welfare)


def bootstrap_ci_mean(
    values: Iterable[float], n_resamples: int = 2000, ci_level: float = 0.95, rng: np.random.Generator | None = None
) -> tuple[float, float]:
    arr = np.asarray(tuple(float(x) for x in values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _bootstrap_ci_mean_difference(
    x: np.ndarray, y: np.ndarray, n_resamples: int, ci_level: float, rng: np.random.Generator
) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), float("nan")
    idx_x = rng.integers(0, x.size, size=(n_resamples, x.size))
    idx_y = rng.integers(0, y.size, size=(n_resamples, y.size))
    diffs = x[idx_x].mean(axis=1) - y[idx_y].mean(axis=1)
    alpha = (1.0 - ci_level) / 2.0
    low, high = np.quantile(diffs, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _permutation_test_pvalue(
    x: np.ndarray, y: np.ndarray, n_permutations: int, rng: np.random.Generator
) -> float:
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    observed = abs(float(x.mean() - y.mean()))
    pooled = np.concatenate([x, y])
    n_x = x.size
    extreme = 0
    for _ in range(n_permutations):
        permuted = rng.permutation(pooled)
        diff = abs(float(permuted[:n_x].mean() - permuted[n_x:].mean()))
        if diff >= observed - 1e-12:
            extreme += 1
    return float((extreme + 1) / (n_permutations + 1))


def cohen_d(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(tuple(float(v) for v in x), dtype=float)
    b = np.asarray(tuple(float(v) for v in y), dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    mean_diff = float(a.mean() - b.mean())
    if a.size < 2 or b.size < 2:
        pooled_std_small = float(np.std(np.concatenate([a, b]), ddof=0))
        if pooled_std_small <= 0.0:
            return 0.0
        return mean_diff / pooled_std_small
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled_num = (a.size - 1) * var_a + (b.size - 1) * var_b
    pooled_den = a.size + b.size - 2
    if pooled_den <= 0:
        return float("nan")
    pooled_std = math.sqrt(pooled_num / pooled_den)
    if pooled_std <= 0.0:
        return 0.0
    return mean_diff / pooled_std


def _fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    if not np.any(valid):
        return q
    p_valid = p[valid]
    order = np.argsort(p_valid)
    sorted_p = p_valid[order]
    m = len(sorted_p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_min = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        corrected = sorted_p[i] * m / rank
        running_min = min(running_min, corrected)
        adjusted_sorted[i] = running_min
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    q[valid] = adjusted
    return q


def ci95(std: float, count: int) -> float:
    if count <= 1 or np.isnan(std):
        return 0.0
    return 1.96 * std / math.sqrt(count)


def summarize_results(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    bootstrap_n_resamples: int = 0,
    bootstrap_ci_level: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    grouped = df.groupby(group_cols, dropna=False)[metric_cols]
    summary = grouped.agg(["mean", "std", "count"]).reset_index()

    flat_cols: list[str] = []
    for col in summary.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(str(col))
    summary.columns = flat_cols

    for metric in metric_cols:
        summary[f"{metric}_ci95"] = summary.apply(
            lambda row: ci95(float(row[f"{metric}_std"]), int(row[f"{metric}_count"])), axis=1
        )

    if bootstrap_n_resamples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        bootstrap_rows: list[dict] = []
        for key, group in df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key))
            for metric in metric_cols:
                low, high = bootstrap_ci_mean(
                    group[metric].to_numpy(dtype=float, copy=False),
                    n_resamples=bootstrap_n_resamples,
                    ci_level=bootstrap_ci_level,
                    rng=rng,
                )
                row[f"{metric}_boot_ci_low"] = low
                row[f"{metric}_boot_ci_high"] = high
            bootstrap_rows.append(row)
        bootstrap_df = pd.DataFrame(bootstrap_rows)
        summary = summary.merge(bootstrap_df, on=group_cols, how="left")
    return summary


def pairwise_significance_tests(
    df: pd.DataFrame,
    *,
    context_cols: list[str],
    strategy_col: str,
    metric_cols: list[str],
    n_permutations: int = 2000,
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    random_seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict] = []

    for context_key, context_df in df.groupby(context_cols, dropna=False):
        if not isinstance(context_key, tuple):
            context_key = (context_key,)
        context_map = dict(zip(context_cols, context_key))

        for metric in metric_cols:
            metric_df = context_df[[strategy_col, metric]].dropna()
            if metric_df.empty:
                continue
            grouped = {
                str(strategy): grp[metric].to_numpy(dtype=float, copy=False)
                for strategy, grp in metric_df.groupby(strategy_col, dropna=False)
                if len(grp) > 0
            }
            strategies = sorted(grouped.keys())
            for strategy_a, strategy_b in combinations(strategies, 2):
                x = grouped[strategy_a]
                y = grouped[strategy_b]
                mean_a = float(np.mean(x)) if x.size else float("nan")
                mean_b = float(np.mean(y)) if y.size else float("nan")
                mean_diff = mean_a - mean_b if not np.isnan(mean_a) and not np.isnan(mean_b) else float("nan")
                ci_low, ci_high = _bootstrap_ci_mean_difference(
                    x,
                    y,
                    n_resamples=n_bootstrap,
                    ci_level=ci_level,
                    rng=rng,
                )
                rows.append(
                    {
                        **context_map,
                        "metric": metric,
                        "strategy_a": strategy_a,
                        "strategy_b": strategy_b,
                        "n_a": int(x.size),
                        "n_b": int(y.size),
                        "mean_a": mean_a,
                        "mean_b": mean_b,
                        "mean_diff": mean_diff,
                        "mean_diff_boot_ci_low": ci_low,
                        "mean_diff_boot_ci_high": ci_high,
                        "p_value": _permutation_test_pvalue(x, y, n_permutations=n_permutations, rng=rng),
                        "effect_size_cohen_d": cohen_d(x, y),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["p_value_fdr"] = np.nan
    for _, index in out.groupby(context_cols + ["metric"], dropna=False).groups.items():
        idx = list(index)
        out.loc[idx, "p_value_fdr"] = _fdr_bh(out.loc[idx, "p_value"].to_numpy(dtype=float))
    return out

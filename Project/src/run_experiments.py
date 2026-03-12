from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.config import ExperimentConfig
from src.domain import make_mechanism, make_ufuns
from src.metrics import (
    distance_to_frontier,
    distance_to_point,
    frontier_optimal_welfare,
    ks_reference_from_frontier,
    money_left_on_table,
    pairwise_significance_tests,
    social_welfare,
    summarize_results,
    welfare_regret,
)
from src.negotiators import strategy_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-domain NegMAS experiments.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/main"))
    parser.add_argument("--n-seeds", type=int, default=40)
    parser.add_argument("--profiles", nargs="+", default=["A", "B", "C", "D", "E", "F"])
    parser.add_argument("--n-steps", nargs="+", type=int, default=[12, 24, 36])
    parser.add_argument("--strategies", nargs="+", default=None, help="Subset of strategy names from strategy_registry()")
    parser.add_argument("--include-self-play", action="store_true", default=True)
    parser.add_argument("--exclude-self-play", action="store_false", dest="include_self_play")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--buyer-reserved-value", type=float, default=0.35)
    parser.add_argument("--seller-reserved-value", type=float, default=0.35)
    parser.add_argument("--pend", type=float, default=0.04)
    parser.add_argument("--pend-per-second", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--permutation-resamples", type=int, default=2000)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _agreement_to_text(agreement) -> str:
    if agreement is None:
        return ""
    return json.dumps(list(agreement))


def _extract_utility_pair(value) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first, second = value[0], value[1]
        if isinstance(first, (int, float, np.floating)) and isinstance(second, (int, float, np.floating)):
            return float(first), float(second)
    return None


def _extract_reference_utils(points) -> tuple[float, float] | None:
    if points is None:
        return None
    if isinstance(points, np.ndarray):
        points = points.tolist()
    if isinstance(points, (list, tuple)):
        if not points:
            return None
        direct = _extract_utility_pair(points)
        if direct is not None:
            return direct
        first = points[0]
        if isinstance(first, np.ndarray):
            first = first.tolist()
        if isinstance(first, (list, tuple)):
            nested = _extract_utility_pair(first)
            if nested is not None:
                return nested
            if first:
                return _extract_utility_pair(first[0])
    return None


def _get_reference_utils(mechanism, method_names: tuple[str, ...]) -> tuple[float, float] | None:
    for method_name in method_names:
        method = getattr(mechanism, method_name, None)
        if callable(method):
            try:
                return _extract_reference_utils(method())
            except Exception:
                continue
    return None


def _safe_reserved_value(ufun, fallback: float) -> float:
    try:
        value = float(ufun.reserved_value)
        if np.isnan(value):
            return float(fallback)
        return value
    except Exception:
        return float(fallback)


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _make_strategy_role_view(raw_df: pd.DataFrame) -> pd.DataFrame:
    buyer_view = raw_df[
        [
            "run_id",
            "seed",
            "profile_id",
            "n_steps",
            "strategy_buyer",
            "strategy_seller",
            "agreement",
            "rounds",
            "u_buyer",
            "welfare",
            "pareto_distance",
            "nash_distance",
            "kalai_distance",
            "ks_distance",
            "money_left_on_table",
            "welfare_regret",
            "nash_gap",
        ]
    ].rename(
        columns={
            "strategy_buyer": "strategy",
            "strategy_seller": "opponent_strategy",
            "u_buyer": "own_utility",
        }
    )
    buyer_view["role"] = "buyer"

    seller_view = raw_df[
        [
            "run_id",
            "seed",
            "profile_id",
            "n_steps",
            "strategy_buyer",
            "strategy_seller",
            "agreement",
            "rounds",
            "u_seller",
            "welfare",
            "pareto_distance",
            "nash_distance",
            "kalai_distance",
            "ks_distance",
            "money_left_on_table",
            "welfare_regret",
            "nash_gap",
        ]
    ].rename(
        columns={
            "strategy_seller": "strategy",
            "strategy_buyer": "opponent_strategy",
            "u_seller": "own_utility",
        }
    )
    seller_view["role"] = "seller"

    role_df = pd.concat([buyer_view, seller_view], ignore_index=True)
    ordered_cols = [
        "run_id",
        "seed",
        "profile_id",
        "n_steps",
        "role",
        "strategy",
        "opponent_strategy",
        "agreement",
        "rounds",
        "own_utility",
        "welfare",
        "pareto_distance",
        "nash_distance",
        "kalai_distance",
        "ks_distance",
        "money_left_on_table",
        "welfare_regret",
        "nash_gap",
    ]
    return role_df[ordered_cols]


def _run_single(
    *,
    run_id: int,
    seed: int,
    profile_id: str,
    n_steps: int,
    buyer_strategy: str,
    seller_strategy: str,
    time_limit: float | None,
    buyer_reserved_value: float,
    seller_reserved_value: float,
    pend: float,
    pend_per_second: float,
    registry: dict[str, type],
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    mechanism = make_mechanism(
        n_steps=n_steps,
        time_limit=time_limit,
        pend=pend,
        pend_per_second=pend_per_second,
    )
    buyer_ufun, seller_ufun = make_ufuns(
        profile_id=profile_id,
        outcome_space=mechanism.outcome_space,
        buyer_reserved_value=buyer_reserved_value,
        seller_reserved_value=seller_reserved_value,
    )
    buyer = registry[buyer_strategy](name=f"buyer_{buyer_strategy}")
    seller = registry[seller_strategy](name=f"seller_{seller_strategy}")
    mechanism.add(buyer, ufun=buyer_ufun)
    mechanism.add(seller, ufun=seller_ufun)
    state = mechanism.run()

    frontier_utils, _ = mechanism.pareto_frontier()
    disagreement_point = (
        _safe_reserved_value(buyer_ufun, buyer_reserved_value),
        _safe_reserved_value(seller_ufun, seller_reserved_value),
    )
    nash_utils = _get_reference_utils(mechanism, ("nash_points",))
    kalai_utils = _get_reference_utils(mechanism, ("kalai_points",))
    ks_utils = _get_reference_utils(mechanism, ("ks_points", "modified_ks_points"))
    if ks_utils is None:
        ks_utils = ks_reference_from_frontier(frontier_utils, disagreement_point)
    frontier_best_welfare = frontier_optimal_welfare(frontier_utils)
    nash_welfare = social_welfare(nash_utils) if nash_utils is not None else float("nan")

    agreement = state.agreement
    agreement_exists = agreement is not None
    rounds = int(state.step)

    if agreement_exists:
        utility_point = (float(buyer_ufun(agreement)), float(seller_ufun(agreement)))
    else:
        utility_point = disagreement_point

    u_buyer, u_seller = utility_point
    welfare = social_welfare(utility_point)
    pareto_distance = distance_to_frontier(utility_point, frontier_utils)
    nash_distance = distance_to_point(utility_point, nash_utils)
    kalai_distance = distance_to_point(utility_point, kalai_utils)
    ks_distance = distance_to_point(utility_point, ks_utils)
    nash_gap = nash_welfare - welfare if not np.isnan(nash_welfare) else float("nan")
    lost = money_left_on_table(welfare, frontier_best_welfare)
    regret = welfare_regret(welfare, frontier_best_welfare)

    return {
        "run_id": run_id,
        "seed": seed,
        "profile_id": profile_id,
        "n_steps": n_steps,
        "strategy_buyer": buyer_strategy,
        "strategy_seller": seller_strategy,
        "agreement": int(agreement_exists),
        "agreement_outcome": _agreement_to_text(agreement),
        "rounds": rounds,
        "u_buyer": u_buyer,
        "u_seller": u_seller,
        "welfare": welfare,
        "frontier_optimal_welfare": frontier_best_welfare,
        "pareto_distance": pareto_distance,
        "nash_distance": nash_distance,
        "kalai_distance": kalai_distance,
        "ks_distance": ks_distance,
        "money_left_on_table": lost,
        "welfare_regret": regret,
        "nash_gap": nash_gap,
    }


def run_experiments(config: ExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    registry = strategy_registry()
    unknown = sorted(set(config.strategies).difference(registry.keys()))
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {sorted(registry.keys())}")
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = list(config.strategies)
    run_rows: list[dict] = []
    run_id = 0
    progress_every = max(0, int(config.progress_every))
    n_pairs = len(strategies) * len(strategies)
    if not config.include_self_play:
        n_pairs -= len(strategies)
    total_runs = len(config.profiles) * len(config.n_steps_list) * n_pairs * int(config.n_seeds)
    tic = perf_counter()
    print(f"[progress] Starting simulations: total_runs={total_runs}", flush=True)

    for profile_id in config.profiles:
        for n_steps in config.n_steps_list:
            for buyer_strategy in strategies:
                for seller_strategy in strategies:
                    if not config.include_self_play and buyer_strategy == seller_strategy:
                        continue
                    for seed in range(config.n_seeds):
                        run_rows.append(
                            _run_single(
                                run_id=run_id,
                                seed=seed,
                                profile_id=profile_id,
                                n_steps=n_steps,
                                buyer_strategy=buyer_strategy,
                                seller_strategy=seller_strategy,
                                time_limit=config.time_limit,
                                buyer_reserved_value=config.buyer_reserved_value,
                                seller_reserved_value=config.seller_reserved_value,
                                pend=config.pend,
                                pend_per_second=config.pend_per_second,
                                registry=registry,
                            )
                        )
                        run_id += 1
                        if progress_every > 0 and (run_id % progress_every == 0 or run_id == total_runs):
                            elapsed = perf_counter() - tic
                            rate = run_id / elapsed if elapsed > 0 else float("inf")
                            remaining = max(total_runs - run_id, 0)
                            eta = remaining / rate if rate > 0 else float("inf")
                            pct = 100.0 * run_id / total_runs if total_runs > 0 else 100.0
                            print(
                                f"[progress] runs {run_id}/{total_runs} ({pct:5.1f}%) "
                                f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta)}",
                                flush=True,
                            )

    raw_df = pd.DataFrame(run_rows)
    metric_cols = [
        "agreement",
        "rounds",
        "u_buyer",
        "u_seller",
        "welfare",
        "frontier_optimal_welfare",
        "pareto_distance",
        "nash_distance",
        "kalai_distance",
        "ks_distance",
        "money_left_on_table",
        "welfare_regret",
        "nash_gap",
    ]
    group_cols = ["profile_id", "n_steps", "strategy_buyer", "strategy_seller"]
    print("[progress] Building main summary (with bootstrap CIs)...", flush=True)
    summary_df = summarize_results(
        raw_df,
        group_cols=group_cols,
        metric_cols=metric_cols,
        bootstrap_n_resamples=config.bootstrap_resamples,
        bootstrap_ci_level=0.95,
        bootstrap_seed=0,
    )

    print("[progress] Building strategy-by-role summary...", flush=True)
    role_df = _make_strategy_role_view(raw_df)
    strategy_metric_cols = [
        "agreement",
        "rounds",
        "own_utility",
        "welfare",
        "pareto_distance",
        "nash_distance",
        "kalai_distance",
        "ks_distance",
        "money_left_on_table",
        "welfare_regret",
        "nash_gap",
    ]
    strategy_summary_df = summarize_results(
        role_df,
        group_cols=["profile_id", "n_steps", "role", "strategy"],
        metric_cols=strategy_metric_cols,
        bootstrap_n_resamples=config.bootstrap_resamples,
        bootstrap_ci_level=0.95,
        bootstrap_seed=0,
    )
    print("[progress] Running pairwise significance tests...", flush=True)
    pairwise_df = pairwise_significance_tests(
        role_df,
        context_cols=["profile_id", "n_steps", "role"],
        strategy_col="strategy",
        metric_cols=strategy_metric_cols,
        n_permutations=config.permutation_resamples,
        n_bootstrap=config.bootstrap_resamples,
        ci_level=0.95,
        random_seed=0,
    )

    raw_path = output_dir / "main_runs.csv"
    summary_path = output_dir / "main_summary.csv"
    strategy_summary_path = output_dir / "main_strategy_summary.csv"
    pairwise_path = output_dir / "main_pairwise_tests.csv"
    config_path = output_dir / "main_config.json"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    strategy_summary_df.to_csv(strategy_summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"[progress] Complete in {_format_duration(perf_counter() - tic)}", flush=True)
    return raw_df, summary_df


def main() -> None:
    args = _parse_args()
    config = ExperimentConfig(
        n_steps_list=tuple(args.n_steps),
        profiles=tuple(args.profiles),
        strategies=tuple(args.strategies) if args.strategies else ExperimentConfig.strategies,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
        include_self_play=args.include_self_play,
        time_limit=args.time_limit,
        buyer_reserved_value=args.buyer_reserved_value,
        seller_reserved_value=args.seller_reserved_value,
        pend=args.pend,
        pend_per_second=args.pend_per_second,
        bootstrap_resamples=args.bootstrap_resamples,
        permutation_resamples=args.permutation_resamples,
        progress_every=args.progress_every,
    )
    raw_df, summary_df = run_experiments(config)
    print(f"Wrote {len(raw_df)} runs to {config.output_dir / 'main_runs.csv'}")
    print(f"Wrote summary to {config.output_dir / 'main_summary.csv'} ({len(summary_df)} rows)")
    print(f"Wrote strategy summary to {config.output_dir / 'main_strategy_summary.csv'}")
    print(f"Wrote pairwise tests to {config.output_dir / 'main_pairwise_tests.csv'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.domain_multilateral import VALID_MULTI_PROFILES, make_multilateral_mechanism, make_procurement_ufuns
from src.metrics import (
    distance_to_frontier,
    frontier_optimal_welfare,
    money_left_on_table,
    social_welfare,
    summarize_results,
    welfare_regret,
)
from src.negotiators import strategy_registry


@dataclass(frozen=True)
class MultiExperimentConfig:
    n_steps_list: tuple[int, ...] = (20, 40)
    profiles: tuple[str, ...] = tuple(sorted(VALID_MULTI_PROFILES))
    n_seeds: int = 20
    n_suppliers: int = 2
    strategies: tuple[str, ...] = (
        "BoulwareTBNegotiator",
        "LinearTBNegotiator",
        "ConcederTBNegotiator",
        "SmartAspirationNegotiator",
        "GeniusBOABaselineNegotiator",
    )
    include_self_play: bool = True
    output_dir: Path = Path("results/multilateral")
    time_limit: float | None = None
    buyer_reserved_value: float = 0.35
    supplier_reserved_value: float = 0.35
    pend: float = 0.04
    pend_per_second: float = 0.02
    bootstrap_resamples: int = 1000
    progress_every: int = 100

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-to-many procurement SAO experiments (3+ agents).")
    parser.add_argument("--output-dir", type=Path, default=Path("results/multilateral"))
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--profiles", nargs="+", default=sorted(VALID_MULTI_PROFILES))
    parser.add_argument("--n-steps", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--n-suppliers", type=int, default=2)
    parser.add_argument("--strategies", nargs="+", default=None, help="Subset of strategy names from strategy_registry()")
    parser.add_argument("--include-self-play", action="store_true", default=True)
    parser.add_argument("--exclude-self-play", action="store_false", dest="include_self_play")
    parser.add_argument("--time-limit", type=float, default=None)
    parser.add_argument("--buyer-reserved-value", type=float, default=0.35)
    parser.add_argument("--supplier-reserved-value", type=float, default=0.35)
    parser.add_argument("--pend", type=float, default=0.04)
    parser.add_argument("--pend-per-second", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "n/a"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _safe_reserved_value(ufun, fallback: float) -> float:
    try:
        value = float(ufun.reserved_value)
        if np.isnan(value):
            return float(fallback)
        return value
    except Exception:
        return float(fallback)


def _strategy_columns(n_suppliers: int) -> list[str]:
    return ["strategy_buyer"] + [f"strategy_supplier_{i + 1}" for i in range(n_suppliers)]


def _run_single(
    *,
    run_id: int,
    seed: int,
    profile_id: str,
    n_steps: int,
    strategy_vector: tuple[str, ...],
    n_suppliers: int,
    time_limit: float | None,
    buyer_reserved_value: float,
    supplier_reserved_value: float,
    pend: float,
    pend_per_second: float,
    registry: dict[str, type],
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    mechanism = make_multilateral_mechanism(
        n_steps=n_steps,
        time_limit=time_limit,
        pend=pend,
        pend_per_second=pend_per_second,
    )
    buyer_ufun, supplier_ufuns = make_procurement_ufuns(
        profile_id=profile_id,
        outcome_space=mechanism.outcome_space,
        n_suppliers=n_suppliers,
        buyer_reserved_value=buyer_reserved_value,
        supplier_reserved_value=supplier_reserved_value,
    )

    buyer_strategy = strategy_vector[0]
    supplier_strategies = strategy_vector[1:]
    mechanism.add(registry[buyer_strategy](name=f"buyer_{buyer_strategy}"), ufun=buyer_ufun)
    for i, (strategy_name, supplier_ufun) in enumerate(zip(supplier_strategies, supplier_ufuns), start=1):
        mechanism.add(registry[strategy_name](name=f"supplier{i}_{strategy_name}"), ufun=supplier_ufun)

    state = mechanism.run()
    frontier_utils, _ = mechanism.pareto_frontier()
    agreement = state.agreement
    agreement_exists = agreement is not None
    rounds = int(state.step)

    if agreement_exists:
        buyer_u = float(buyer_ufun(agreement))
        supplier_utils = [float(ufun(agreement)) for ufun in supplier_ufuns]
    else:
        buyer_u = _safe_reserved_value(buyer_ufun, buyer_reserved_value)
        supplier_utils = [
            _safe_reserved_value(ufun, supplier_reserved_value + 0.03 * i) for i, ufun in enumerate(supplier_ufuns)
        ]

    utilities = [buyer_u] + supplier_utils
    utility_point = tuple(utilities)
    welfare = social_welfare(utility_point)
    frontier_best_welfare = frontier_optimal_welfare(frontier_utils)

    nash_points = getattr(mechanism, "nash_points", lambda: [])()
    if nash_points:
        first = nash_points[0]
        nash_utils = first[0] if isinstance(first, (list, tuple)) and len(first) > 0 else first
        nash_welfare = float(sum(float(x) for x in nash_utils))
    else:
        nash_welfare = float("nan")

    row = {
        "run_id": run_id,
        "seed": seed,
        "profile_id": profile_id,
        "n_steps": n_steps,
        "n_agents": n_suppliers + 1,
        "agreement": int(agreement_exists),
        "agreement_outcome": json.dumps(list(agreement)) if agreement_exists else "",
        "rounds": rounds,
        "u_buyer": buyer_u,
        "u_supplier_avg": float(np.mean(supplier_utils)),
        "u_supplier_min": float(np.min(supplier_utils)),
        "u_supplier_max": float(np.max(supplier_utils)),
        "welfare": welfare,
        "min_utility": float(np.min(utilities)),
        "utility_range": float(np.max(utilities) - np.min(utilities)),
        "pareto_distance": distance_to_frontier(utility_point, frontier_utils),
        "money_left_on_table": money_left_on_table(welfare, frontier_best_welfare),
        "welfare_regret": welfare_regret(welfare, frontier_best_welfare),
        "nash_gap": nash_welfare - welfare if not np.isnan(nash_welfare) else float("nan"),
    }
    row["strategy_buyer"] = buyer_strategy
    for i, strategy_name in enumerate(supplier_strategies, start=1):
        row[f"strategy_supplier_{i}"] = strategy_name
        row[f"u_supplier_{i}"] = supplier_utils[i - 1]
    return row


def run_experiments(config: MultiExperimentConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config.n_suppliers < 2:
        raise ValueError(f"n_suppliers must be >= 2, got {config.n_suppliers}")

    registry = strategy_registry()
    unknown = sorted(set(config.strategies).difference(registry.keys()))
    if unknown:
        raise ValueError(f"Unknown strategies: {unknown}. Available: {sorted(registry.keys())}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    strategies = list(config.strategies)
    strategy_vectors = list(itertools.product(strategies, repeat=config.n_suppliers + 1))
    if not config.include_self_play:
        strategy_vectors = [vector for vector in strategy_vectors if len(set(vector)) > 1]

    total_runs = len(config.profiles) * len(config.n_steps_list) * len(strategy_vectors) * int(config.n_seeds)
    progress_every = max(0, int(config.progress_every))
    tic = perf_counter()
    print(f"[progress] Starting multilateral simulations: total_runs={total_runs}", flush=True)

    run_rows: list[dict] = []
    run_id = 0
    for profile_id in config.profiles:
        for n_steps in config.n_steps_list:
            for strategy_vector in strategy_vectors:
                for seed in range(config.n_seeds):
                    run_rows.append(
                        _run_single(
                            run_id=run_id,
                            seed=seed,
                            profile_id=profile_id,
                            n_steps=n_steps,
                            strategy_vector=strategy_vector,
                            n_suppliers=config.n_suppliers,
                            time_limit=config.time_limit,
                            buyer_reserved_value=config.buyer_reserved_value,
                            supplier_reserved_value=config.supplier_reserved_value,
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
        "u_supplier_avg",
        "u_supplier_min",
        "u_supplier_max",
        "welfare",
        "min_utility",
        "utility_range",
        "pareto_distance",
        "money_left_on_table",
        "welfare_regret",
        "nash_gap",
    ]
    group_cols = ["profile_id", "n_steps"] + _strategy_columns(config.n_suppliers)

    print("[progress] Building multilateral summary...", flush=True)
    summary_df = summarize_results(
        raw_df,
        group_cols=group_cols,
        metric_cols=metric_cols,
        bootstrap_n_resamples=config.bootstrap_resamples,
        bootstrap_ci_level=0.95,
        bootstrap_seed=0,
    )

    raw_path = output_dir / "multi_runs.csv"
    summary_path = output_dir / "multi_summary.csv"
    config_path = output_dir / "multi_config.json"
    raw_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    config_path.write_text(json.dumps(config.to_dict(), indent=2))
    print(f"[progress] Complete in {_format_duration(perf_counter() - tic)}", flush=True)
    return raw_df, summary_df


def main() -> None:
    args = _parse_args()
    config = MultiExperimentConfig(
        n_steps_list=tuple(args.n_steps),
        profiles=tuple(args.profiles),
        n_seeds=args.n_seeds,
        n_suppliers=args.n_suppliers,
        strategies=tuple(args.strategies) if args.strategies else MultiExperimentConfig.strategies,
        include_self_play=args.include_self_play,
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        buyer_reserved_value=args.buyer_reserved_value,
        supplier_reserved_value=args.supplier_reserved_value,
        pend=args.pend,
        pend_per_second=args.pend_per_second,
        bootstrap_resamples=args.bootstrap_resamples,
        progress_every=args.progress_every,
    )
    raw_df, summary_df = run_experiments(config)
    print(f"Wrote {len(raw_df)} runs to {config.output_dir / 'multi_runs.csv'}")
    print(f"Wrote summary to {config.output_dir / 'multi_summary.csv'} ({len(summary_df)} rows)")


if __name__ == "__main__":
    main()

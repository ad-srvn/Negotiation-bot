from __future__ import annotations

import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from negmas.helpers import humanize_time
from negmas.inout import Scenario
from negmas.outcomes.outcome_space import make_os
from negmas.tournaments.neg import cartesian_tournament

from src.domain import VALID_PROFILES, make_issues, make_ufuns
from src.negotiators import strategy_registry


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run appendix random-scenario tournament.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/appendix"))
    parser.add_argument("--n-scenarios", type=int, default=6)
    parser.add_argument("--n-repetitions", type=int, default=2)
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def get_scenarios(n: int = 6, seed: int = 2026) -> list[Scenario]:
    """
    Randomized robustness scenarios over rich asymmetric profile families.

    Families are sampled from the fixed-domain set (A..F), which includes
    nonlinear value functions and a categorical issue. Each scenario randomizes
    reservation values to create additional robustness variation.
    """
    rng = np.random.default_rng(seed)
    family_ids = sorted(VALID_PROFILES)
    scenarios: list[Scenario] = []
    for i in range(n):
        family_id = str(rng.choice(family_ids))
        issues = tuple(make_issues())
        outcome_space = make_os(issues, name=f"S{i}_{family_id}")
        buyer_rv = float(rng.uniform(0.2, 0.6))
        seller_rv = float(rng.uniform(0.2, 0.6))
        ufuns = make_ufuns(
            profile_id=family_id,
            outcome_space=outcome_space,
            buyer_reserved_value=buyer_rv,
            seller_reserved_value=seller_rv,
        )
        scenarios.append(Scenario(outcome_space=outcome_space, ufuns=ufuns))
    return scenarios


def run_appendix_tournament(
    output_dir: Path,
    n_scenarios: int = 6,
    n_repetitions: int = 2,
    n_steps: int = 50,
    seed: int = 2026,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    registry = strategy_registry()
    competitors = [registry[name] for name in registry]
    scenarios = get_scenarios(n_scenarios, seed=seed)

    tic = time.perf_counter()
    results = cartesian_tournament(
        competitors=competitors,
        scenarios=scenarios,
        n_steps=n_steps,
        n_repetitions=n_repetitions,
        path=output_dir / "tournament_artifacts",
        path_exists="overwrite",
        njobs=-1,
        save_scenario_figs=False,
        memory_optimization="none",
        storage_optimization="none",
    )
    elapsed = humanize_time(time.perf_counter() - tic, show_ms=True)
    print(f"Tournament completed in {elapsed}")

    config = {
        "n_scenarios": n_scenarios,
        "n_repetitions": n_repetitions,
        "n_steps": n_steps,
        "seed": seed,
        "competitors": list(registry.keys()),
        "profile_families": sorted(VALID_PROFILES),
    }
    (output_dir / "appendix_config.json").write_text(json.dumps(config, indent=2))

    with (output_dir / "appendix_results.pkl").open("wb") as f:
        pickle.dump(results, f)

    summary_df = None
    if hasattr(results, "scores_summary") and isinstance(results.scores_summary, pd.DataFrame):
        summary_df = results.scores_summary.reset_index()
    elif hasattr(results, "scores") and isinstance(results.scores, pd.DataFrame):
        summary_df = results.scores.reset_index(drop=True)

    if summary_df is not None and len(summary_df.columns) > 0:
        summary_df.to_csv(output_dir / "appendix_scores.csv", index=False)
        print(f"Wrote appendix score summary to {output_dir / 'appendix_scores.csv'}")
    else:
        fallback = pd.DataFrame(
            [
                {
                    "note": "No tabular score summary available from tournament object",
                    "n_scenarios": n_scenarios,
                    "n_repetitions": n_repetitions,
                    "n_steps": n_steps,
                }
            ]
        )
        fallback.to_csv(output_dir / "appendix_scores.csv", index=False)
        print("No tabular score summary returned; wrote fallback appendix_scores.csv with run metadata.")


def main() -> None:
    args = _parse_args()
    run_appendix_tournament(
        output_dir=args.output_dir,
        n_scenarios=args.n_scenarios,
        n_repetitions=args.n_repetitions,
        n_steps=args.n_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

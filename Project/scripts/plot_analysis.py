from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate notebook analysis plots and add plots for newer project metrics."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/main_final"),
        help="Directory containing main_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for figures (default: <results-dir>/figures_analysis)",
    )
    parser.add_argument(
        "--multilateral-runs",
        type=Path,
        default=None,
        help="Optional path to multilateral runs CSV (e.g., results/multilateral/multi_runs.csv)",
    )
    parser.add_argument("--trajectory-profile", type=str, default="A")
    parser.add_argument("--trajectory-buyer", type=str, default="SmartAspirationNegotiator")
    parser.add_argument("--trajectory-seller", type=str, default="BoulwareTBNegotiator")
    parser.add_argument("--trajectory-steps", type=int, default=50)
    return parser.parse_args()


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _catplot_save(g, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    g.fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(g.fig)
    return path


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _pair_col(df: pd.DataFrame) -> pd.Series:
    return df["strategy_buyer"].astype(str) + " vs " + df["strategy_seller"].astype(str)


def plot_agreement_rate(runs: pd.DataFrame, output_dir: Path) -> Path:
    _require_columns(
        runs,
        {"profile_id", "n_steps", "strategy_buyer", "strategy_seller", "agreement"},
        "main_runs.csv",
    )
    agg = runs.groupby(["profile_id", "n_steps", "strategy_buyer", "strategy_seller"], as_index=False)[
        "agreement"
    ].mean()
    agg["pair"] = _pair_col(agg)

    g = sns.catplot(
        data=agg,
        x="pair",
        y="agreement",
        col="profile_id",
        hue="n_steps",
        kind="bar",
        col_wrap=3,
        height=3.6,
        aspect=1.3,
        errorbar=None,
    )
    g.set_axis_labels("Strategy pairing", "Agreement rate")
    g.set_titles("profile={col_name}")
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=90)
    g.fig.suptitle("Agreement rate by strategy pairing and deadline", y=1.02)
    return _catplot_save(g, output_dir / "analysis_agreement_rate_by_pair.png")


def plot_rounds_to_agreement(runs: pd.DataFrame, output_dir: Path) -> Path | None:
    _require_columns(
        runs,
        {"strategy_buyer", "strategy_seller", "profile_id", "agreement", "rounds"},
        "main_runs.csv",
    )
    rounds_df = runs[runs["agreement"] == 1].copy()
    if rounds_df.empty:
        return None
    rounds_df["pair"] = _pair_col(rounds_df)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.boxplot(data=rounds_df, x="pair", y="rounds", hue="profile_id", ax=ax)
    ax.set_title("Rounds to agreement (agreements only)")
    ax.set_xlabel("Strategy pairing")
    ax.set_ylabel("Rounds")
    ax.tick_params(axis="x", rotation=90)
    return _save_fig(fig, output_dir / "analysis_rounds_to_agreement.png")


def plot_effective_rounds(runs: pd.DataFrame, output_dir: Path) -> Path:
    _require_columns(
        runs,
        {"strategy_buyer", "strategy_seller", "profile_id", "agreement", "rounds", "n_steps"},
        "main_runs.csv",
    )
    df = runs.copy()
    # Count failed negotiations as full deadline consumption to avoid optimistic bias.
    df["effective_rounds"] = df["rounds"].where(df["agreement"] == 1, df["n_steps"])
    df["pair"] = _pair_col(df)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=df, x="pair", y="effective_rounds", hue="profile_id", errorbar=("ci", 95), ax=ax)
    ax.set_title("Effective rounds by pairing (failures counted as deadline)")
    ax.set_xlabel("Strategy pairing")
    ax.set_ylabel("Effective rounds")
    ax.tick_params(axis="x", rotation=90)
    return _save_fig(fig, output_dir / "analysis_effective_rounds_including_failures.png")


def plot_welfare_distribution(runs: pd.DataFrame, output_dir: Path) -> Path:
    _require_columns(runs, {"strategy_buyer", "strategy_seller", "profile_id", "welfare"}, "main_runs.csv")
    welfare_df = runs.dropna(subset=["welfare"]).copy()
    welfare_df["pair"] = _pair_col(welfare_df)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.violinplot(data=welfare_df, x="pair", y="welfare", hue="profile_id", cut=0, ax=ax)
    ax.set_title("Joint welfare distribution by pairing")
    ax.set_xlabel("Strategy pairing")
    ax.set_ylabel("Welfare")
    ax.tick_params(axis="x", rotation=90)
    return _save_fig(fig, output_dir / "analysis_welfare_distribution.png")


def plot_pareto_distance(runs: pd.DataFrame, output_dir: Path) -> Path:
    _require_columns(
        runs,
        {"strategy_buyer", "strategy_seller", "profile_id", "pareto_distance"},
        "main_runs.csv",
    )
    pareto_df = runs.dropna(subset=["pareto_distance"]).copy()
    pareto_df["pair"] = _pair_col(pareto_df)

    fig, ax = plt.subplots(figsize=(16, 7))
    sns.barplot(data=pareto_df, x="pair", y="pareto_distance", hue="profile_id", errorbar=("ci", 95), ax=ax)
    ax.set_title("Distance to Pareto frontier (lower is better)")
    ax.set_xlabel("Strategy pairing")
    ax.set_ylabel("Pareto distance")
    ax.tick_params(axis="x", rotation=90)
    return _save_fig(fig, output_dir / "analysis_pareto_distance.png")


def plot_pairing_heatmaps(runs: pd.DataFrame, output_dir: Path) -> list[Path]:
    needed = {
        "strategy_buyer",
        "strategy_seller",
        "agreement",
        "welfare",
        "pareto_distance",
        "rounds",
        "n_steps",
    }
    _require_columns(runs, needed, "main_runs.csv")

    df = runs.copy()
    df["effective_rounds"] = df["rounds"].where(df["agreement"] == 1, df["n_steps"])
    metrics = [
        ("agreement", "Agreement rate"),
        ("welfare", "Mean welfare"),
        ("pareto_distance", "Mean Pareto distance (lower better)"),
        ("effective_rounds", "Mean effective rounds"),
    ]

    created: list[Path] = []
    for metric, title in metrics:
        pivot = (
            df.groupby(["strategy_buyer", "strategy_seller"], as_index=False)[metric]
            .mean()
            .pivot(index="strategy_buyer", columns="strategy_seller", values=metric)
        )
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": metric},
            ax=ax,
        )
        ax.set_title(f"{title} by ordered strategy matchup")
        ax.set_xlabel("Seller strategy")
        ax.set_ylabel("Buyer strategy")
        out_path = output_dir / f"analysis_heatmap_{metric}.png"
        created.append(_save_fig(fig, out_path))
    return created


def plot_reference_distances(runs: pd.DataFrame, output_dir: Path) -> Path:
    needed = {
        "profile_id",
        "n_steps",
        "nash_distance",
        "kalai_distance",
        "ks_distance",
    }
    _require_columns(runs, needed, "main_runs.csv")

    dist = runs.dropna(subset=["nash_distance", "kalai_distance", "ks_distance"], how="all").copy()
    melted = dist.melt(
        id_vars=["profile_id", "n_steps"],
        value_vars=["nash_distance", "kalai_distance", "ks_distance"],
        var_name="reference_metric",
        value_name="distance",
    ).dropna(subset=["distance"])

    g = sns.catplot(
        data=melted,
        x="reference_metric",
        y="distance",
        hue="n_steps",
        col="profile_id",
        col_wrap=3,
        kind="bar",
        height=3.6,
        aspect=1.3,
        errorbar=("ci", 95),
    )
    g.set_axis_labels("Reference point", "Distance")
    g.set_titles("profile={col_name}")
    g.fig.suptitle("Distances to Nash / Kalai / KS reference points", y=1.02)
    return _catplot_save(g, output_dir / "analysis_reference_distances.png")


def plot_efficiency_metrics(runs: pd.DataFrame, output_dir: Path) -> Path:
    needed = {"profile_id", "n_steps", "money_left_on_table", "welfare_regret"}
    _require_columns(runs, needed, "main_runs.csv")

    agg = runs.groupby(["profile_id", "n_steps"], as_index=False)[["money_left_on_table", "welfare_regret"]].mean()
    melted = agg.melt(
        id_vars=["profile_id", "n_steps"],
        value_vars=["money_left_on_table", "welfare_regret"],
        var_name="efficiency_metric",
        value_name="value",
    )

    g = sns.catplot(
        data=melted,
        x="efficiency_metric",
        y="value",
        hue="n_steps",
        col="profile_id",
        col_wrap=3,
        kind="bar",
        height=3.6,
        aspect=1.3,
        errorbar=("ci", 95),
    )
    g.set_axis_labels("Metric", "Mean value")
    g.set_titles("profile={col_name}")
    g.fig.suptitle("Money left on table and welfare regret", y=1.02)
    return _catplot_save(g, output_dir / "analysis_efficiency_metrics.png")


def plot_failure_tradeoff(runs: pd.DataFrame, output_dir: Path) -> Path:
    needed = {
        "profile_id",
        "n_steps",
        "strategy_buyer",
        "strategy_seller",
        "agreement",
        "welfare",
        "welfare_regret",
    }
    _require_columns(runs, needed, "main_runs.csv")

    agg = runs.groupby(["profile_id", "n_steps", "strategy_buyer", "strategy_seller"], as_index=False).agg(
        agreement_rate=("agreement", "mean"),
        mean_welfare=("welfare", "mean"),
        mean_welfare_regret=("welfare_regret", "mean"),
    )
    agg["failure_rate"] = 1.0 - agg["agreement_rate"]

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=agg,
        x="failure_rate",
        y="mean_welfare",
        hue="n_steps",
        style="profile_id",
        size="mean_welfare_regret",
        sizes=(40, 220),
        alpha=0.8,
        ax=ax,
    )
    ax.set_title("Failure-rate vs welfare tradeoff by strategy pairing")
    ax.set_xlabel("Failure rate (1 - agreement rate)")
    ax.set_ylabel("Mean welfare")
    return _save_fig(fig, output_dir / "analysis_failure_vs_welfare_tradeoff.png")


def _make_role_view(runs: pd.DataFrame) -> pd.DataFrame:
    buyer = runs[["profile_id", "n_steps", "strategy_buyer", "u_buyer"]].rename(
        columns={"strategy_buyer": "strategy", "u_buyer": "own_utility"}
    )
    buyer["role"] = "buyer"

    seller = runs[["profile_id", "n_steps", "strategy_seller", "u_seller"]].rename(
        columns={"strategy_seller": "strategy", "u_seller": "own_utility"}
    )
    seller["role"] = "seller"

    return pd.concat([buyer, seller], ignore_index=True)


def plot_strategy_utility_by_role(runs: pd.DataFrame, output_dir: Path) -> Path:
    needed = {"strategy_buyer", "strategy_seller", "u_buyer", "u_seller"}
    _require_columns(runs, needed, "main_runs.csv")

    role_df = _make_role_view(runs)
    agg = role_df.groupby(["strategy", "role"], as_index=False)["own_utility"].mean()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=agg, x="strategy", y="own_utility", hue="role", errorbar=("ci", 95), ax=ax)
    ax.set_title("Mean own utility by strategy and role")
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Mean own utility")
    ax.tick_params(axis="x", rotation=30)
    return _save_fig(fig, output_dir / "analysis_strategy_own_utility_by_role.png")


def plot_trajectory_example(
    output_dir: Path,
    profile_id: str,
    buyer_strategy: str,
    seller_strategy: str,
    n_steps: int,
) -> Path:
    from src.domain import make_mechanism, make_ufuns
    from src.negotiators import strategy_registry

    registry = strategy_registry()
    mechanism = make_mechanism(n_steps=n_steps)
    buyer_ufun, seller_ufun = make_ufuns(profile_id, mechanism.outcome_space)
    mechanism.add(registry[buyer_strategy](name="buyer"), ufun=buyer_ufun)
    mechanism.add(registry[seller_strategy](name="seller"), ufun=seller_ufun)
    state = mechanism.run()

    trajectory: list[tuple[int, float, float]] = []
    for step, _negotiator, offer in mechanism.extended_trace:
        if offer is None:
            continue
        trajectory.append((int(step), float(buyer_ufun(offer)), float(seller_ufun(offer))))

    fig, ax = plt.subplots(figsize=(7, 6))
    if trajectory:
        xs = [row[1] for row in trajectory]
        ys = [row[2] for row in trajectory]
        indices = list(range(len(trajectory)))
        ax.plot(xs, ys, color="tab:blue", alpha=0.5, linewidth=1.2)
        scatter = ax.scatter(xs, ys, c=indices, cmap="viridis", s=28, alpha=0.9)
        fig.colorbar(scatter, ax=ax, label="Offer index")
        ax.scatter([xs[0]], [ys[0]], color="black", marker="o", s=70, label="start")
        ax.scatter([xs[-1]], [ys[-1]], color="red", marker="X", s=90, label="last")

    frontier_utils, _ = mechanism.pareto_frontier()
    if frontier_utils:
        fx = [float(p[0]) for p in frontier_utils]
        fy = [float(p[1]) for p in frontier_utils]
        ax.scatter(fx, fy, color="tab:gray", alpha=0.4, s=12, label="Pareto frontier")

    if state.agreement is not None:
        bu = float(buyer_ufun(state.agreement))
        su = float(seller_ufun(state.agreement))
        ax.scatter([bu], [su], color="gold", edgecolor="black", marker="*", s=180, label="agreement")

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Buyer utility")
    ax.set_ylabel("Seller utility")
    ax.set_title(
        f"Utility-space trajectory ({buyer_strategy} vs {seller_strategy}, profile={profile_id}, n_steps={n_steps})"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    return _save_fig(fig, output_dir / "analysis_utility_trajectory_example.png")


def plot_multilateral(multilateral_runs: pd.DataFrame, output_dir: Path) -> list[Path]:
    required = {"profile_id", "n_steps", "strategy_buyer", "agreement", "welfare_regret", "welfare"}
    _require_columns(multilateral_runs, required, "multi_runs.csv")

    created: list[Path] = []

    agg = multilateral_runs.groupby(["profile_id", "n_steps", "strategy_buyer"], as_index=False)["agreement"].mean()
    g1 = sns.catplot(
        data=agg,
        x="strategy_buyer",
        y="agreement",
        hue="n_steps",
        col="profile_id",
        col_wrap=3,
        kind="bar",
        height=3.6,
        aspect=1.3,
        errorbar=None,
    )
    g1.set_axis_labels("Buyer strategy", "Agreement rate")
    for ax in g1.axes.flat:
        ax.tick_params(axis="x", rotation=30)
    g1.fig.suptitle("Multilateral: agreement rate by buyer strategy", y=1.02)
    created.append(_catplot_save(g1, output_dir / "analysis_multilateral_agreement_by_buyer_strategy.png"))

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=multilateral_runs, x="strategy_buyer", y="welfare_regret", hue="profile_id", ax=ax)
    ax.set_title("Multilateral: welfare regret by buyer strategy")
    ax.set_xlabel("Buyer strategy")
    ax.set_ylabel("Welfare regret")
    ax.tick_params(axis="x", rotation=30)
    created.append(_save_fig(fig, output_dir / "analysis_multilateral_welfare_regret_by_buyer_strategy.png"))

    return created


def main() -> None:
    args = _parse_args()
    sns.set_theme(style="whitegrid")

    results_dir = args.results_dir
    runs_path = results_dir / "main_runs.csv"
    if not runs_path.exists():
        raise FileNotFoundError(f"Could not find {runs_path}")

    output_dir = args.output_dir or (results_dir / "figures_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = pd.read_csv(runs_path)

    created: list[Path] = []
    created.append(plot_agreement_rate(runs, output_dir))
    rounds_path = plot_rounds_to_agreement(runs, output_dir)
    if rounds_path is not None:
        created.append(rounds_path)
    created.append(plot_effective_rounds(runs, output_dir))
    created.append(plot_welfare_distribution(runs, output_dir))
    created.append(plot_pareto_distance(runs, output_dir))
    created.extend(plot_pairing_heatmaps(runs, output_dir))

    # New-project additions.
    created.append(plot_reference_distances(runs, output_dir))
    created.append(plot_efficiency_metrics(runs, output_dir))
    created.append(plot_failure_tradeoff(runs, output_dir))
    created.append(plot_strategy_utility_by_role(runs, output_dir))
    created.append(
        plot_trajectory_example(
            output_dir,
            profile_id=args.trajectory_profile,
            buyer_strategy=args.trajectory_buyer,
            seller_strategy=args.trajectory_seller,
            n_steps=args.trajectory_steps,
        )
    )

    multilateral_path = args.multilateral_runs
    if multilateral_path is None:
        candidate = Path("results/multilateral/multi_runs.csv")
        if candidate.exists():
            multilateral_path = candidate

    if multilateral_path is not None and Path(multilateral_path).exists():
        multi_runs = pd.read_csv(multilateral_path)
        created.extend(plot_multilateral(multi_runs, output_dir))

    print(f"Wrote {len(created)} figure(s) to {output_dir}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()

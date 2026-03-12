from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_METRICS = (
    "agreement",
    "own_utility",
    "welfare",
    "nash_distance",
    "kalai_distance",
    "ks_distance",
    "money_left_on_table",
    "welfare_regret",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strategy summary + pairwise significance outputs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/main"),
        help="Directory containing main_strategy_summary.csv and main_pairwise_tests.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write figures (default: <results-dir>/figures)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics to plot (must exist in the summary/test CSVs)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="FDR threshold for significance markers on pairwise heatmaps",
    )
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, needed: set[str], source_name: str) -> None:
    missing = sorted(needed.difference(df.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _clean_name(value: str) -> str:
    return str(value).replace(" ", "_").replace("/", "_")


def plot_strategy_bootstrap(summary_df: pd.DataFrame, output_dir: Path, metrics: list[str]) -> list[Path]:
    created: list[Path] = []
    roles = sorted(summary_df["role"].dropna().astype(str).unique())
    profiles = sorted(summary_df["profile_id"].dropna().astype(str).unique())
    steps = sorted(summary_df["n_steps"].dropna().unique())

    for metric in metrics:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_boot_ci_low"
        high_col = f"{metric}_boot_ci_high"
        required = {"profile_id", "n_steps", "role", "strategy", mean_col, low_col, high_col}
        _require_columns(summary_df, required, "main_strategy_summary.csv")

        for role in roles:
            role_df = summary_df[summary_df["role"].astype(str) == role].copy()
            if role_df.empty:
                continue

            nrows = max(1, len(profiles))
            ncols = max(1, len(steps))
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(4.5 * ncols, 3.5 * nrows),
                squeeze=False,
                constrained_layout=True,
            )
            fig.suptitle(f"{metric} by strategy ({role}) with bootstrap CI", fontsize=12)

            for r, profile_id in enumerate(profiles):
                for c, n_steps in enumerate(steps):
                    ax = axes[r][c]
                    subset = role_df[(role_df["profile_id"].astype(str) == profile_id) & (role_df["n_steps"] == n_steps)]
                    if subset.empty:
                        ax.set_axis_off()
                        continue

                    subset = subset.sort_values("strategy")
                    x_labels = subset["strategy"].astype(str).tolist()
                    x = list(range(len(x_labels)))
                    y = subset[mean_col].astype(float).to_numpy()
                    low = subset[low_col].astype(float).to_numpy()
                    high = subset[high_col].astype(float).to_numpy()
                    # Bootstrap intervals can be noisy with small samples; make
                    # sure errorbar extents are non-negative for Matplotlib.
                    low = np.where(np.isnan(low), y, low)
                    high = np.where(np.isnan(high), y, high)
                    low = np.minimum(low, y)
                    high = np.maximum(high, y)
                    yerr_low = np.maximum(0.0, y - low)
                    yerr_high = np.maximum(0.0, high - y)

                    ax.errorbar(
                        x,
                        y,
                        yerr=[yerr_low, yerr_high],
                        fmt="o",
                        capsize=3,
                        linestyle="none",
                        color="#1f77b4",
                    )
                    ax.set_xticks(x, x_labels, rotation=30, ha="right")
                    ax.set_title(f"profile={profile_id}, n_steps={n_steps}")
                    ax.set_ylabel(metric)
                    ax.grid(alpha=0.25, axis="y")

            out_path = output_dir / f"strategy_bootstrap_{_clean_name(metric)}_{_clean_name(role)}.png"
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            created.append(out_path)

    return created


def _effect_size_matrix(
    pairwise_subset: pd.DataFrame, strategies: list[str], alpha: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect = pd.DataFrame(0.0, index=strategies, columns=strategies)
    significant = pd.DataFrame(False, index=strategies, columns=strategies)

    for _, row in pairwise_subset.iterrows():
        a = str(row["strategy_a"])
        b = str(row["strategy_b"])
        d = float(row["effect_size_cohen_d"])
        q = float(row["p_value_fdr"])
        effect.loc[a, b] = d
        effect.loc[b, a] = -d
        sig = q < alpha
        significant.loc[a, b] = sig
        significant.loc[b, a] = sig

    return effect, significant


def plot_pairwise_effect_heatmaps(
    pairwise_df: pd.DataFrame, output_dir: Path, metrics: list[str], alpha: float
) -> list[Path]:
    created: list[Path] = []
    required = {
        "profile_id",
        "n_steps",
        "role",
        "metric",
        "strategy_a",
        "strategy_b",
        "effect_size_cohen_d",
        "p_value_fdr",
    }
    _require_columns(pairwise_df, required, "main_pairwise_tests.csv")

    contexts = (
        pairwise_df[["profile_id", "n_steps", "role"]].drop_duplicates().sort_values(["profile_id", "n_steps", "role"])
    )
    for _, context in contexts.iterrows():
        profile_id = str(context["profile_id"])
        n_steps = context["n_steps"]
        role = str(context["role"])
        context_subset = pairwise_df[
            (pairwise_df["profile_id"].astype(str) == profile_id)
            & (pairwise_df["n_steps"] == n_steps)
            & (pairwise_df["role"].astype(str) == role)
        ]

        for metric in metrics:
            subset = context_subset[context_subset["metric"].astype(str) == metric]
            if subset.empty:
                continue

            strategies = sorted(set(subset["strategy_a"].astype(str)).union(set(subset["strategy_b"].astype(str))))
            effect, significant = _effect_size_matrix(subset, strategies, alpha)

            annot = effect.copy().map(lambda x: f"{x:.2f}")
            for i in strategies:
                for j in strategies:
                    if i == j:
                        annot.loc[i, j] = "0.00"
                    elif significant.loc[i, j]:
                        annot.loc[i, j] = f"{effect.loc[i, j]:.2f}*"

            fig, ax = plt.subplots(figsize=(1.1 * len(strategies) + 3, 1.1 * len(strategies) + 2), constrained_layout=True)
            sns.heatmap(
                effect,
                ax=ax,
                cmap="coolwarm",
                center=0.0,
                annot=annot,
                fmt="",
                linewidths=0.4,
                cbar_kws={"label": "Cohen's d (row - col)"},
            )
            ax.set_title(f"Pairwise effect sizes: {metric} | profile={profile_id}, n_steps={n_steps}, role={role}")
            ax.set_xlabel("strategy")
            ax.set_ylabel("strategy")

            out_path = (
                output_dir
                / f"pairwise_effect_{_clean_name(metric)}_profile-{_clean_name(profile_id)}_steps-{n_steps}_role-{_clean_name(role)}.png"
            )
            fig.savefig(out_path, dpi=170)
            plt.close(fig)
            created.append(out_path)

    return created


def main() -> None:
    args = _parse_args()
    sns.set_theme(style="whitegrid")

    results_dir = args.results_dir
    output_dir = args.output_dir or (results_dir / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    strategy_summary_path = results_dir / "main_strategy_summary.csv"
    pairwise_path = results_dir / "main_pairwise_tests.csv"

    if not strategy_summary_path.exists():
        raise FileNotFoundError(f"Could not find {strategy_summary_path}")
    if not pairwise_path.exists():
        raise FileNotFoundError(f"Could not find {pairwise_path}")

    strategy_summary_df = pd.read_csv(strategy_summary_path)
    pairwise_df = pd.read_csv(pairwise_path)

    metrics = [str(m) for m in args.metrics]
    created: list[Path] = []
    created.extend(plot_strategy_bootstrap(strategy_summary_df, output_dir, metrics))
    created.extend(plot_pairwise_effect_heatmaps(pairwise_df, output_dir, metrics, args.alpha))

    print(f"Wrote {len(created)} figure(s) to {output_dir}")
    for path in created:
        print(path)


if __name__ == "__main__":
    main()

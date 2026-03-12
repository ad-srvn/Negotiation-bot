from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import pandas as pd


@dataclass(frozen=True)
class RunLayout:
    root: Path
    main_dir: Path
    multilateral_dir: Path
    appendix_dir: Path
    figures_stats_dir: Path
    figures_analysis_dir: Path
    logs_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean workspace artifacts and run the full NegMAS pipeline sequentially."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/presentation"),
        help="Root folder where presentation-ready run bundles are created.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional run label. Default: run_YYYYMMDD_HHMMSS",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a fast verification pipeline instead of the full matrix.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip pytest stage.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Skip cache cleanup and result archiving.",
    )
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _archive_legacy_results(project_root: Path, output_root: Path) -> Path | None:
    results_root = project_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    archive_root = results_root / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)

    to_archive: list[Path] = []
    for child in results_root.iterdir():
        if child.name == "archive":
            continue
        if child == output_root:
            continue
        to_archive.append(child)

    if not to_archive:
        return None

    archive_dir = archive_root / f"pre_cleanup_{_timestamp()}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for child in to_archive:
        target = archive_dir / child.name
        if target.exists():
            target = archive_dir / f"{child.name}_{_timestamp()}"
        shutil.move(str(child), str(target))
    return archive_dir


def _cleanup_caches(project_root: Path) -> dict[str, int]:
    removed = {"dirs": 0, "files": 0}
    dir_names = {"__pycache__", ".pytest_cache", ".pycache_tmp", ".ipynb_checkpoints"}
    for path in sorted(project_root.rglob("*")):
        if path.is_dir() and path.name in dir_names:
            shutil.rmtree(path, ignore_errors=True)
            removed["dirs"] += 1
    for path in sorted(project_root.rglob(".DS_Store")):
        try:
            path.unlink()
            removed["files"] += 1
        except FileNotFoundError:
            pass
    return removed


def _make_layout(project_root: Path, output_root: Path, label: str) -> RunLayout:
    root = output_root / label
    layout = RunLayout(
        root=root,
        main_dir=root / "main",
        multilateral_dir=root / "multilateral",
        appendix_dir=root / "appendix",
        figures_stats_dir=root / "figures" / "stats",
        figures_analysis_dir=root / "figures" / "analysis",
        logs_dir=root / "logs",
    )
    for path in (
        layout.root,
        layout.main_dir,
        layout.multilateral_dir,
        layout.appendix_dir,
        layout.figures_stats_dir,
        layout.figures_analysis_dir,
        layout.logs_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    (project_root / "results" / "presentation").mkdir(parents=True, exist_ok=True)
    return layout


def _stream_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> None:
    env = os.environ.copy()
    mpl_dir = cwd / ".mplconfig"
    cache_dir = cwd / ".cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLBACKEND"] = "Agg"
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    with log_path.open("w", encoding="utf-8") as log:
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(cmd)}")


def _run_step(step_name: str, cmd: list[str], project_root: Path, logs_dir: Path) -> dict:
    print(f"\n=== {step_name} ===")
    print(" ".join(cmd))
    started = perf_counter()
    log_path = logs_dir / f"{step_name.lower().replace(' ', '_')}.log"
    _stream_subprocess(cmd, cwd=project_root, log_path=log_path)
    duration = perf_counter() - started
    print(f"[done] {step_name} ({duration:.1f}s)")
    return {
        "step": step_name,
        "command": cmd,
        "log": str(log_path),
        "duration_seconds": duration,
    }


def _summary_markdown(layout: RunLayout, quick: bool) -> str:
    main_runs = pd.read_csv(layout.main_dir / "main_runs.csv")
    main_summary = pd.read_csv(layout.main_dir / "main_summary.csv")
    multi_runs = pd.read_csv(layout.multilateral_dir / "multi_runs.csv")

    def ordered_pair(df: pd.DataFrame) -> pd.Series:
        return df["strategy_buyer"].astype(str) + " vs " + df["strategy_seller"].astype(str)

    main_runs = main_runs.copy()
    main_runs["ordered_pair"] = ordered_pair(main_runs)
    pair_agg = (
        main_runs.groupby("ordered_pair", as_index=False)
        .agg(
            agreement_rate=("agreement", "mean"),
            welfare_mean=("welfare", "mean"),
            pareto_distance_mean=("pareto_distance", "mean"),
            rounds_agreement_mean=("rounds", lambda s: float(s[main_runs.loc[s.index, "agreement"] == 1].mean())),
            count=("run_id", "count"),
        )
        .sort_values(["welfare_mean", "agreement_rate"], ascending=[False, False])
    )

    best_welfare = pair_agg.iloc[0]
    best_pareto = pair_agg.sort_values("pareto_distance_mean", ascending=True).iloc[0]

    buyer_role = main_runs[["strategy_buyer", "u_buyer", "agreement"]].rename(
        columns={"strategy_buyer": "strategy", "u_buyer": "own_utility"}
    )
    seller_role = main_runs[["strategy_seller", "u_seller", "agreement"]].rename(
        columns={"strategy_seller": "strategy", "u_seller": "own_utility"}
    )
    role_df = pd.concat([buyer_role, seller_role], ignore_index=True)
    strategy_agg = (
        role_df.groupby("strategy", as_index=False)
        .agg(own_utility_mean=("own_utility", "mean"), agreement_rate=("agreement", "mean"))
        .sort_values(["own_utility_mean", "agreement_rate"], ascending=[False, False])
    )

    appendix_note = "appendix_scores.csv contains NegMAS tournament summary metrics (including `advantage`)."
    if (layout.appendix_dir / "appendix_scores.csv").exists():
        appendix_note = (
            "appendix_scores.csv written with aggregated tournament metrics; `advantage` is directly produced by NegMAS."
        )

    top_strategy_table = strategy_agg.head(5).to_string(index=False)
    top_pair_table = pair_agg.head(10).to_string(index=False)

    lines = [
        "# Presentation Summary",
        "",
        f"- Run mode: {'quick' if quick else 'full'}",
        f"- Main runs: {len(main_runs)}",
        f"- Main summary rows: {len(main_summary)}",
        f"- Main agreement rate: {main_runs['agreement'].mean():.4f}",
        f"- Main strategies observed: {', '.join(sorted(main_runs['strategy_buyer'].astype(str).unique()))}",
        f"- Best ordered pairing by welfare: {best_welfare['ordered_pair']} (mean={best_welfare['welfare_mean']:.4f})",
        f"- Best ordered pairing by Pareto distance: {best_pareto['ordered_pair']} (mean={best_pareto['pareto_distance_mean']:.4f})",
        f"- Multilateral runs: {len(multi_runs)}",
        f"- Multilateral agreement rate: {multi_runs['agreement'].mean():.4f}",
        f"- {appendix_note}",
        "",
        "## Top Strategies (Role-Aggregated)",
        "",
        "```text",
        top_strategy_table,
        "```",
        "",
        "## Ordered Pairing Snapshot",
        "",
        "```text",
        top_pair_table,
        "```",
        "",
        "## Result Layout",
        "",
        "- `main/`: raw and aggregated bilateral outputs",
        "- `multilateral/`: raw and aggregated multilateral outputs",
        "- `appendix/`: tournament outputs",
        "- `figures/stats/`: bootstrap/effect-size figures",
        "- `figures/analysis/`: presentation plots",
        "- `logs/`: full sequential run logs",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[1]
    output_root = (project_root / args.output_root).resolve()
    label = args.label or f"run_{_timestamp()}"

    cleanup_report: dict[str, object] = {"archived_to": None, "removed_dirs": 0, "removed_files": 0}
    if not args.skip_cleanup:
        archived_to = _archive_legacy_results(project_root, output_root)
        removed = _cleanup_caches(project_root)
        cleanup_report["archived_to"] = str(archived_to) if archived_to is not None else None
        cleanup_report["removed_dirs"] = removed["dirs"]
        cleanup_report["removed_files"] = removed["files"]

    layout = _make_layout(project_root, output_root, label)
    py = sys.executable

    if args.quick:
        main_cmd = [
            py,
            "-m",
            "src.run_experiments",
            "--output-dir",
            str(layout.main_dir),
            "--n-seeds",
            "3",
            "--profiles",
            "A",
            "B",
            "--n-steps",
            "12",
            "24",
            "--time-limit",
            "1.0",
            "--bootstrap-resamples",
            "200",
            "--permutation-resamples",
            "200",
            "--progress-every",
            "20",
        ]
        multi_cmd = [
            py,
            "-m",
            "src.run_multilateral_experiments",
            "--output-dir",
            str(layout.multilateral_dir),
            "--n-seeds",
            "3",
            "--profiles",
            "M1",
            "M2",
            "--n-steps",
            "20",
            "--time-limit",
            "1.0",
            "--n-suppliers",
            "2",
            "--bootstrap-resamples",
            "200",
            "--progress-every",
            "20",
        ]
        appendix_cmd = [
            py,
            "-m",
            "src.appendix_tournament",
            "--output-dir",
            str(layout.appendix_dir),
            "--n-scenarios",
            "3",
            "--n-repetitions",
            "1",
            "--n-steps",
            "30",
            "--seed",
            "2026",
        ]
    else:
        main_cmd = [
            py,
            "-m",
            "src.run_experiments",
            "--output-dir",
            str(layout.main_dir),
            "--n-seeds",
            "40",
            "--profiles",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "--n-steps",
            "12",
            "24",
            "36",
            "--time-limit",
            "1.0",
            "--buyer-reserved-value",
            "0.35",
            "--seller-reserved-value",
            "0.35",
            "--pend",
            "0.04",
            "--pend-per-second",
            "0.02",
            "--bootstrap-resamples",
            "2000",
            "--permutation-resamples",
            "2000",
            "--progress-every",
            "50",
        ]
        multi_cmd = [
            py,
            "-m",
            "src.run_multilateral_experiments",
            "--output-dir",
            str(layout.multilateral_dir),
            "--n-seeds",
            "20",
            "--profiles",
            "M1",
            "M2",
            "M3",
            "--n-steps",
            "20",
            "40",
            "--time-limit",
            "1.0",
            "--n-suppliers",
            "2",
            "--buyer-reserved-value",
            "0.35",
            "--supplier-reserved-value",
            "0.35",
            "--pend",
            "0.04",
            "--pend-per-second",
            "0.02",
            "--bootstrap-resamples",
            "1000",
            "--progress-every",
            "50",
        ]
        appendix_cmd = [
            py,
            "-m",
            "src.appendix_tournament",
            "--output-dir",
            str(layout.appendix_dir),
            "--n-scenarios",
            "6",
            "--n-repetitions",
            "2",
            "--n-steps",
            "50",
            "--seed",
            "2026",
        ]

    steps: list[dict] = []
    started_wall = datetime.now().isoformat(timespec="seconds")
    started = perf_counter()
    status = "success"
    error_message: str | None = None

    try:
        steps.append(_run_step("Smoke Test", [py, "scripts/smoke_test.py"], project_root, layout.logs_dir))
        if not args.skip_tests:
            steps.append(_run_step("Pytest", [py, "-m", "pytest", "-q"], project_root, layout.logs_dir))
        steps.append(_run_step("Main Experiments", main_cmd, project_root, layout.logs_dir))
        steps.append(
            _run_step(
                "Plot Stats",
                [
                    py,
                    "scripts/plot_stats.py",
                    "--results-dir",
                    str(layout.main_dir),
                    "--output-dir",
                    str(layout.figures_stats_dir),
                ],
                project_root,
                layout.logs_dir,
            )
        )
        steps.append(_run_step("Appendix Tournament", appendix_cmd, project_root, layout.logs_dir))
        steps.append(_run_step("Multilateral Experiments", multi_cmd, project_root, layout.logs_dir))
        steps.append(
            _run_step(
                "Plot Analysis",
                [
                    py,
                    "scripts/plot_analysis.py",
                    "--results-dir",
                    str(layout.main_dir),
                    "--output-dir",
                    str(layout.figures_analysis_dir),
                    "--multilateral-runs",
                    str(layout.multilateral_dir / "multi_runs.csv"),
                ],
                project_root,
                layout.logs_dir,
            )
        )
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        finished_wall = datetime.now().isoformat(timespec="seconds")
        total_duration = perf_counter() - started
        manifest = {
            "started_at": started_wall,
            "finished_at": finished_wall,
            "duration_seconds": total_duration,
            "mode": "quick" if args.quick else "full",
            "status": status,
            "error": error_message,
            "cleanup": cleanup_report,
            "steps": steps,
            "layout": {
                "root": str(layout.root),
                "main_dir": str(layout.main_dir),
                "appendix_dir": str(layout.appendix_dir),
                "multilateral_dir": str(layout.multilateral_dir),
                "figures_stats_dir": str(layout.figures_stats_dir),
                "figures_analysis_dir": str(layout.figures_analysis_dir),
                "logs_dir": str(layout.logs_dir),
            },
        }
        (layout.root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_md = _summary_markdown(layout, quick=args.quick)
    (layout.root / "SUMMARY.md").write_text(summary_md, encoding="utf-8")
    (project_root / "results" / "presentation" / "LATEST.txt").write_text(str(layout.root) + "\n", encoding="utf-8")

    print("\n=== Completed Sequential Pipeline ===")
    print(f"Bundle: {layout.root}")
    print(f"Summary: {layout.root / 'SUMMARY.md'}")
    print(f"Manifest: {layout.root / 'manifest.json'}")


if __name__ == "__main__":
    main()

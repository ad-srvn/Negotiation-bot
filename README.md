# Game Theory Negotiation Project

# Automated Negotiation Experiments (NegMAS)

This is the codebase for the Game Theory project.  
It runs three experiment tracks end-to-end:

- Bilateral fixed-domain experiments (`A..F`)
- Multilateral procurement experiments (`M1..M3`)
- Appendix robustness tournament

## 1) Setup

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) One-Command Sequential Pipeline (Recommended)

```bash
python scripts/run_all.py
```

What it does, sequentially:

1. Cleans cache artifacts and removes stale metadata files.
2. Runs smoke test.
3. Runs `pytest`.
4. Runs full bilateral experiments.
5. Generates bilateral stats/effect-size figures.
6. Runs appendix tournament.
7. Runs multilateral experiments.
8. Generates analysis/presentation figures.
9. Writes `SUMMARY.md` and `manifest.json` for the run.

Notes:

- New runs are written to a new folder under `results/presentation/`.
- `results/presentation/LATEST.txt` is updated to point to the newest run.

Quick verification mode:

```bash
python scripts/run_all.py --quick
```

## 3) Output Layout

Each run is saved under:

```text
results/presentation/run_YYYYMMDD_HHMMSS/
```

Inside each run bundle:

- `main/`: `main_runs.csv`, `main_summary.csv`, `main_strategy_summary.csv`, `main_pairwise_tests.csv`, `main_config.json`
- `multilateral/`: `multi_runs.csv`, `multi_summary.csv`, `multi_config.json`
- `appendix/`: `appendix_scores.csv`, `appendix_results.pkl`, `appendix_config.json`, tournament artifacts
- `figures/stats/`: bootstrap CI and pairwise effect-size figures
- `figures/analysis/`: presentation figures (including ordered matchup heatmaps)
- `logs/`: full logs for every sequential step
- `SUMMARY.md`: short narrative summary + top pairings/strategies
- `manifest.json`: reproducibility metadata and executed commands

`results/presentation/LATEST.txt` points to the newest run bundle path.

## 4) Strategies

- `BoulwareTBNegotiator`
- `LinearTBNegotiator`
- `ConcederTBNegotiator`
- `SmartAspirationNegotiator` (custom, opponent-aware aspiration)
- `GeniusBOABaselineNegotiator` (BOA-inspired baseline implemented in NegMAS, not an external GENIUS bridge)

## 5) Technical Interpretation Notes

- Pairing-dependent metrics are analyzed by **ordered matchup** (`buyer_strategy`, `seller_strategy`) to avoid buyer-only confounding.
- Rounds are reported both:
  - on agreements only, and
  - as **effective rounds** where failures are counted as consuming the deadline.
- Pairwise tests use permutation p-values with FDR correction and Cohen's d.

## 6) Direct Commands (if needed)

Bilateral only:

```bash
python -m src.run_experiments --output-dir results/main_final
```

Multilateral only:

```bash
python -m src.run_multilateral_experiments --output-dir results/multilateral
```

Appendix only:

```bash
python -m src.appendix_tournament --output-dir results/appendix
```

Generate figures for an existing run:

```bash
python scripts/plot_stats.py --results-dir <RUN_DIR>/main --output-dir <RUN_DIR>/figures/stats
python scripts/plot_analysis.py --results-dir <RUN_DIR>/main --output-dir <RUN_DIR>/figures/analysis --multilateral-runs <RUN_DIR>/multilateral/multi_runs.csv
```

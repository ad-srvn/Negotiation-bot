# Automated Negotiation Experiments (NegMAS)

This repository implements bilateral and multilateral SAO (Stacked Alternating Offers) negotiation experiments in NegMAS. It compares classic time-based concession strategies with a custom opponent-aware aspiration strategy under deadline pressure, reservation values, asymmetric utility profiles, and both two-party and one-to-many procurement settings.

The project is designed for reproducible benchmarking: it includes fixed-domain experiments, a multilateral extension, a randomized appendix tournament, and statistical analysis outputs (bootstrap confidence intervals, permutation tests, FDR correction, and effect sizes).

## Project Parts
- `src/run_experiments.py`: bilateral fixed-domain runner (`A..F`)
- `src/run_multilateral_experiments.py`: multilateral procurement runner (`M1..M3`)
- `src/appendix_tournament.py`: randomized robustness tournament
- `src/domain.py`, `src/domain_multilateral.py`: issue spaces and utility-profile generators
- `src/negotiators.py`: strategy registry, custom `SmartAspirationNegotiator`, and `GeniusBOABaselineNegotiator`
- `src/metrics.py`: frontier/reference metrics, welfare regret, CIs, and pairwise significance tests
- `scripts/run_all.py`: end-to-end sequential pipeline
- `scripts/plot_stats.py`: bootstrap/effect-size figures from bilateral summaries
- `scripts/plot_analysis.py`: presentation analysis figures from run CSVs
- `tests/`: unit and integration checks

## Strategy Set
- `BoulwareTBNegotiator`
- `LinearTBNegotiator`
- `ConcederTBNegotiator`
- `SmartAspirationNegotiator`
- `GeniusBOABaselineNegotiator`

## Profile Families
### Bilateral (`A..F`) in `src/domain.py`
Issues: `price`, `quantity`, `delivery_time`, `service_level`.

- `A`: balanced baseline buyer/seller preferences
- `B`: seller is highly delivery-time sensitive
- `C`: nonlinear price-vs-quantity tension
- `D`: service-level dominant weighting
- `E`: delivery-pressure and price asymmetry
- `F`: strong specialization (buyer quantity vs seller price emphasis)

### Multilateral (`M1..M3`) in `src/domain_multilateral.py`
Issues: `unit_price`, `volume`, `delivery_window`, `quality_tier` with one buyer and multiple suppliers.

- `M1`: balanced procurement setting
- `M2`: buyer emphasizes speed and quality
- `M3`: buyer is strongly cost-focused

## Environment Setup
From repo root:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Everything
```bash
python scripts/run_all.py
```

This pipeline runs smoke test, pytest, bilateral experiments, stats plots, appendix tournament, multilateral experiments, and analysis plots sequentially, then writes a run bundle with `SUMMARY.md` and `manifest.json`.

Quick verification mode:
```bash
python scripts/run_all.py --quick
```

## Run Components Manually
### Smoke test
```bash
python scripts/smoke_test.py
```

### Tests
```bash
pytest -q
```

### Bilateral main experiments
```bash
python -m src.run_experiments \
  --output-dir results/main_final \
  --n-seeds 40 \
  --profiles A B C D E F \
  --n-steps 12 24 36 \
  --buyer-reserved-value 0.35 \
  --seller-reserved-value 0.35 \
  --pend 0.04 \
  --pend-per-second 0.02 \
  --bootstrap-resamples 2000 \
  --permutation-resamples 2000 \
  --progress-every 50
```

### Appendix tournament
```bash
python -m src.appendix_tournament \
  --output-dir results/appendix \
  --n-scenarios 6 \
  --n-repetitions 2 \
  --n-steps 50 \
  --seed 2026
```

### Multilateral experiments
```bash
python -m src.run_multilateral_experiments \
  --output-dir results/multilateral \
  --n-seeds 20 \
  --profiles M1 M2 M3 \
  --n-steps 20 40 \
  --n-suppliers 2 \
  --buyer-reserved-value 0.35 \
  --supplier-reserved-value 0.35 \
  --pend 0.04 \
  --pend-per-second 0.02 \
  --bootstrap-resamples 1000 \
  --progress-every 50
```

### Plotting from existing results
```bash
python scripts/plot_stats.py \
  --results-dir <RUN_DIR>/main \
  --output-dir <RUN_DIR>/figures/stats

python scripts/plot_analysis.py \
  --results-dir <RUN_DIR>/main \
  --output-dir <RUN_DIR>/figures/analysis \
  --multilateral-runs <RUN_DIR>/multilateral/multi_runs.csv
```

## Output Layout
Default sequential bundles are written to:

```text
results/presentation/run_YYYYMMDD_HHMMSS/
```

Each run bundle contains:
- `main/`: bilateral raw + aggregated outputs
- `multilateral/`: multilateral raw + aggregated outputs
- `appendix/`: tournament outputs
- `figures/stats/`: CI/effect-size figures
- `figures/analysis/`: analysis/presentation figures
- `logs/`: per-stage execution logs
- `SUMMARY.md`: concise results summary
- `manifest.json`: exact commands, timing, and run metadata

`results/presentation/LATEST.txt` points to the latest run bundle.

Current bundled results in this repo are under:
- `results/presentation/run_full_presentable/`

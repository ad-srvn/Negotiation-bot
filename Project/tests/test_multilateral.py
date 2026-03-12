from __future__ import annotations

import pytest

negmas = pytest.importorskip("negmas")

from src.domain_multilateral import VALID_MULTI_PROFILES, make_multilateral_mechanism, make_procurement_ufuns  # noqa: E402
from src.run_multilateral_experiments import MultiExperimentConfig, run_experiments  # noqa: E402


def test_multilateral_domain_and_ufuns_construct() -> None:
    mechanism = make_multilateral_mechanism(n_steps=12)
    buyer_ufun, supplier_ufuns = make_procurement_ufuns(
        profile_id=sorted(VALID_MULTI_PROFILES)[0],
        outcome_space=mechanism.outcome_space,
        n_suppliers=3,
    )
    assert buyer_ufun is not None
    assert len(supplier_ufuns) == 3


def test_multilateral_runner_writes_outputs(tmp_path) -> None:
    config = MultiExperimentConfig(
        output_dir=tmp_path / "multi",
        n_seeds=1,
        profiles=("M1",),
        n_steps_list=(12,),
        n_suppliers=2,
        strategies=("LinearTBNegotiator", "GeniusBOABaselineNegotiator"),
        include_self_play=False,
        bootstrap_resamples=50,
        progress_every=0,
    )
    raw_df, summary_df = run_experiments(config)
    assert len(raw_df) > 0
    assert len(summary_df) > 0
    assert (config.output_dir / "multi_runs.csv").exists()
    assert (config.output_dir / "multi_summary.csv").exists()
    assert "u_supplier_avg" in raw_df.columns
    assert "money_left_on_table" in raw_df.columns

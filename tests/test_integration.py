from __future__ import annotations

import random

import numpy as np
import pytest

negmas = pytest.importorskip("negmas")

from src.domain import VALID_PROFILES, make_mechanism, make_ufuns  # noqa: E402
from src.negotiators import strategy_registry  # noqa: E402


def _run_session(buyer_name: str, seller_name: str, seed: int) -> tuple:
    random.seed(seed)
    np.random.seed(seed)
    registry = strategy_registry()
    session = make_mechanism(n_steps=30)
    buyer_ufun, seller_ufun = make_ufuns("A", session.outcome_space)
    session.add(registry[buyer_name](name=f"buyer_{buyer_name}"), ufun=buyer_ufun)
    session.add(registry[seller_name](name=f"seller_{seller_name}"), ufun=seller_ufun)
    state = session.run()
    return state.agreement, state.step


def test_all_strategies_run_end_to_end() -> None:
    names = list(strategy_registry().keys())
    for name in names:
        agreement, step = _run_session(name, "LinearTBNegotiator", seed=7)
        assert step >= 0
        assert agreement is None or isinstance(agreement, tuple)


def test_reproducible_with_same_seed() -> None:
    first = _run_session("BoulwareTBNegotiator", "ConcederTBNegotiator", seed=11)
    second = _run_session("BoulwareTBNegotiator", "ConcederTBNegotiator", seed=11)
    assert first == second


def test_all_profile_families_construct_and_run() -> None:
    registry = strategy_registry()
    for profile_id in sorted(VALID_PROFILES):
        session = make_mechanism(n_steps=12)
        buyer_ufun, seller_ufun = make_ufuns(profile_id, session.outcome_space)
        session.add(registry["BoulwareTBNegotiator"](name=f"buyer_{profile_id}"), ufun=buyer_ufun)
        session.add(registry["LinearTBNegotiator"](name=f"seller_{profile_id}"), ufun=seller_ufun)
        state = session.run()
        assert state.step >= 0

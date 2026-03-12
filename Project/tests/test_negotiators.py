from __future__ import annotations

import pytest

negmas = pytest.importorskip("negmas")

from negmas.sao.negotiators import BoulwareTBNegotiator  # noqa: E402

from src.domain import make_mechanism, make_ufuns  # noqa: E402
from src.negotiators import GeniusBOABaselineNegotiator, SmartAspirationNegotiator, strategy_registry  # noqa: E402


def test_smart_negotiator_generates_valid_offers() -> None:
    session = make_mechanism(n_steps=20)
    buyer_ufun, seller_ufun = make_ufuns("A", session.outcome_space)
    buyer = SmartAspirationNegotiator(name="buyer")
    seller = BoulwareTBNegotiator(name="seller")
    session.add(buyer, ufun=buyer_ufun)
    session.add(seller, ufun=seller_ufun)
    session.run()

    assert len(session.extended_trace) > 0
    for _, _, offer in session.extended_trace:
        # If this call succeeds and returns finite utility, the offer is valid.
        assert buyer_ufun(offer) >= 0.0
        assert seller_ufun(offer) >= 0.0


def test_smart_negotiator_uses_full_offer_history_for_issue_importance() -> None:
    negotiator = SmartAspirationNegotiator(name="smart")
    negotiator._partner_offers = [
        (9, 1, 9),
        (9, 2, 9),
        (8, 3, 9),
        (8, 4, 9),
    ]
    profile = negotiator._estimate_partner_profile()
    assert profile is not None
    _, weights, _ = profile
    # delivery_time never changes, so it should be inferred most important.
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]


def test_smart_negotiator_switches_to_faster_late_concession() -> None:
    negotiator = SmartAspirationNegotiator(name="smart", aspiration_type="boulware", late_aspiration_type="linear")
    # After the deadline switch, aspiration should drop below pure Boulware.
    assert negotiator._aspiration_utility(0.95) < negotiator._asp_early.utility_at(0.95)


def test_genius_boa_baseline_registered() -> None:
    registry = strategy_registry()
    assert "GeniusBOABaselineNegotiator" in registry
    assert registry["GeniusBOABaselineNegotiator"] is GeniusBOABaselineNegotiator

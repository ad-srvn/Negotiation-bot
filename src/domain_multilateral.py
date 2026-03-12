from __future__ import annotations

import math
from typing import Any, Callable

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


PROFILE_M1 = "M1"
PROFILE_M2 = "M2"
PROFILE_M3 = "M3"
VALID_MULTI_PROFILES = {PROFILE_M1, PROFILE_M2, PROFILE_M3}


def make_procurement_issues() -> list:
    """Creates a one-to-many procurement domain (1 buyer, multiple suppliers)."""
    return [
        make_issue(name="unit_price", values=10),  # 0..9
        make_issue(name="volume", values=list(range(1, 11))),  # 1..10
        make_issue(name="delivery_window", values=["fast", "standard", "slow"]),
        make_issue(name="quality_tier", values=["basic", "standard", "premium"]),
    ]


def make_multilateral_mechanism(
    n_steps: int,
    time_limit: float | None = None,
    pend: float = 0.0,
    pend_per_second: float = 0.0,
) -> SAOMechanism:
    kwargs = {"issues": make_procurement_issues(), "n_steps": n_steps}
    if time_limit is not None:
        kwargs["time_limit"] = time_limit
    kwargs["pend"] = pend
    kwargs["pend_per_second"] = pend_per_second
    return SAOMechanism(**kwargs)


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def _norm_0_9(x: int) -> float:
    return _clip01(float(x) / 9.0)


def _norm_1_10(x: int) -> float:
    return _clip01((float(x) - 1.0) / 9.0)


def _linear_low_0_9(x: int) -> float:
    return 9.0 * (1.0 - _norm_0_9(x))


def _linear_high_0_9(x: int) -> float:
    return 9.0 * _norm_0_9(x)


def _convex_high_0_9(x: int) -> float:
    return 9.0 * (_norm_0_9(x) ** 2)


def _concave_high_0_9(x: int) -> float:
    return 9.0 * math.sqrt(_norm_0_9(x))


def _concave_high_1_10(x: int) -> float:
    return 9.0 * math.sqrt(_norm_1_10(x))


def _concave_low_0_9(x: int) -> float:
    return 9.0 * math.sqrt(1.0 - _norm_0_9(x))


def _volume_peak(target: float, slope: float) -> Callable[[int], float]:
    def value(q: int) -> float:
        return _clip01(1.0 - slope * abs(float(q) - target) / 9.0) * 9.0

    return value


def _categorical(scores: dict[str, float]) -> Callable[[Any], float]:
    def value(option: Any) -> float:
        return float(scores.get(str(option), 0.0))

    return value


def _buyer_profile(profile_id: str) -> tuple[dict[str, Callable[[Any], float]], dict[str, float]]:
    base_values = {
        "unit_price": _linear_low_0_9,
        "volume": _concave_high_1_10,
        "delivery_window": _categorical({"fast": 9.0, "standard": 5.0, "slow": 0.0}),
        "quality_tier": _categorical({"basic": 0.0, "standard": 5.0, "premium": 9.0}),
    }
    if profile_id == PROFILE_M1:
        weights = {"unit_price": 2.0, "volume": 1.3, "delivery_window": 1.6, "quality_tier": 1.1}
        return base_values, weights
    if profile_id == PROFILE_M2:
        weights = {"unit_price": 1.2, "volume": 0.9, "delivery_window": 3.0, "quality_tier": 2.4}
        return base_values, weights
    if profile_id == PROFILE_M3:
        weights = {"unit_price": 3.2, "volume": 1.8, "delivery_window": 0.8, "quality_tier": 0.7}
        return base_values, weights
    raise ValueError(f"Unknown profile_id={profile_id!r}. Expected one of {sorted(VALID_MULTI_PROFILES)}")


def _supplier_templates(profile_id: str) -> list[tuple[dict[str, Callable[[Any], float]], dict[str, float]]]:
    if profile_id == PROFILE_M1:
        premium = (
            {
                "unit_price": _convex_high_0_9,
                "volume": _volume_peak(target=6.0, slope=1.2),
                "delivery_window": _categorical({"fast": 6.0, "standard": 9.0, "slow": 5.0}),
                "quality_tier": _categorical({"basic": 1.0, "standard": 6.0, "premium": 9.0}),
            },
            {"unit_price": 2.5, "volume": 1.0, "delivery_window": 1.4, "quality_tier": 2.0},
        )
        economy = (
            {
                "unit_price": _concave_high_0_9,
                "volume": _concave_high_1_10,
                "delivery_window": _categorical({"fast": 1.0, "standard": 6.0, "slow": 9.0}),
                "quality_tier": _categorical({"basic": 9.0, "standard": 5.0, "premium": 1.0}),
            },
            {"unit_price": 2.1, "volume": 2.0, "delivery_window": 1.5, "quality_tier": 1.0},
        )
        return [premium, economy]

    if profile_id == PROFILE_M2:
        fast_specialist = (
            {
                "unit_price": _convex_high_0_9,
                "volume": _volume_peak(target=5.0, slope=1.4),
                "delivery_window": _categorical({"fast": 9.0, "standard": 6.0, "slow": 1.0}),
                "quality_tier": _categorical({"basic": 0.0, "standard": 6.0, "premium": 9.0}),
            },
            {"unit_price": 2.3, "volume": 0.9, "delivery_window": 2.8, "quality_tier": 2.1},
        )
        low_cost = (
            {
                "unit_price": _linear_high_0_9,
                "volume": _concave_high_1_10,
                "delivery_window": _categorical({"fast": 0.0, "standard": 5.0, "slow": 9.0}),
                "quality_tier": _categorical({"basic": 9.0, "standard": 6.0, "premium": 2.0}),
            },
            {"unit_price": 2.8, "volume": 1.9, "delivery_window": 1.3, "quality_tier": 0.9},
        )
        return [fast_specialist, low_cost]

    if profile_id == PROFILE_M3:
        constrained = (
            {
                "unit_price": _convex_high_0_9,
                "volume": _volume_peak(target=4.0, slope=2.1),
                "delivery_window": _categorical({"fast": 2.0, "standard": 5.0, "slow": 9.0}),
                "quality_tier": _categorical({"basic": 8.0, "standard": 5.0, "premium": 2.0}),
            },
            {"unit_price": 3.0, "volume": 2.8, "delivery_window": 1.6, "quality_tier": 0.9},
        )
        scale = (
            {
                "unit_price": _concave_high_0_9,
                "volume": _concave_high_1_10,
                "delivery_window": _categorical({"fast": 1.0, "standard": 6.0, "slow": 9.0}),
                "quality_tier": _categorical({"basic": 9.0, "standard": 5.0, "premium": 1.0}),
            },
            {"unit_price": 1.7, "volume": 3.1, "delivery_window": 1.1, "quality_tier": 0.7},
        )
        return [constrained, scale]

    raise ValueError(f"Unknown profile_id={profile_id!r}. Expected one of {sorted(VALID_MULTI_PROFILES)}")


def make_procurement_ufuns(
    profile_id: str,
    outcome_space,
    n_suppliers: int = 2,
    buyer_reserved_value: float = 0.35,
    supplier_reserved_value: float = 0.35,
) -> tuple[LUFun, tuple[LUFun, ...]]:
    """
    Returns (buyer_ufun, supplier_ufuns) for one-to-many procurement.
    """
    if n_suppliers < 2:
        raise ValueError(f"n_suppliers must be >= 2, got {n_suppliers}")

    buyer_values, buyer_weights = _buyer_profile(profile_id)
    supplier_templates = _supplier_templates(profile_id)

    buyer_ufun = LUFun(
        values=buyer_values,
        weights=buyer_weights,
        outcome_space=outcome_space,
        reserved_value=buyer_reserved_value,
    ).scale_max(1.0)

    supplier_ufuns: list[LUFun] = []
    for i in range(n_suppliers):
        values, weights = supplier_templates[i % len(supplier_templates)]
        rv = min(0.95, max(0.0, supplier_reserved_value + 0.03 * i))
        supplier_ufuns.append(
            LUFun(
                values=values,
                weights=weights,
                outcome_space=outcome_space,
                reserved_value=rv,
            ).scale_max(1.0)
        )

    return buyer_ufun, tuple(supplier_ufuns)

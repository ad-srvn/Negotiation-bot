from __future__ import annotations

import math
from typing import Any, Callable

from negmas import SAOMechanism, make_issue
from negmas.preferences import LinearAdditiveUtilityFunction as LUFun


PROFILE_A = "A"
PROFILE_B = "B"
PROFILE_C = "C"
PROFILE_D = "D"
PROFILE_E = "E"
PROFILE_F = "F"
VALID_PROFILES = {PROFILE_A, PROFILE_B, PROFILE_C, PROFILE_D, PROFILE_E, PROFILE_F}


def make_issues() -> list:
    """Creates the fixed mixed-issue buyer-seller procurement domain."""
    return [
        make_issue(name="price", values=10),  # 0..9
        make_issue(name="quantity", values=list(range(1, 11))),  # 1..10
        make_issue(name="delivery_time", values=10),  # 0..9
        make_issue(name="service_level", values=["basic", "standard", "premium"]),
    ]


def make_mechanism(
    n_steps: int,
    time_limit: float | None = None,
    pend: float = 0.0,
    pend_per_second: float = 0.0,
) -> SAOMechanism:
    kwargs = {"issues": make_issues(), "n_steps": n_steps}
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


def _linear_high_0_9(x: int) -> float:
    return 9.0 * _norm_0_9(x)


def _linear_low_0_9(x: int) -> float:
    return 9.0 * (1.0 - _norm_0_9(x))


def _linear_high_1_10(x: int) -> float:
    return 9.0 * _norm_1_10(x)


def _linear_low_1_10(x: int) -> float:
    return 9.0 * (1.0 - _norm_1_10(x))


def _convex_high_0_9(x: int) -> float:
    return 9.0 * (_norm_0_9(x) ** 2)


def _convex_low_0_9(x: int) -> float:
    return 9.0 * ((1.0 - _norm_0_9(x)) ** 2)


def _convex_high_1_10(x: int) -> float:
    return 9.0 * (_norm_1_10(x) ** 2)


def _convex_low_1_10(x: int) -> float:
    return 9.0 * ((1.0 - _norm_1_10(x)) ** 2)


def _concave_high_0_9(x: int) -> float:
    return 9.0 * math.sqrt(_norm_0_9(x))


def _concave_low_0_9(x: int) -> float:
    return 9.0 * math.sqrt(1.0 - _norm_0_9(x))


def _concave_high_1_10(x: int) -> float:
    return 9.0 * math.sqrt(_norm_1_10(x))


def _concave_low_1_10(x: int) -> float:
    return 9.0 * math.sqrt(1.0 - _norm_1_10(x))


def _categorical(scores: dict[str, float]) -> Callable[[Any], float]:
    def value(option: Any) -> float:
        return float(scores.get(str(option), 0.0))

    return value


def _profile_spec(
    profile_id: str,
) -> tuple[dict[str, Callable[[Any], float]], dict[str, float], dict[str, Callable[[Any], float]], dict[str, float]]:
    if profile_id == PROFILE_A:
        buyer_values = {
            "price": _linear_low_0_9,
            "quantity": _linear_high_1_10,
            "delivery_time": _linear_low_0_9,
            "service_level": _categorical({"basic": 0.0, "standard": 5.0, "premium": 9.0}),
        }
        seller_values = {
            "price": _linear_high_0_9,
            "quantity": _linear_low_1_10,
            "delivery_time": _linear_high_0_9,
            "service_level": _categorical({"basic": 9.0, "standard": 4.0, "premium": 0.0}),
        }
        buyer_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 1.0, "service_level": 0.8}
        seller_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 1.0, "service_level": 0.8}
        return buyer_values, buyer_weights, seller_values, seller_weights

    if profile_id == PROFILE_B:
        buyer_values = {
            "price": _linear_low_0_9,
            "quantity": _linear_high_1_10,
            "delivery_time": _linear_low_0_9,
            "service_level": _categorical({"basic": 0.0, "standard": 6.0, "premium": 9.0}),
        }
        seller_values = {
            "price": _linear_high_0_9,
            "quantity": _linear_low_1_10,
            "delivery_time": _concave_high_0_9,
            "service_level": _categorical({"basic": 9.0, "standard": 4.0, "premium": 1.0}),
        }
        buyer_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 1.2, "service_level": 1.0}
        seller_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 3.5, "service_level": 1.5}
        return buyer_values, buyer_weights, seller_values, seller_weights

    if profile_id == PROFILE_C:
        buyer_values = {
            "price": _convex_low_0_9,
            "quantity": _concave_high_1_10,
            "delivery_time": _linear_low_0_9,
            "service_level": _categorical({"basic": 1.0, "standard": 6.0, "premium": 9.0}),
        }
        seller_values = {
            "price": _concave_high_0_9,
            "quantity": _convex_low_1_10,
            "delivery_time": _linear_high_0_9,
            "service_level": _categorical({"basic": 8.0, "standard": 5.0, "premium": 1.0}),
        }
        buyer_weights = {"price": 3.0, "quantity": 1.2, "delivery_time": 1.0, "service_level": 0.8}
        seller_weights = {"price": 1.8, "quantity": 3.0, "delivery_time": 1.0, "service_level": 0.7}
        return buyer_values, buyer_weights, seller_values, seller_weights

    if profile_id == PROFILE_D:
        buyer_values = {
            "price": _linear_low_0_9,
            "quantity": _linear_high_1_10,
            "delivery_time": _linear_low_0_9,
            "service_level": _categorical({"basic": 0.0, "standard": 2.0, "premium": 9.0}),
        }
        seller_values = {
            "price": _linear_high_0_9,
            "quantity": _linear_low_1_10,
            "delivery_time": _linear_high_0_9,
            "service_level": _categorical({"basic": 9.0, "standard": 2.0, "premium": 0.0}),
        }
        buyer_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 0.8, "service_level": 4.0}
        seller_weights = {"price": 1.0, "quantity": 1.0, "delivery_time": 0.8, "service_level": 4.0}
        return buyer_values, buyer_weights, seller_values, seller_weights

    if profile_id == PROFILE_E:
        buyer_values = {
            "price": _linear_low_0_9,
            "quantity": _concave_high_1_10,
            "delivery_time": _convex_low_0_9,
            "service_level": _categorical({"basic": 0.0, "standard": 5.0, "premium": 8.0}),
        }
        seller_values = {
            "price": _convex_high_0_9,
            "quantity": _linear_low_1_10,
            "delivery_time": _convex_high_0_9,
            "service_level": _categorical({"basic": 8.0, "standard": 4.0, "premium": 1.0}),
        }
        buyer_weights = {"price": 1.5, "quantity": 1.0, "delivery_time": 3.5, "service_level": 1.0}
        seller_weights = {"price": 2.5, "quantity": 1.2, "delivery_time": 2.2, "service_level": 1.0}
        return buyer_values, buyer_weights, seller_values, seller_weights

    if profile_id == PROFILE_F:
        buyer_values = {
            "price": _linear_low_0_9,
            "quantity": _convex_high_1_10,
            "delivery_time": _concave_low_0_9,
            "service_level": _categorical({"basic": 4.0, "standard": 9.0, "premium": 6.0}),
        }
        seller_values = {
            "price": _convex_high_0_9,
            "quantity": _convex_low_1_10,
            "delivery_time": _linear_high_0_9,
            "service_level": _categorical({"basic": 9.0, "standard": 5.0, "premium": 2.0}),
        }
        buyer_weights = {"price": 1.2, "quantity": 3.8, "delivery_time": 1.0, "service_level": 1.6}
        seller_weights = {"price": 3.8, "quantity": 1.0, "delivery_time": 1.2, "service_level": 1.6}
        return buyer_values, buyer_weights, seller_values, seller_weights

    raise ValueError(f"Unknown profile_id={profile_id!r}. Expected one of {sorted(VALID_PROFILES)}")


def make_ufuns(
    profile_id: str,
    outcome_space,
    buyer_reserved_value: float = 0.0,
    seller_reserved_value: float = 0.0,
) -> tuple[LUFun, LUFun]:
    """Returns (buyer_ufun, seller_ufun) for the selected profile."""
    buyer_values, buyer_weights, seller_values, seller_weights = _profile_spec(profile_id)
    buyer_ufun = LUFun(
        values=buyer_values,
        weights=buyer_weights,
        outcome_space=outcome_space,
        reserved_value=buyer_reserved_value,
    ).scale_max(1.0)
    seller_ufun = LUFun(
        values=seller_values,
        weights=seller_weights,
        outcome_space=outcome_space,
        reserved_value=seller_reserved_value,
    ).scale_max(1.0)
    return buyer_ufun, seller_ufun

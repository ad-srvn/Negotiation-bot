from __future__ import annotations

from collections import Counter
from random import choice
from typing import Any

from negmas import PolyAspiration, PresortingInverseUtilityFunction
from negmas.sao import ResponseType, SAONegotiator
from negmas.sao.negotiators import (
    AspirationNegotiator,
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
)


class SmartAspirationNegotiator(SAONegotiator):
    """
    Time-based negotiator with opponent-aware offering.

    It starts with a Boulware-style aspiration and can switch to a faster
    concession mode near deadline (typically Linear). It also tracks all
    opponent offers to estimate which issues are more important to the partner,
    then biases proposals toward those inferred preferences.
    """

    def __init__(
        self,
        *args,
        aspiration_type: str = "boulware",
        late_aspiration_type: str = "linear",
        deadline_switch_time: float = 0.8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._asp_early = PolyAspiration(1.0, aspiration_type)
        self._asp_late = PolyAspiration(1.0, late_aspiration_type)
        self._switch_time = min(1.0, max(0.0, float(deadline_switch_time)))
        self._inv = None
        self._partner_offers: list[tuple[Any, ...]] = []
        self._min = None
        self._max = None
        self._best = None

    def on_preferences_changed(self, changes):
        self._inv = PresortingInverseUtilityFunction(self.ufun)
        self._inv.init()
        worst, self._best = self.ufun.extreme_outcomes()
        self._min, self._max = self.ufun(worst), self.ufun(self._best)
        self._partner_offers = []
        super().on_preferences_changed(changes)

    def _aspiration_utility(self, relative_time: float | None) -> float:
        t = float(0.0 if relative_time is None else relative_time)
        t = min(1.0, max(0.0, t))
        if t <= self._switch_time or self._switch_time >= 1.0:
            return self._asp_early.utility_at(t)
        switch_utility = self._asp_early.utility_at(self._switch_time)
        late_t = (t - self._switch_time) / max(1e-9, 1.0 - self._switch_time)
        return switch_utility * self._asp_late.utility_at(late_t)

    def _aspiration_level(self, relative_time: float | None) -> float:
        if self._min is None or self._max is None:
            return 0.0
        return (self._max - self._min) * self._aspiration_utility(relative_time) + self._min

    def _estimate_partner_profile(self) -> tuple[list[Any], list[float], list[float | None]] | None:
        if not self._partner_offers:
            return None
        n_issues = len(self._partner_offers[0])
        targets: list[Any] = []
        importances: list[float] = []
        scales: list[float | None] = []
        for i in range(n_issues):
            values = [offer[i] for offer in self._partner_offers]
            numeric_vals: list[float] = []
            numeric = True
            for value in values:
                try:
                    numeric_vals.append(float(value))
                except (TypeError, ValueError):
                    numeric = False
                    break
            if numeric:
                span = max(numeric_vals) - min(numeric_vals)
                targets.append(sum(numeric_vals) / len(numeric_vals))
                importances.append(1.0 / (span + 1e-3))
                scales.append(max(span, 1.0))
                continue

            counts = Counter(values)
            targets.append(counts.most_common(1)[0][0])
            importances.append(1.0 / max(1, len(counts)))
            scales.append(None)

        total = sum(importances)
        if total > 0:
            importances = [w / total for w in importances]
        return targets, importances, scales

    def _distance_to_partner_profile(
        self, outcome: tuple[Any, ...], profile: tuple[list[Any], list[float], list[float | None]]
    ) -> float:
        targets, weights, scales = profile
        distance = 0.0
        for value, target, weight, scale in zip(outcome, targets, weights, scales):
            if scale is None:
                distance += weight * (0.0 if value == target else 1.0)
                continue
            try:
                distance += weight * abs(float(value) - float(target)) / scale
            except (TypeError, ValueError):
                distance += weight
        return distance

    def respond(self, state, source: str | None = None):
        offer = state.current_offer
        if offer is None:
            return ResponseType.REJECT_OFFER
        self._partner_offers.append(tuple(offer))
        if self._min is None or self._max is None:
            return super().respond(state, source)
        return (
            ResponseType.ACCEPT_OFFER
            if float(self.ufun(offer)) >= self._aspiration_level(state.relative_time) - 1e-6
            else ResponseType.REJECT_OFFER
        )

    def propose(self, state, dest: str | None = None):
        if self._inv is None or self._best is None:
            return self.nmi.random_outcomes(1)[0]

        aspiration_level = self._aspiration_level(state.relative_time)
        outcomes = list(self._inv.some((aspiration_level - 1e-6, self._max + 1e-6), False))

        if not outcomes:
            return self._best

        profile = self._estimate_partner_profile()
        if profile is None:
            return choice(outcomes)

        return min(outcomes, key=lambda o: self._distance_to_partner_profile(o, profile))


class GeniusBOABaselineNegotiator(AspirationNegotiator):
    """
    BOA/Genius-style baseline negotiator.

    Uses aspiration-based concession/acceptance behavior from NegMAS builtin
    AspirationNegotiator to provide a stronger external baseline.
    """

    def __init__(
        self,
        *args,
        max_aspiration: float = 1.0,
        aspiration_type: str | float = "boulware",
        stochastic: bool = True,
        presort: bool = True,
        tolerance: float = 0.002,
        **kwargs,
    ):
        super().__init__(
            *args,
            max_aspiration=max_aspiration,
            aspiration_type=aspiration_type,
            stochastic=stochastic,
            presort=presort,
            tolerance=tolerance,
            **kwargs,
        )

    def respond(self, state, source: str | None = None):
        return super().respond(state, source)

    def propose(self, state, dest: str | None = None):
        return super().propose(state, dest)


def strategy_registry() -> dict[str, type]:
    return {
        "BoulwareTBNegotiator": BoulwareTBNegotiator,
        "LinearTBNegotiator": LinearTBNegotiator,
        "ConcederTBNegotiator": ConcederTBNegotiator,
        "SmartAspirationNegotiator": SmartAspirationNegotiator,
        "GeniusBOABaselineNegotiator": GeniusBOABaselineNegotiator,
    }

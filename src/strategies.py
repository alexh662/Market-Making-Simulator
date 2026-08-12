"""Quoting strategies and the observation contract they see.

All strategies conform to the Strategy protocol from CLAUDE.md section 6.1.
MarketObservation is deliberately narrow: t, S, q, and the strategy's own
fill history. It must never carry a reference to the pre-generated future
price path, since that path is used only by the simulator's informed-flow
classification (section 2.7) and must stay invisible to quoting logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Protocol


class FillRecord(NamedTuple):
    t: float
    side: str
    price: float
    size: int
    mid_at_fill: float
    was_informed: bool
    mid_after_horizon: float


@dataclass(frozen=True)
class MarketObservation:
    t: float
    S: float
    q: float
    fills_so_far: tuple[FillRecord, ...]


class Strategy(Protocol):
    def quote(self, obs: MarketObservation) -> tuple[float, float]: ...


@dataclass(frozen=True)
class SymmetricFixedSpread:
    """Constant spread about the mid, no inventory response. The naive
    baseline strategy that isolates exactly what inventory skewing buys."""

    spread: float

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        half = self.spread / 2.0
        return obs.S - half, obs.S + half


@dataclass(frozen=True)
class AvellanedaStoikov:
    """The optimal quoting policy from CLAUDE.md sections 2.3-2.5. Quotes are
    placed symmetrically about a reservation price that shifts away from the
    mid as inventory builds, so that the skew itself falls out of the
    control solution rather than being a bolted-on heuristic."""

    gamma: float
    sigma: float
    kappa: float
    T: float

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        time_remaining = self.T - obs.t

        # reservation price: shifts below mid when long, above when short
        inventory_shift = obs.q * self.gamma * self.sigma**2 * time_remaining
        r = obs.S - inventory_shift

        # inventory risk term: widens with vol, risk aversion, time left
        inventory_term = self.gamma * self.sigma**2 * time_remaining
        # market microstructure term: governed by fill-intensity decay,
        # independent of inventory and time remaining
        microstructure_term = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        spread_total = inventory_term + microstructure_term

        return r - spread_total / 2.0, r + spread_total / 2.0

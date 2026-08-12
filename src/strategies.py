"""Quoting strategies and the observation contract they see.

All strategies conform to the Strategy protocol from CLAUDE.md section 6.1.
MarketObservation is deliberately narrow: t, S, q, and the strategy's own
fill history. It must never carry a reference to the pre-generated future
price path, since that path is used only by the simulator's informed-flow
classification (section 2.7) and must stay invisible to quoting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Protocol

import numpy as np


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
        # log1p(x), not log(1+x): at small gamma (e.g. 0.0005 in the section
        # 6.3 sweep), gamma/kappa is small enough that the naive 1+x loses
        # precision in the addition before the log is even taken, and the
        # error is then amplified by the 2/gamma factor. See DECISIONS.md.
        microstructure_term = (2.0 / self.gamma) * np.log1p(self.gamma / self.kappa)
        spread_total = inventory_term + microstructure_term

        return r - spread_total / 2.0, r + spread_total / 2.0


# Offset used by the two "with limit" strategies to withdraw a quote once
# q_max is exceeded. Large enough that A * exp(-kappa * offset) underflows to
# exactly 0.0 in float64 for any kappa configured in this project, so the
# withdrawn side's fill probability is exactly zero, not merely small.
NO_QUOTE_OFFSET = 1.0e4


@dataclass(frozen=True)
class FixedSpreadWithInventoryLimit:
    """As SymmetricFixedSpread, but stops quoting the side that would worsen
    inventory once |q| exceeds q_max. A crude but realistic control,
    contrasted against AS's continuous skew (section 6.1)."""

    spread: float
    q_max: float

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        half = self.spread / 2.0
        bid = obs.S - half
        ask = obs.S + half
        if obs.q > self.q_max:
            bid = obs.S - NO_QUOTE_OFFSET  # long past the cap: stop buying
        if obs.q < -self.q_max:
            ask = obs.S + NO_QUOTE_OFFSET  # short past the cap: stop selling
        return bid, ask


@dataclass(frozen=True)
class AvellanedaStoikovWithLimit:
    """AvellanedaStoikov plus a hard inventory cap: the side that would
    worsen inventory stops quoting once |q| exceeds q_max, layered on top of
    the continuous skew rather than replacing it."""

    gamma: float
    sigma: float
    kappa: float
    T: float
    q_max: float

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        inner = AvellanedaStoikov(gamma=self.gamma, sigma=self.sigma, kappa=self.kappa, T=self.T)
        bid, ask = inner.quote(obs)
        if obs.q > self.q_max:
            bid = obs.S - NO_QUOTE_OFFSET
        if obs.q < -self.q_max:
            ask = obs.S + NO_QUOTE_OFFSET
        return bid, ask


@dataclass(frozen=True)
class RandomQuoter:
    """Uniform random total spread in a plausible range, redrawn
    independently every step. A floor: if a strategy cannot beat this,
    something is broken (section 6.1). See DECISIONS.md for the range."""

    rng: np.random.Generator
    low: float = 0.7
    high: float = 2.0

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        half = self.rng.uniform(self.low, self.high) / 2.0
        return obs.S - half, obs.S + half

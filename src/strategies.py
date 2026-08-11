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

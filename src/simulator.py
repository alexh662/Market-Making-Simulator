"""The episode event loop, per CLAUDE.md section 4.

Everything downstream (decomposition, metrics, plots) reads from the two
DataFrames an episode produces here; nothing may recompute state by
re-running the simulation. That means the state record must carry enough
of the price path (including the terminal mid) for the PnL identity in
section 5.1 to be checked from the frames alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.price_process import generate_price_path
from src.strategies import NO_QUOTE_OFFSET, FillRecord, MarketObservation, Strategy

_FILL_COLUMNS = ["t", "side", "price", "size", "mid_at_fill", "was_informed", "mid_after_horizon"]


@dataclass(frozen=True)
class StateRecord:
    t: float
    mid: float
    inventory: float
    bid: float
    ask: float
    reservation_price: float
    spread_total: float
    cash: float
    mtm_pnl: float


@dataclass(frozen=True)
class EpisodeResult:
    states: pd.DataFrame
    fills: pd.DataFrame
    mark_to_market_pnl: float
    forced_liquidation_pnl: float
    rounding_collisions: int
    mid_crossing_count: int


def _round_quotes(raw_bid: float, raw_ask: float, tick_size: float) -> tuple[float, float, bool]:
    """Round bid down and ask up to the tick grid so rounding never tightens
    the spread. Widen by one tick each side if rounding collapses bid >= ask,
    per CLAUDE.md section 2.5."""
    eps = tick_size * 1e-6
    n_decimals = max(0, -int(round(math.log10(tick_size))))
    bid = round(math.floor(raw_bid / tick_size + eps) * tick_size, n_decimals)
    ask = round(math.ceil(raw_ask / tick_size - eps) * tick_size, n_decimals)
    collided = bid >= ask
    if collided:
        bid = round(bid - tick_size, n_decimals)
        ask = round(ask + tick_size, n_decimals)
    return bid, ask, collided


def _apply_fill(q: float, cash: float, side: str, price: float, size: int) -> tuple[float, float]:
    """Update inventory and cash for one fill. A bid fill is us buying (q up,
    cash down); an ask fill is us selling (q down, cash up)."""
    if side == "bid":
        return q + size, cash - price * size
    return q - size, cash + price * size


def run_episode(
    strategy: Strategy,
    *,
    S0: float,
    sigma: float,
    dt: float,
    T: float,
    A: float,
    kappa: float,
    order_size: int,
    phi_informed: float,
    informed_horizon: int,
    tick_size: float,
    adverse_selection_horizon: int,
    liquidation_cost_multiplier: float,
    rng: np.random.Generator,
) -> EpisodeResult:
    """Run one Monte Carlo episode: generate a price path, quote, fill, and
    record every state and fill.

    Informed flow (section 2.7) is implemented as a gate on ordinary
    arrivals: an arrival drawn on a side is reclassified as informed with
    probability phi_informed, and an informed arrival only executes if that
    side is the profitable one given the mid-price move over the next
    informed_horizon steps (a seller is correct if the price is about to
    fall, a buyer is correct if it is about to rise). An informed arrival on
    the wrong side simply does not execute, since an informed trader would
    not knowingly take an unprofitable trade. See DECISIONS.md.
    """
    n_steps = round(T / dt)
    path = generate_price_path(S0=S0, sigma=sigma, dt=dt, T=T, rng=rng)

    # pre-classify each potential arrival's informed status and fill draw up
    # front, in one fixed rng order, so runs are reproducible from the seed
    u_bid = rng.random(n_steps)
    u_ask = rng.random(n_steps)
    u_informed_bid = rng.random(n_steps)
    u_informed_ask = rng.random(n_steps)

    q = 0.0
    cash = 0.0
    fill_records: list[FillRecord] = []
    state_records: list[StateRecord] = []
    rounding_collisions = 0
    mid_crossing_count = 0

    for i in range(n_steps):
        t = i * dt
        S_i = path[i]
        obs = MarketObservation(t=t, S=S_i, q=q, fills_so_far=tuple(fill_records))
        raw_bid, raw_ask = strategy.quote(obs)
        reservation_price = (raw_bid + raw_ask) / 2.0
        # a large inventory skew can push the raw quote entirely to one side
        # of the mid; this is a real artefact of the model at extreme
        # inventory, so it is counted rather than silently allowed
        if raw_bid > S_i or raw_ask < S_i:
            mid_crossing_count += 1
        bid, ask, collided = _round_quotes(raw_bid, raw_ask, tick_size)
        rounding_collisions += int(collided)

        # distances measured from the mid, not the reservation price
        delta_b = S_i - bid
        delta_a = ask - S_i
        lambda_b = A * math.exp(-kappa * delta_b)
        lambda_a = A * math.exp(-kappa * delta_a)
        p_b = min(1.0, 1.0 - math.exp(-lambda_b * dt))
        p_a = min(1.0, 1.0 - math.exp(-lambda_a * dt))

        bid_fill = u_bid[i] < p_b
        ask_fill = u_ask[i] < p_a

        h = min(informed_horizon, n_steps - i)
        h_as = min(adverse_selection_horizon, n_steps - i)
        future_informed = path[i + h]
        mid_after_horizon = path[i + h_as]

        bid_informed = False
        if bid_fill:
            bid_informed = u_informed_bid[i] < phi_informed
            if bid_informed and not (future_informed < S_i):
                bid_fill = False

        ask_informed = False
        if ask_fill:
            ask_informed = u_informed_ask[i] < phi_informed
            if ask_informed and not (future_informed > S_i):
                ask_fill = False

        if bid_fill:
            q, cash = _apply_fill(q, cash, "bid", bid, order_size)
            fill_records.append(FillRecord(
                t=t, side="bid", price=bid, size=order_size,
                mid_at_fill=S_i, was_informed=bid_informed,
                mid_after_horizon=mid_after_horizon,
            ))
        if ask_fill:
            q, cash = _apply_fill(q, cash, "ask", ask, order_size)
            fill_records.append(FillRecord(
                t=t, side="ask", price=ask, size=order_size,
                mid_at_fill=S_i, was_informed=ask_informed,
                mid_after_horizon=mid_after_horizon,
            ))

        # a strategy withdraws a side by quoting it NO_QUOTE_OFFSET away from
        # the mid (see src/strategies.py); that is not a real two-sided
        # quote, so it must not count toward the average spread used for the
        # section 4.2 liquidation cost. Recorded as NaN, consistent with how
        # the terminal row already uses NaN for "no quote this row" --
        # pandas .mean() skips NaN, so downstream averages are unaffected.
        withdrawn = delta_b > NO_QUOTE_OFFSET / 2 or delta_a > NO_QUOTE_OFFSET / 2
        state_records.append(StateRecord(
            t=t, mid=S_i, inventory=q, bid=bid, ask=ask,
            reservation_price=reservation_price,
            spread_total=float("nan") if withdrawn else ask - bid,
            cash=cash, mtm_pnl=cash + q * S_i,
        ))

    # terminal row carries the final mid so the PnL identity can be checked
    # from the states frame alone, without re-running the simulation
    S_final = path[n_steps]
    mark_to_market_pnl = cash + q * S_final
    state_records.append(StateRecord(
        t=n_steps * dt, mid=S_final, inventory=q,
        bid=float("nan"), ask=float("nan"),
        reservation_price=float("nan"), spread_total=float("nan"),
        cash=cash, mtm_pnl=mark_to_market_pnl,
    ))

    states_df = pd.DataFrame(state_records)
    fills_df = pd.DataFrame(fill_records, columns=_FILL_COLUMNS)

    avg_spread = states_df["spread_total"].iloc[:-1].mean()
    avg_spread = 0.0 if pd.isna(avg_spread) else avg_spread
    liquidation_cost = liquidation_cost_multiplier * avg_spread
    forced_liquidation_pnl = mark_to_market_pnl - abs(q) * liquidation_cost

    return EpisodeResult(
        states=states_df,
        fills=fills_df,
        mark_to_market_pnl=mark_to_market_pnl,
        forced_liquidation_pnl=forced_liquidation_pnl,
        rounding_collisions=rounding_collisions,
        mid_crossing_count=mid_crossing_count,
    )


def pnl_identity_components(states: pd.DataFrame, fills: pd.DataFrame) -> tuple[float, float]:
    """Compute the two terms of the PnL identity (section 5.1) purely from
    the recorded frames: (A) spread capture, (B) inventory PnL using
    post-fill inventory at each step against the next step's mid."""
    if fills.empty:
        spread_capture = 0.0
    else:
        signed_size = np.where(fills["side"] == "bid", fills["size"], -fills["size"])
        spread_capture = float(np.sum((fills["mid_at_fill"] - fills["price"]) * signed_size))

    mids = states["mid"].to_numpy()
    post_fill_inventory = states["inventory"].to_numpy()[:-1]
    inventory_pnl = float(np.sum(post_fill_inventory * np.diff(mids)))

    return spread_capture, inventory_pnl

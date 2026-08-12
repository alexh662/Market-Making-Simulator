import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.simulator import _round_quotes, pnl_identity_components, run_episode
from src.strategies import AvellanedaStoikov, FillRecord, MarketObservation

_BASE_CONFIG = load_config(Path(__file__).resolve().parent.parent / "config" / "base.yaml")

DEFAULT_GAMMA = _BASE_CONFIG.strategy.gamma  # config/base.yaml strategy.gamma (CLAUDE.md section 12)
DEFAULT_SIGMA = 2.0
DEFAULT_KAPPA = 1.5
DEFAULT_T = 1.0
GAMMA_SWEEP = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # section 6.3

DEFAULT_SIM_PARAMS = dict(
    S0=100.0,
    sigma=DEFAULT_SIGMA,
    dt=0.005,
    T=DEFAULT_T,
    A=140.0,
    kappa=DEFAULT_KAPPA,
    order_size=1,
    phi_informed=0.15,
    informed_horizon=20,
    tick_size=0.01,
    adverse_selection_horizon=20,
    liquidation_cost_multiplier=0.5,
)
BASE_SEED = 20260811
N_PATHS = 100


def _make(gamma=DEFAULT_GAMMA, sigma=DEFAULT_SIGMA, kappa=DEFAULT_KAPPA, T=DEFAULT_T):
    return AvellanedaStoikov(gamma=gamma, sigma=sigma, kappa=kappa, T=T)


def _obs(t: float, S: float, q: float) -> MarketObservation:
    return MarketObservation(t=t, S=S, q=q, fills_so_far=())


def _reservation_price(strategy: AvellanedaStoikov, t: float, S: float, q: float) -> float:
    bid, ask = strategy.quote(_obs(t, S, q))
    return (bid + ask) / 2.0


def _spread(strategy: AvellanedaStoikov, t: float, S: float = 100.0, q: float = 0.0) -> float:
    bid, ask = strategy.quote(_obs(t, S, q))
    return ask - bid


def _strictly_increasing(values: list[float]) -> bool:
    return all(a < b for a, b in zip(values, values[1:]))


# --- reservation price (section 2.3) ---------------------------------------


def test_r_equals_S_exactly_when_q_zero():
    strategy = _make()
    assert _reservation_price(strategy, t=0.3, S=101.0, q=0.0) == 101.0


def test_r_below_S_when_long_above_S_when_short():
    strategy = _make()
    assert _reservation_price(strategy, t=0.3, S=101.0, q=5.0) < 101.0
    assert _reservation_price(strategy, t=0.3, S=101.0, q=-5.0) > 101.0


def test_abs_r_minus_S_increases_monotonically_in_abs_q():
    strategy = _make()
    diffs = [
        abs(_reservation_price(strategy, t=0.3, S=100.0, q=q) - 100.0)
        for q in [1.0, 2.0, 5.0, 10.0, 20.0]
    ]
    assert _strictly_increasing(diffs)


def test_abs_r_minus_S_increases_monotonically_in_gamma():
    diffs = [
        abs(_reservation_price(_make(gamma=g), t=0.3, S=100.0, q=5.0) - 100.0)
        for g in GAMMA_SWEEP
    ]
    assert _strictly_increasing(diffs)


def test_abs_r_minus_S_increases_monotonically_in_sigma():
    diffs = [
        abs(_reservation_price(_make(sigma=s), t=0.3, S=100.0, q=5.0) - 100.0)
        for s in [0.5, 1.0, 2.0, 4.0, 8.0]
    ]
    assert _strictly_increasing(diffs)


def test_abs_r_minus_S_increases_monotonically_in_time_remaining():
    strategy = _make()
    # descending t means ascending (T - t)
    diffs = [
        abs(_reservation_price(strategy, t=t, S=100.0, q=5.0) - 100.0)
        for t in [0.9, 0.7, 0.5, 0.3, 0.1]
    ]
    assert _strictly_increasing(diffs)


# --- total spread (section 2.4) ---------------------------------------------


def test_total_spread_at_horizon_equals_microstructure_term_only():
    strategy = _make()
    expected_microstructure_term = (2.0 / DEFAULT_GAMMA) * math.log(1.0 + DEFAULT_GAMMA / DEFAULT_KAPPA)
    assert _spread(strategy, t=DEFAULT_T) == pytest.approx(expected_microstructure_term, abs=1e-9)


def test_total_spread_increases_in_sigma():
    spreads = [_spread(_make(sigma=s), t=0.3) for s in [0.5, 1.0, 2.0, 4.0]]
    assert _strictly_increasing(spreads)


def test_total_spread_increases_in_time_remaining():
    spreads = [_spread(_make(), t=t) for t in [0.9, 0.7, 0.5, 0.3, 0.1]]
    assert _strictly_increasing(spreads)


def test_total_spread_increases_in_gamma_at_frozen_default_parameters():
    # NOT a universal property of the closed-form spread: the microstructure
    # term (2/gamma)*ln(1+gamma/kappa) decreases in gamma while the inventory
    # term gamma*sigma^2*(T-t) increases in gamma. This only holds because,
    # at the frozen defaults (sigma=2.0, kappa=1.5, T=1.0) evaluated at t=0,
    # the inventory term's increase dominates across the whole gamma sweep.
    # See DECISIONS.md for a counter-example near the episode horizon.
    spreads = [_spread(_make(gamma=g), t=0.0) for g in GAMMA_SWEEP]
    assert _strictly_increasing(spreads)


# --- tick rounding and mid-crossing diagnostics -----------------------------


def test_bid_below_ask_after_tick_rounding_across_inventory_range():
    strategy = _make()
    tick_size = 0.01
    for q in range(-100, 101):
        raw_bid, raw_ask = strategy.quote(_obs(t=0.3, S=100.0, q=float(q)))
        bid, ask, _ = _round_quotes(raw_bid, raw_ask, tick_size)
        assert bid < ask


def test_mid_crossing_is_counted_not_silently_allowed():
    # gamma=0.5, not the frozen default: at the frozen default (gamma=0.01)
    # a crossing requires roughly |q| >= 17 while T-t is still near 1, and a
    # 20000-seed search found zero paths that ever reach it (see
    # DECISIONS.md). This test exercises the counting mechanism itself, not
    # frozen-default behaviour, so it deliberately uses a gamma known to
    # produce crossings reliably; path 1 gives 13.
    strategy = _make(gamma=0.5)
    rng = np.random.default_rng((BASE_SEED, 1))
    result = run_episode(strategy, rng=rng, **DEFAULT_SIM_PARAMS)

    # recompute independently from the recorded states frame: the inventory
    # used to quote step i is the post-fill inventory recorded at step i-1
    # (0 for the first step)
    prior_q = result.states["inventory"].shift(1).fillna(0.0).to_numpy()[:-1]
    ts = result.states["t"].to_numpy()[:-1]
    mids = result.states["mid"].to_numpy()[:-1]

    expected_crossings = 0
    for t, S, q in zip(ts, mids, prior_q):
        raw_bid, raw_ask = strategy.quote(_obs(t=t, S=S, q=q))
        if raw_bid > S or raw_ask < S:
            expected_crossings += 1

    assert result.mid_crossing_count == expected_crossings
    assert expected_crossings > 0


# --- no lookahead (section 2.7 / section 8) ---------------------------------


class _RecordingStrategy:
    """Wraps a strategy to capture every MarketObservation it is handed, so
    the observation contract itself can be inspected after a run."""

    def __init__(self, inner):
        self._inner = inner
        self.observations: list[MarketObservation] = []

    def quote(self, obs: MarketObservation) -> tuple[float, float]:
        self.observations.append(obs)
        return self._inner.quote(obs)


def test_no_lookahead_in_market_observation():
    recorder = _RecordingStrategy(_make())
    rng = np.random.default_rng((BASE_SEED, 2))
    run_episode(recorder, rng=rng, **DEFAULT_SIM_PARAMS)

    n_steps = round(DEFAULT_SIM_PARAMS["T"] / DEFAULT_SIM_PARAMS["dt"])
    assert len(recorder.observations) == n_steps

    allowed_fields = {"t", "S", "q", "fills_so_far"}
    for step_index, obs in enumerate(recorder.observations):
        field_names = {f.name for f in dataclasses.fields(obs)}
        assert field_names == allowed_fields

        # t, S, q are plain scalars: structurally incapable of holding the
        # 201-element pre-generated path array
        assert np.ndim(obs.t) == 0
        assert np.ndim(obs.S) == 0
        assert np.ndim(obs.q) == 0

        assert isinstance(obs.fills_so_far, tuple)
        for fill in obs.fills_so_far:
            assert isinstance(fill, FillRecord)
        # at most two fills (bid + ask) per elapsed step, all strictly
        # before this one -- never as long as the full path
        assert len(obs.fills_so_far) <= 2 * step_index


# --- pnl identity regression with AvellanedaStoikov (section 5.1) ----------


def test_pnl_identity_reconciles_with_avellaneda_stoikov_across_100_paths():
    strategy = _make()
    residuals = []
    spread_samples = []

    for p in range(1, N_PATHS + 1):
        rng = np.random.default_rng((BASE_SEED, p))
        result = run_episode(strategy, rng=rng, **DEFAULT_SIM_PARAMS)
        spread_capture, inventory_pnl = pnl_identity_components(result.states, result.fills)
        residual = abs((spread_capture + inventory_pnl) - result.mark_to_market_pnl)
        residuals.append(residual)
        spread_samples.append(result.states["spread_total"].iloc[:-1])

    max_residual = max(residuals)
    print(
        f"\nmax abs reconciliation residual (AvellanedaStoikov, {N_PATHS} paths): "
        f"{max_residual:.3e}"
    )

    empirical_avg_spread = pd.concat(spread_samples).mean()
    theoretical_avg_spread = (
        DEFAULT_GAMMA * DEFAULT_SIGMA**2 * DEFAULT_T / 2.0
        + (2.0 / DEFAULT_GAMMA) * math.log(1.0 + DEFAULT_GAMMA / DEFAULT_KAPPA)
    )
    print(
        f"time-averaged total spread at defaults: "
        f"empirical (post-rounding, realised)={empirical_avg_spread:.4f} "
        f"theoretical (raw, closed-form)={theoretical_avg_spread:.4f}"
    )

    assert max_residual < 1e-9

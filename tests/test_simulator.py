import numpy as np
import pandas as pd
import pytest

from src.simulator import _apply_fill, pnl_identity_components, run_episode
from src.strategies import SymmetricFixedSpread

DEFAULT_PARAMS = dict(
    S0=100.0,
    sigma=2.0,
    dt=0.005,
    T=1.0,
    A=140.0,
    kappa=1.5,
    order_size=1,
    phi_informed=0.15,
    informed_horizon=20,
    tick_size=0.01,
    adverse_selection_horizon=20,
    liquidation_cost_multiplier=0.5,
)
BASE_SEED = 20260811
N_PATHS = 100


def _run_default_path(path_index: int, **overrides):
    params = {**DEFAULT_PARAMS, **overrides}
    rng = np.random.default_rng((BASE_SEED, path_index))
    strategy = SymmetricFixedSpread(spread=1.0)
    return run_episode(strategy, rng=rng, **params)


def test_pnl_identity_reconciles_on_100_paths_and_has_enough_fills():
    residuals = []
    fill_counts = []

    for p in range(1, N_PATHS + 1):
        result = _run_default_path(p)
        spread_capture, inventory_pnl = pnl_identity_components(result.states, result.fills)
        residual = abs((spread_capture + inventory_pnl) - result.mark_to_market_pnl)
        residuals.append(residual)
        fill_counts.append(len(result.fills))

    max_residual = max(residuals)
    fill_counts_arr = np.array(fill_counts)

    print(f"\nmax absolute reconciliation residual across {N_PATHS} paths: {max_residual:.3e}")
    print(
        "fill count distribution: "
        f"min={fill_counts_arr.min()} "
        f"p25={np.percentile(fill_counts_arr, 25):.1f} "
        f"median={np.median(fill_counts_arr):.1f} "
        f"p75={np.percentile(fill_counts_arr, 75):.1f} "
        f"max={fill_counts_arr.max()} "
        f"mean={fill_counts_arr.mean():.1f}"
    )

    assert max_residual < 1e-9
    assert fill_counts_arr.min() >= 50


def test_four_fill_sign_cases():
    # non-zero starting inventory/cash so the assertions exercise addition
    # against the prior state, not just the sign of a zero-based update
    q0, cash0 = 5.0, 200.0

    q, cash = _apply_fill(q0, cash0, "bid", price=99.5, size=3)
    assert q == q0 + 3
    assert cash == cash0 - 99.5 * 3

    q, cash = _apply_fill(q0, cash0, "ask", price=100.5, size=2)
    assert q == q0 - 2
    assert cash == cash0 + 100.5 * 2


def test_zero_vol_zero_informed_pnl_is_positive_and_equals_spread_capture():
    result = _run_default_path(1, sigma=0.0, phi_informed=0.0)

    assert len(result.fills) > 0
    spread_capture, inventory_pnl = pnl_identity_components(result.states, result.fills)

    assert result.mark_to_market_pnl > 0
    assert result.mark_to_market_pnl == pytest.approx(spread_capture, abs=1e-9)
    assert inventory_pnl == pytest.approx(0.0, abs=1e-9)

    signed_size = np.where(
        result.fills["side"] == "bid", result.fills["size"], -result.fills["size"]
    )
    adverse_selection = float(
        np.sum(signed_size * (result.fills["mid_after_horizon"] - result.fills["mid_at_fill"]))
    )
    assert adverse_selection == pytest.approx(0.0, abs=1e-9)


def test_zero_arrival_intensity_gives_no_fills_and_zero_pnl():
    result = _run_default_path(1, A=0.0)

    assert len(result.fills) == 0
    assert result.mark_to_market_pnl == 0.0
    assert (result.states["inventory"] == 0.0).all()


def test_same_seed_gives_bit_identical_frames():
    strategy = SymmetricFixedSpread(spread=1.0)
    rng1 = np.random.default_rng(4242)
    rng2 = np.random.default_rng(4242)

    result1 = run_episode(strategy, rng=rng1, **DEFAULT_PARAMS)
    result2 = run_episode(strategy, rng=rng2, **DEFAULT_PARAMS)

    pd.testing.assert_frame_equal(result1.states, result2.states)
    pd.testing.assert_frame_equal(result1.fills, result2.fills)
    assert result1.mark_to_market_pnl == result2.mark_to_market_pnl
    assert result1.forced_liquidation_pnl == result2.forced_liquidation_pnl

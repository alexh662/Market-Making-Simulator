import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.decomposition import (
    Decomposition,
    adverse_selection,
    count_clamped_fills,
    decompose,
    standard_error,
)
from src.simulator import pnl_identity_components, run_episode
from src.strategies import AvellanedaStoikov, SymmetricFixedSpread

_CFG = load_config("config/base.yaml")
N_PATHS = 500

_FILL_COLUMNS = ["t", "side", "price", "size", "mid_at_fill", "was_informed", "mid_after_horizon"]


def _sim_params(**overrides):
    params = dict(
        S0=_CFG.price.S0,
        sigma=_CFG.price.sigma,
        dt=_CFG.price.dt,
        T=_CFG.price.T,
        A=_CFG.flow.A,
        kappa=_CFG.flow.kappa,
        order_size=_CFG.flow.order_size,
        phi_informed=_CFG.flow.phi_informed,
        informed_horizon=_CFG.flow.informed_horizon,
        tick_size=_CFG.simulation.tick_size,
        adverse_selection_horizon=_CFG.simulation.adverse_selection_horizon,
        liquidation_cost_multiplier=_CFG.simulation.liquidation_cost_multiplier,
    )
    params.update(overrides)
    return params


def _strategies():
    return {
        "SymmetricFixedSpread": SymmetricFixedSpread(spread=_CFG.strategy.fixed_spread),
        "AvellanedaStoikov": AvellanedaStoikov(
            gamma=_CFG.strategy.gamma,
            sigma=_CFG.price.sigma,
            kappa=_CFG.flow.kappa,
            T=_CFG.price.T,
        ),
    }


def _run_paths(strategy, n_paths=N_PATHS, **overrides):
    params = _sim_params(**overrides)
    results = []
    for path in range(1, n_paths + 1):
        rng = np.random.default_rng((_CFG.simulation.base_seed, path))
        episode = run_episode(strategy, rng=rng, **params)
        results.append((episode, decompose(
            episode.states,
            episode.fills,
            adverse_selection_horizon=params["adverse_selection_horizon"],
        )))
    return results


@pytest.fixture(scope="module")
def default_runs():
    return {name: _run_paths(s) for name, s in _strategies().items()}


@pytest.fixture(scope="module")
def adverse_by_phi():
    out = {}
    for phi in (0.0, 0.4):
        for name, s in _strategies().items():
            runs = _run_paths(s, phi_informed=phi)
            out[(name, phi)] = np.array([d.adverse_selection for _, d in runs])
    return out


# --- sign convention -------------------------------------------------------


def test_hand_constructed_adverse_selection_pins_sign_convention():
    # Two fills chosen so neither the sizes (2 vs 3) nor the forward moves
    # (-1.0 vs +0.5) match, so the expected value cannot be reproduced by a
    # flipped sign convention or by swapping which side counts as the buy:
    #   buy  : +2 * (99.0  - 100.0) = -2.0
    #   sell : -3 * (100.5 - 100.0) = -1.5
    # Both terms are negative here, which is what adverse selection means:
    # we bought and it fell, we sold and it rose.
    fills = pd.DataFrame(
        [
            (0.00, "bid", 99.5, 2, 100.0, False, 99.0),
            (0.10, "ask", 100.5, 3, 100.0, False, 100.5),
        ],
        columns=_FILL_COLUMNS,
    )

    assert adverse_selection(fills) == pytest.approx(-3.5, abs=1e-12)

    # inverting the convention would give +3.5, and swapping the sides would
    # give a different magnitude split, so this literal pins both
    assert adverse_selection(fills) != pytest.approx(3.5, abs=1e-12)


# --- the identities --------------------------------------------------------


def test_spread_capture_plus_inventory_pnl_equals_total_pnl(default_runs):
    for name, runs in default_runs.items():
        residuals = [
            abs((d.spread_capture + d.inventory_pnl) - episode.mark_to_market_pnl)
            for episode, d in runs
        ]
        assert max(residuals) < 1e-9, f"{name} max residual {max(residuals):.3e}"


def test_spread_capture_plus_adverse_selection_plus_residual_equals_total(default_runs):
    for name, runs in default_runs.items():
        residuals = [
            abs((d.spread_capture + d.adverse_selection + d.residual) - d.total_pnl)
            for _, d in runs
        ]
        assert max(residuals) < 1e-9, f"{name} max residual {max(residuals):.3e}"


# --- degenerate and driven cases -------------------------------------------


def test_zero_volatility_gives_exactly_zero_adverse_selection():
    # Smoke test only: with sigma = 0 the mid never moves, so every forward
    # move is exactly 0 and the sum is 0 regardless of the sign convention.
    # It catches a wrong *column* (e.g. reading price instead of mid) but
    # says nothing about signs -- that is what the hand-constructed test above
    # is for.
    for strategy in _strategies().values():
        runs = _run_paths(strategy, n_paths=5, sigma=0.0, phi_informed=0.0)
        for _, d in runs:
            assert d.adverse_selection == 0.0


def test_adverse_selection_without_informed_flow_is_indistinguishable_from_zero(adverse_by_phi):
    # NOTE: this assertion deliberately differs from "strictly negative".
    #
    # With phi_informed = 0 there is no endogenous adverse selection in this
    # simulator, and the mean is zero rather than negative. The fill indicator
    # at step i is a function of the path up to i and of independent uniform
    # draws at i, while the forward move (mid_{t+h} - mid_t) is a sum of
    # strictly later, independent increments of a driftless walk. The two are
    # independent, so the expectation is exactly 0.
    #
    # Measured directly: pooled over ~400 episodes, the mean forward move is
    # +0.0022 (SE 0.0041) after buys and -0.0027 (SE 0.0041) after sells.
    # Across eight different base seeds the 500-path mean came out positive in
    # 4/8 (fixed spread) and 5/8 (AS) -- i.e. the sign is pure noise, so
    # asserting "strictly negative" would encode a seed-dependent accident.
    # See DECISIONS.md.
    for name in _strategies():
        adverse = adverse_by_phi[(name, 0.0)]
        t_stat = adverse.mean() / standard_error(adverse)
        assert abs(t_stat) < 3.0, f"{name} mean {adverse.mean():+.4f} is t={t_stat:+.2f} from zero"


def test_informed_flow_drives_adverse_selection_more_negative(adverse_by_phi):
    for name in _strategies():
        none, heavy = adverse_by_phi[(name, 0.0)], adverse_by_phi[(name, 0.4)]
        assert heavy.mean() < none.mean()
        # separation must be large relative to noise, not a coin flip
        gap_se = np.hypot(standard_error(heavy), standard_error(none))
        assert (none.mean() - heavy.mean()) / gap_se > 5.0


# --- clamping --------------------------------------------------------------


def test_clamped_fill_count_matches_fills_within_horizon_of_path_end():
    strategy = _strategies()["AvellanedaStoikov"]
    episode, d = _run_paths(strategy, n_paths=1)[0]

    n_steps = len(episode.states) - 1
    horizon = _CFG.simulation.adverse_selection_horizon
    step_index = np.rint(episode.fills["t"].to_numpy() / _CFG.price.dt).astype(int)
    expected = int(np.count_nonzero(step_index + horizon > n_steps))

    assert d.clamped_fill_count == expected
    # the frozen defaults give 200 steps and h=20, so clamping must engage
    assert expected > 0


def test_clamped_fills_use_final_mid_as_forward_mid():
    strategy = _strategies()["AvellanedaStoikov"]
    episode, _ = _run_paths(strategy, n_paths=1)[0]

    n_steps = len(episode.states) - 1
    horizon = _CFG.simulation.adverse_selection_horizon
    final_mid = episode.states["mid"].to_numpy()[-1]
    step_index = np.rint(episode.fills["t"].to_numpy() / _CFG.price.dt).astype(int)

    clamped = episode.fills.loc[step_index + horizon > n_steps]
    assert len(clamped) > 0
    assert np.allclose(clamped["mid_after_horizon"].to_numpy(), final_mid)

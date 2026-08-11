import pytest

from src.config import FlowConfig, PriceConfig, SimulationConfig, StrategyConfig


def test_sigma_le_zero_raises():
    with pytest.raises(ValueError):
        PriceConfig(S0=100.0, sigma=0.0, dt=0.005, T=1.0)


def test_dt_le_zero_raises():
    with pytest.raises(ValueError):
        PriceConfig(S0=100.0, sigma=2.0, dt=0.0, T=1.0)


def test_kappa_le_zero_raises():
    with pytest.raises(ValueError):
        FlowConfig(A=140.0, kappa=0.0, order_size=1, phi_informed=0.15, informed_horizon=20)


def test_gamma_le_zero_raises():
    with pytest.raises(ValueError):
        StrategyConfig(gamma=0.0, q_max=25, fixed_spread=1.0)


def test_fixed_spread_le_zero_raises():
    with pytest.raises(ValueError):
        StrategyConfig(gamma=0.1, q_max=25, fixed_spread=0.0)


def test_tick_size_le_zero_raises():
    with pytest.raises(ValueError):
        SimulationConfig(
            n_paths=2000,
            base_seed=1,
            tick_size=0.0,
            adverse_selection_horizon=20,
            liquidation_cost_multiplier=0.5,
        )


def test_n_paths_lt_one_raises():
    with pytest.raises(ValueError):
        SimulationConfig(
            n_paths=0,
            base_seed=1,
            tick_size=0.01,
            adverse_selection_horizon=20,
            liquidation_cost_multiplier=0.5,
        )


@pytest.mark.parametrize("phi", [-0.1, 1.1])
def test_phi_informed_outside_unit_interval_raises(phi):
    with pytest.raises(ValueError):
        FlowConfig(A=140.0, kappa=1.5, order_size=1, phi_informed=phi, informed_horizon=20)


def test_T_over_dt_not_integer_raises():
    with pytest.raises(ValueError):
        PriceConfig(S0=100.0, sigma=2.0, dt=0.003, T=1.0)


def test_T_over_dt_integer_within_tolerance_does_not_raise():
    # 0.005 does not divide 1.0 exactly in floating point, but is within 1e-9
    PriceConfig(S0=100.0, sigma=2.0, dt=0.005, T=1.0)

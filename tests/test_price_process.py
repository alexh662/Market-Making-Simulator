import numpy as np
import pytest

from src.price_process import generate_price_path


def test_same_seed_gives_identical_arrays():
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    path1 = generate_price_path(S0=100.0, sigma=2.0, dt=0.005, T=1.0, rng=rng1)
    path2 = generate_price_path(S0=100.0, sigma=2.0, dt=0.005, T=1.0, rng=rng2)
    np.testing.assert_array_equal(path1, path2)


def test_different_seeds_give_different_arrays():
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(2)
    path1 = generate_price_path(S0=100.0, sigma=2.0, dt=0.005, T=1.0, rng=rng1)
    path2 = generate_price_path(S0=100.0, sigma=2.0, dt=0.005, T=1.0, rng=rng2)
    assert not np.array_equal(path1, path2)


def test_path_length_equals_round_T_over_dt_plus_one():
    rng = np.random.default_rng(0)
    T, dt = 1.0, 0.005
    path = generate_price_path(S0=100.0, sigma=2.0, dt=dt, T=T, rng=rng)
    assert len(path) == round(T / dt) + 1


def test_first_element_equals_S0():
    rng = np.random.default_rng(0)
    S0 = 100.0
    path = generate_price_path(S0=S0, sigma=2.0, dt=0.005, T=1.0, rng=rng)
    assert path[0] == S0


def test_increment_std_matches_sigma_sqrt_dt():
    rng = np.random.default_rng(7)
    sigma, dt = 2.0, 0.005
    n_steps = 200_000
    T = n_steps * dt
    path = generate_price_path(S0=100.0, sigma=sigma, dt=dt, T=T, rng=rng)
    diffs = np.diff(path)
    expected_std = sigma * np.sqrt(dt)
    observed_std = diffs.std()
    assert observed_std == pytest.approx(expected_std, rel=0.05)

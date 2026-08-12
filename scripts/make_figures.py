"""Regenerate results/figures. Run with `uv run python scripts/make_figures.py`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.plotting import plot_quote_skew_example
from src.simulator import run_episode
from src.strategies import AvellanedaStoikov

QUOTE_SKEW_GAMMA = load_config(Path(__file__).resolve().parent.parent / "config" / "base.yaml").strategy.gamma
QUOTE_SKEW_SEED = 53


def make_quote_skew_example() -> None:
    strategy = AvellanedaStoikov(gamma=QUOTE_SKEW_GAMMA, sigma=2.0, kappa=1.5, T=1.0)
    rng = np.random.default_rng(QUOTE_SKEW_SEED)
    result = run_episode(
        strategy,
        rng=rng,
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
    plot_quote_skew_example(
        result.states,
        "results/figures/quote_skew_example.png",
        seed=QUOTE_SKEW_SEED,
        gamma=QUOTE_SKEW_GAMMA,
    )
    print(f"max |inventory| in episode: {result.states['inventory'].abs().max()}")
    print("wrote results/figures/quote_skew_example.png")


if __name__ == "__main__":
    make_quote_skew_example()

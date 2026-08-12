"""Regenerate results/figures. Run with `uv run python scripts/make_figures.py`."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.plotting import plot_quote_skew_example
from src.simulator import run_episode
from src.strategies import AvellanedaStoikov

# gamma=0.01 is the low end of the section 6.3 sweep, chosen (over the
# frozen default gamma=0.1) so inventory swings are large enough to make
# the reservation-price skew visually striking; see DECISIONS.md.
QUOTE_SKEW_GAMMA = 0.01
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

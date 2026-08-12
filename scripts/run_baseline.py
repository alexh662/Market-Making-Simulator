"""Print the section 5.3 PnL breakdown for both strategies at the frozen
defaults. Run with `uv run python scripts/run_baseline.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.decomposition import decompose, format_breakdown
from src.simulator import run_episode
from src.strategies import AvellanedaStoikov, SymmetricFixedSpread

N_PATHS = 500
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "base.yaml"


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    sim_params = dict(
        S0=cfg.price.S0,
        sigma=cfg.price.sigma,
        dt=cfg.price.dt,
        T=cfg.price.T,
        A=cfg.flow.A,
        kappa=cfg.flow.kappa,
        order_size=cfg.flow.order_size,
        phi_informed=cfg.flow.phi_informed,
        informed_horizon=cfg.flow.informed_horizon,
        tick_size=cfg.simulation.tick_size,
        adverse_selection_horizon=cfg.simulation.adverse_selection_horizon,
        liquidation_cost_multiplier=cfg.simulation.liquidation_cost_multiplier,
    )

    strategies = {
        "SymmetricFixedSpread": SymmetricFixedSpread(spread=cfg.strategy.fixed_spread),
        "AvellanedaStoikov": AvellanedaStoikov(
            gamma=cfg.strategy.gamma,
            sigma=cfg.price.sigma,
            kappa=cfg.flow.kappa,
            T=cfg.price.T,
        ),
    }

    as_spreads = []
    for name, strategy in strategies.items():
        decompositions = []
        for path in range(1, N_PATHS + 1):
            rng = np.random.default_rng((cfg.simulation.base_seed, path))
            episode = run_episode(strategy, rng=rng, **sim_params)
            decompositions.append(decompose(
                episode.states,
                episode.fills,
                adverse_selection_horizon=cfg.simulation.adverse_selection_horizon,
            ))
            if name == "AvellanedaStoikov":
                # exclude the terminal row, which carries NaN quotes
                as_spreads.append(episode.states["spread_total"].to_numpy()[:-1])
        print(format_breakdown(name, decompositions))
        print()

    # calibration target for the M5 fixed-spread baseline, per section 16:
    # setting the baseline's width to the time-averaged AS spread isolates
    # skewing rather than width in the headline comparison
    mean_spread = float(np.mean(np.concatenate(as_spreads)))
    print(f"time-averaged AvellanedaStoikov total spread over {N_PATHS} paths: {mean_spread:.4f}")
    print(f"(current strategy.fixed_spread in config: {cfg.strategy.fixed_spread})")


if __name__ == "__main__":
    main()

"""Print the section 5.3 PnL breakdown (M4), then the section 6.2 headline
comparison across all five strategies over 2000 paths (M5). Run with
`uv run python scripts/run_baseline.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import Config, load_config
from src.decomposition import Decomposition, decompose, format_breakdown, standard_error
from src.plotting import plot_inventory_paths, plot_pnl_distribution
from src.simulator import EpisodeResult, run_episode
from src.strategies import (
    AvellanedaStoikov,
    AvellanedaStoikovWithLimit,
    FixedSpreadWithInventoryLimit,
    RandomQuoter,
    SymmetricFixedSpread,
)

N_PATHS = 500
N_PATHS_HEADLINE = 2000  # section 6.2 / section 1's "2000 Monte Carlo paths"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "base.yaml"
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

IDENTITY_TOLERANCE = 1e-9
MIN_FILLS_REQUIRED = 20


def _sim_params(cfg: Config) -> dict:
    return dict(
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


def _strategy_factories(cfg: Config) -> dict:
    """One factory per strategy, keyed by name in section 6.1's order.

    Four strategies are stateless and reused across paths; RandomQuoter
    carries its own mutable RNG, so it gets a fresh instance per path,
    seeded deterministically from (base_seed, path, 1) -- a stream
    independent of the episode's own (base_seed, path) price-path RNG.
    """
    as_kwargs = dict(gamma=cfg.strategy.gamma, sigma=cfg.price.sigma, kappa=cfg.flow.kappa, T=cfg.price.T)
    fixed = SymmetricFixedSpread(spread=cfg.strategy.fixed_spread)
    fixed_limit = FixedSpreadWithInventoryLimit(spread=cfg.strategy.fixed_spread, q_max=cfg.strategy.q_max)
    av_stoikov = AvellanedaStoikov(**as_kwargs)
    av_stoikov_limit = AvellanedaStoikovWithLimit(**as_kwargs, q_max=cfg.strategy.q_max)

    return {
        "SymmetricFixedSpread": lambda path: fixed,
        "FixedSpreadWithInventoryLimit": lambda path: fixed_limit,
        "AvellanedaStoikov": lambda path: av_stoikov,
        "AvellanedaStoikovWithLimit": lambda path: av_stoikov_limit,
        "RandomQuoter": lambda path: RandomQuoter(rng=np.random.default_rng((cfg.simulation.base_seed, path, 1))),
    }


class _Accumulator:
    """Streams per-path results into the section 6.2 summary statistics
    without holding every path's full state frame in memory."""

    def __init__(self, q_max: float | None, keep_n_states: int):
        self.q_max = q_max
        self.keep_n_states = keep_n_states
        self.decompositions: list[Decomposition] = []
        self.kept_states: list[pd.DataFrame] = []
        self.sum_abs_q = 0.0
        self.count_abs_q = 0
        self.max_abs_q = 0.0
        self.sum_terminal_abs_q = 0.0
        self.sum_binding = 0
        self.count_binding = 0
        self.min_fill_count = None

    def add(self, episode: EpisodeResult, decomposition: Decomposition) -> None:
        self.decompositions.append(decomposition)
        n_fills = len(episode.fills)
        self.min_fill_count = n_fills if self.min_fill_count is None else min(self.min_fill_count, n_fills)

        trading_rows = episode.states.iloc[:-1]  # exclude the terminal NaN-quote row
        inv = trading_rows["inventory"].to_numpy()
        self.sum_abs_q += float(np.abs(inv).sum())
        self.count_abs_q += len(inv)
        self.max_abs_q = max(self.max_abs_q, float(np.abs(inv).max()))
        self.sum_terminal_abs_q += abs(float(episode.states["inventory"].iloc[-1]))

        if self.q_max is not None:
            # inventory used to quote step i is the post-fill inventory
            # recorded at step i-1 (0 for the first step), matching what
            # MarketObservation.q actually was at quote time
            prior_q = trading_rows["inventory"].shift(1).fillna(0.0).to_numpy()
            binding = (prior_q > self.q_max) | (prior_q < -self.q_max)
            self.sum_binding += int(binding.sum())
            self.count_binding += len(binding)

        if len(self.kept_states) < self.keep_n_states:
            self.kept_states.append(episode.states)

    @property
    def mean_abs_q(self) -> float:
        return self.sum_abs_q / self.count_abs_q

    @property
    def terminal_abs_q_mean(self) -> float:
        return self.sum_terminal_abs_q / len(self.decompositions)

    @property
    def binding_fraction(self) -> float:
        return self.sum_binding / self.count_binding


def run_headline_comparison(cfg: Config) -> None:
    """Section 6.2: all five strategies, 2000 paths, one CSV row each, plus
    the section 7 inventory-paths and PnL-distribution figures.

    Sharpe here is mean terminal PnL divided by the standard deviation of
    terminal PnL across paths, not annualised -- each of the 2000 paths is
    one independent trading session, not a point on a time series, so there
    is no per-period return to annualise. See DECISIONS.md.
    """
    sim_params = _sim_params(cfg)
    factories = _strategy_factories(cfg)
    limit_q_max = {
        "FixedSpreadWithInventoryLimit": cfg.strategy.q_max,
        "AvellanedaStoikovWithLimit": cfg.strategy.q_max,
    }
    keep_states_for = {"SymmetricFixedSpread", "AvellanedaStoikov"}

    accumulators: dict[str, _Accumulator] = {}
    for name, factory in factories.items():
        acc = _Accumulator(
            q_max=limit_q_max.get(name),
            keep_n_states=20 if name in keep_states_for else 0,
        )
        for path in range(1, N_PATHS_HEADLINE + 1):
            rng = np.random.default_rng((cfg.simulation.base_seed, path))
            strategy = factory(path)
            episode = run_episode(strategy, rng=rng, **sim_params)
            d = decompose(
                episode.states, episode.fills,
                adverse_selection_horizon=cfg.simulation.adverse_selection_horizon,
            )

            residual = abs((d.spread_capture + d.inventory_pnl) - episode.mark_to_market_pnl)
            assert residual < IDENTITY_TOLERANCE, (
                f"{name} path {path}: pnl identity residual {residual:.3e} >= {IDENTITY_TOLERANCE:.0e}"
            )
            assert len(episode.fills) >= MIN_FILLS_REQUIRED, (
                f"{name} path {path}: only {len(episode.fills)} fills, need >= {MIN_FILLS_REQUIRED}"
            )

            acc.add(episode, d)
        accumulators[name] = acc
        print(
            f"{name}: pnl identity reconciled to < {IDENTITY_TOLERANCE:.0e} "
            f"on all {N_PATHS_HEADLINE} paths, min fill count {acc.min_fill_count}"
        )
    print()

    rows = []
    n_steps = round(cfg.price.T / cfg.price.dt)
    for name, acc in accumulators.items():
        total = np.array([d.total_pnl for d in acc.decompositions])
        spread = np.array([d.spread_capture for d in acc.decompositions])
        adverse = np.array([d.adverse_selection for d in acc.decompositions])
        fills = np.array([d.fill_count for d in acc.decompositions], dtype=float)
        pnl_std = float(np.std(total, ddof=1))

        rows.append({
            "strategy": name,
            "total_pnl_mean": total.mean(),
            "total_pnl_se": standard_error(total),
            "pnl_std": pnl_std,
            "sharpe": total.mean() / pnl_std,
            "spread_capture_mean": spread.mean(),
            "adverse_selection_mean": adverse.mean(),
            "mean_abs_inventory": acc.mean_abs_q,
            "max_abs_inventory": acc.max_abs_q,
            "terminal_abs_inventory_mean": acc.terminal_abs_q_mean,
            "fill_count_mean": fills.mean(),
            "fill_rate": fills.mean() / n_steps,
        })

        print(format_breakdown(name, acc.decompositions))
        print()

    table = pd.DataFrame(rows)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLES_DIR / "headline_comparison.csv"
    with open(csv_path, "w") as f:
        # sharpe = mean(terminal PnL) / std(terminal PnL) across paths, not
        # annualised: each path is one independent session, not a point on
        # a time series (see DECISIONS.md, "Sharpe is mean over std of
        # terminal PnL, not annualised")
        f.write(
            f"# sharpe = mean(terminal pnl across {N_PATHS_HEADLINE} paths) / "
            "std(terminal pnl across paths), not annualised -- see DECISIONS.md\n"
        )
        table.to_csv(f, index=False)

    print(f"=== headline comparison ({N_PATHS_HEADLINE} paths, sharpe = mean / std of terminal PnL, not annualised) ===")
    print(table.to_string(index=False))
    print()

    # --- required extra reporting -------------------------------------
    as_total = np.array([d.total_pnl for d in accumulators["AvellanedaStoikov"].decompositions])
    fixed_total = np.array([d.total_pnl for d in accumulators["SymmetricFixedSpread"].decompositions])
    # paired: both strategies replay the same (base_seed, path) price path
    # and fill draws, so the per-path difference is the correct comparison,
    # not independent-sample SEs added in quadrature
    diff = as_total - fixed_total
    print(
        f"AvellanedaStoikov - SymmetricFixedSpread mean PnL diff (paired by path): "
        f"{diff.mean():+.4f}  (SE {standard_error(diff):.4f}, t={diff.mean()/standard_error(diff):+.2f})"
    )

    as_std = np.std(as_total, ddof=1)
    fixed_std = np.std(fixed_total, ddof=1)
    reduction_pct = 100.0 * (fixed_std - as_std) / fixed_std
    print(
        f"AvellanedaStoikov terminal PnL std reduction vs rewidened SymmetricFixedSpread: "
        f"{reduction_pct:+.2f}%  (AS std {as_std:.4f} vs fixed std {fixed_std:.4f})"
    )

    binding_fractions = {}
    for name in ("FixedSpreadWithInventoryLimit", "AvellanedaStoikovWithLimit"):
        frac = accumulators[name].binding_fraction
        binding_fractions[name] = frac
        print(f"{name}: inventory limit (q_max={cfg.strategy.q_max}) binding on {frac:.4%} of steps")
    print()

    # The paired comparison and the derived reduction are the project's
    # headline result, so they are written to a table rather than existing
    # only as stdout -- every published number must be readable from
    # results/tables/ without re-running the sweep.
    pd.DataFrame([{
        "n_paths": N_PATHS_HEADLINE,
        "paired_mean_pnl_diff_as_minus_fixed": diff.mean(),
        "paired_diff_se": standard_error(diff),
        "paired_diff_t_stat": diff.mean() / standard_error(diff),
        "as_pnl_std": as_std,
        "fixed_pnl_std": fixed_std,
        "pnl_std_reduction_pct": reduction_pct,
        "fixed_spread_limit_binding_fraction": binding_fractions["FixedSpreadWithInventoryLimit"],
        "as_limit_binding_fraction": binding_fractions["AvellanedaStoikovWithLimit"],
    }]).to_csv(TABLES_DIR / "as_vs_fixed_paired.csv", index=False)

    plot_inventory_paths(
        accumulators["SymmetricFixedSpread"].kept_states,
        accumulators["AvellanedaStoikov"].kept_states,
        str(FIGURES_DIR / "inventory_paths.png"),
    )
    plot_pnl_distribution(fixed_total, as_total, str(FIGURES_DIR / "pnl_distribution.png"))
    print("wrote results/figures/inventory_paths.png")
    print("wrote results/figures/pnl_distribution.png")


def run_m4_decomposition(cfg: Config) -> None:
    """M4: the section 5.3 breakdown for the two original strategies over
    500 paths. Kept as-is; naturally reflects whatever is in config now."""
    sim_params = _sim_params(cfg)

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
    print()


def main() -> None:
    cfg = load_config(CONFIG_PATH)
    run_m4_decomposition(cfg)
    run_headline_comparison(cfg)


if __name__ == "__main__":
    main()

"""Sweep orchestration and Monte Carlo aggregation, per CLAUDE.md section 6.

Every sweep reuses the same base_seed across every parameter value being
swept (the same (base_seed, path) scheme used everywhere else in this
project), so comparisons across parameter values are paired -- both
strategies or both parameter settings replay the same underlying price path
and fill draws at a given path index -- rather than contaminated by
independent seed noise. Sweeps override simulator/strategy parameters at
call time; nothing here ever writes to config/base.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import Config
from src.decomposition import Decomposition, decompose, standard_error
from src.simulator import run_episode
from src.strategies import AvellanedaStoikov, Strategy

IDENTITY_TOLERANCE = 1e-9
MIN_FILLS_REQUIRED = 20
N_PATHS = 2000


def base_sim_params(cfg: Config, **overrides) -> dict:
    """The simulator's parameter dict at the frozen defaults, with any
    sweep-specific overrides applied at call time."""
    params = dict(
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
    params.update(overrides)
    return params


@dataclass
class ConfigurationResult:
    """Aggregated 2000-path Monte Carlo result for one (strategy, params)
    point in a sweep."""

    label: str
    decompositions: list[Decomposition]
    min_fill_count: int
    below_fill_floor_count: int  # paths with fewer than MIN_FILLS_REQUIRED fills
    mean_abs_inventory: float
    max_abs_inventory: float
    mean_spread_total: float  # withdrawn-quote (NaN) steps excluded

    @property
    def total_pnl(self) -> np.ndarray:
        return np.array([d.total_pnl for d in self.decompositions])

    @property
    def spread_capture(self) -> np.ndarray:
        return np.array([d.spread_capture for d in self.decompositions])

    @property
    def adverse_selection(self) -> np.ndarray:
        return np.array([d.adverse_selection for d in self.decompositions])

    @property
    def fill_count(self) -> np.ndarray:
        return np.array([d.fill_count for d in self.decompositions], dtype=float)

    @property
    def pnl_mean(self) -> float:
        return float(self.total_pnl.mean())

    @property
    def pnl_se(self) -> float:
        return standard_error(self.total_pnl)

    @property
    def pnl_std(self) -> float:
        return float(np.std(self.total_pnl, ddof=1))

    @property
    def sharpe(self) -> float:
        return self.pnl_mean / self.pnl_std


def run_configuration(
    label: str,
    strategy_factory: Callable[[int], Strategy],
    params: dict,
    *,
    base_seed: int,
    n_paths: int = N_PATHS,
) -> ConfigurationResult:
    """Run n_paths Monte Carlo episodes for one configuration.

    The section 5.1 PnL identity is a hard, non-negotiable gate: it is an
    accounting identity that must hold regardless of parameters, so a
    violation is asserted immediately (a genuine bug, not a parameter
    effect) before the result is used for anything.

    The "at least 20 fills per path" floor is a diagnostic for whether a
    result is economically meaningful, not an accounting identity -- unlike
    the PnL identity, it CAN legitimately be violated by extreme corners of
    a deliberately wide parameter sweep (e.g. severe kappa misspecification
    makes the strategy quote far too wide for the true fill intensity, so
    realised fill counts drop; see DECISIONS.md). A hard abort there would
    make it impossible to report exactly the degenerate corners this
    project's sweeps are designed to surface, so violations are counted and
    surfaced in ConfigurationResult.below_fill_floor_count / min_fill_count
    instead of aborting the run.
    """
    decompositions: list[Decomposition] = []
    min_fill_count: int | None = None
    below_fill_floor_count = 0
    sum_abs_q = 0.0
    count_abs_q = 0
    max_abs_q = 0.0
    spread_samples: list[np.ndarray] = []

    for path in range(1, n_paths + 1):
        rng = np.random.default_rng((base_seed, path))
        strategy = strategy_factory(path)
        episode = run_episode(strategy, rng=rng, **params)
        d = decompose(
            episode.states, episode.fills,
            adverse_selection_horizon=params["adverse_selection_horizon"],
        )

        residual = abs((d.spread_capture + d.inventory_pnl) - episode.mark_to_market_pnl)
        assert residual < IDENTITY_TOLERANCE, (
            f"{label} path {path}: pnl identity residual {residual:.3e} >= {IDENTITY_TOLERANCE:.0e}"
        )
        n_fills = len(episode.fills)
        if n_fills < MIN_FILLS_REQUIRED:
            below_fill_floor_count += 1

        min_fill_count = n_fills if min_fill_count is None else min(min_fill_count, n_fills)
        decompositions.append(d)

        trading_rows = episode.states.iloc[:-1]
        inv = trading_rows["inventory"].to_numpy()
        sum_abs_q += float(np.abs(inv).sum())
        count_abs_q += len(inv)
        max_abs_q = max(max_abs_q, float(np.abs(inv).max()))

        spreads = trading_rows["spread_total"].to_numpy()
        spread_samples.append(spreads[~np.isnan(spreads)])

    pooled_spread = np.concatenate(spread_samples)
    return ConfigurationResult(
        label=label,
        decompositions=decompositions,
        min_fill_count=min_fill_count,
        below_fill_floor_count=below_fill_floor_count,
        mean_abs_inventory=sum_abs_q / count_abs_q,
        max_abs_inventory=max_abs_q,
        mean_spread_total=float(pooled_spread.mean()) if len(pooled_spread) else float("nan"),
    )


def bootstrap_sharpe_se(total_pnl: np.ndarray, n_boot: int = 2000, seed: int = 0) -> float:
    """Bootstrap standard error of the Sharpe ratio (mean/std of terminal
    PnL), used to test whether adjacent sweep points' Sharpe estimates are
    resolved from one another."""
    rng = np.random.default_rng(seed)
    n = len(total_pnl)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = total_pnl[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    boot_sharpes = means / stds
    return float(np.std(boot_sharpes, ddof=1))


# --- sweep 1: gamma, section 6.3 --------------------------------------------


def gamma_sweep(cfg: Config, gammas: list[float]) -> dict[float, ConfigurationResult]:
    params = base_sim_params(cfg)
    results = {}
    for gamma in gammas:
        strategy = AvellanedaStoikov(gamma=gamma, sigma=cfg.price.sigma, kappa=cfg.flow.kappa, T=cfg.price.T)
        results[gamma] = run_configuration(
            f"gamma={gamma}", lambda p, s=strategy: s, params, base_seed=cfg.simulation.base_seed,
        )
    return results


def gamma_sweep_table(results: dict[float, ConfigurationResult]) -> pd.DataFrame:
    rows = []
    for gamma, r in results.items():
        sharpe_se = bootstrap_sharpe_se(r.total_pnl)
        rows.append({
            "gamma": gamma,
            "pnl_mean": r.pnl_mean,
            "pnl_se": r.pnl_se,
            "pnl_std": r.pnl_std,
            "sharpe": r.sharpe,
            "sharpe_bootstrap_se": sharpe_se,
            "mean_fill_count": r.fill_count.mean(),
            "mean_abs_inventory": r.mean_abs_inventory,
            "max_abs_inventory": r.max_abs_inventory,
            "min_fill_count": r.min_fill_count,
            "below_fill_floor_count": r.below_fill_floor_count,
        })
    return pd.DataFrame(rows)


# --- sweep 2: informed flow, section 6.6 ------------------------------------


def informed_flow_sweep(cfg: Config, phis: list[float]) -> dict[float, ConfigurationResult]:
    strategy = AvellanedaStoikov(gamma=cfg.strategy.gamma, sigma=cfg.price.sigma, kappa=cfg.flow.kappa, T=cfg.price.T)
    results = {}
    for phi in phis:
        params = base_sim_params(cfg, phi_informed=phi)
        results[phi] = run_configuration(
            f"phi_informed={phi}", lambda p, s=strategy: s, params, base_seed=cfg.simulation.base_seed,
        )
    return results


def informed_flow_table(results: dict[float, ConfigurationResult]) -> pd.DataFrame:
    rows = []
    for phi, r in results.items():
        rows.append({
            "phi_informed": phi,
            "spread_capture_mean": r.spread_capture.mean(),
            "adverse_selection_mean": r.adverse_selection.mean(),
            "adverse_selection_pct_of_spread_capture": 100.0 * r.adverse_selection.mean() / r.spread_capture.mean(),
            "pnl_mean": r.pnl_mean,
            "pnl_se": r.pnl_se,
            "mean_fill_count": r.fill_count.mean(),
            "min_fill_count": r.min_fill_count,
            "below_fill_floor_count": r.below_fill_floor_count,
        })
    return pd.DataFrame(rows)


# --- sweep 3: volatility, section 6.5 ---------------------------------------


def volatility_sweep(cfg: Config, sigma_multipliers: list[float]) -> dict[tuple[float, str], ConfigurationResult]:
    """AvellanedaStoikov vs SymmetricFixedSpread at each sigma. The fixed
    baseline's spread stays at cfg.strategy.fixed_spread (the calibrated
    1.3590) at every sigma -- not recalibrated -- per CLAUDE.md's explicit
    instruction that this is the point of the experiment."""
    from src.strategies import SymmetricFixedSpread

    results = {}
    for mult in sigma_multipliers:
        sigma = cfg.price.sigma * mult
        params = base_sim_params(cfg, sigma=sigma)

        as_strategy = AvellanedaStoikov(gamma=cfg.strategy.gamma, sigma=sigma, kappa=cfg.flow.kappa, T=cfg.price.T)
        results[(mult, "AvellanedaStoikov")] = run_configuration(
            f"sigma_x{mult}/AS", lambda p, s=as_strategy: s, params, base_seed=cfg.simulation.base_seed,
        )

        fixed_strategy = SymmetricFixedSpread(spread=cfg.strategy.fixed_spread)
        results[(mult, "SymmetricFixedSpread")] = run_configuration(
            f"sigma_x{mult}/fixed", lambda p, s=fixed_strategy: s, params, base_seed=cfg.simulation.base_seed,
        )
    return results


def volatility_table(
    results: dict[tuple[float, str], ConfigurationResult], sigma_multipliers: list[float]
) -> pd.DataFrame:
    rows = []
    for mult in sigma_multipliers:
        as_r = results[(mult, "AvellanedaStoikov")]
        fixed_r = results[(mult, "SymmetricFixedSpread")]
        rows.append({
            "sigma_multiplier": mult,
            "as_mean_spread": as_r.mean_spread_total,
            "as_mean_abs_inventory": as_r.mean_abs_inventory,
            "as_max_abs_inventory": as_r.max_abs_inventory,
            "as_pnl_mean": as_r.pnl_mean,
            "as_pnl_std": as_r.pnl_std,
            "fixed_mean_abs_inventory": fixed_r.mean_abs_inventory,
            "fixed_max_abs_inventory": fixed_r.max_abs_inventory,
            "fixed_pnl_mean": fixed_r.pnl_mean,
            "fixed_pnl_std": fixed_r.pnl_std,
            "as_min_fill_count": as_r.min_fill_count,
            "fixed_min_fill_count": fixed_r.min_fill_count,
            "as_below_fill_floor_count": as_r.below_fill_floor_count,
            "fixed_below_fill_floor_count": fixed_r.below_fill_floor_count,
        })
    return pd.DataFrame(rows)


# --- sweep 4a: kappa sensitivity, section 6.4 -------------------------------


def kappa_sensitivity_sweep(cfg: Config, kappas: list[float]) -> dict[float, ConfigurationResult]:
    """True kappa varies; the strategy is always correctly informed of it."""
    results = {}
    for kappa in kappas:
        params = base_sim_params(cfg, kappa=kappa)
        strategy = AvellanedaStoikov(gamma=cfg.strategy.gamma, sigma=cfg.price.sigma, kappa=kappa, T=cfg.price.T)
        results[kappa] = run_configuration(
            f"true_kappa={kappa}", lambda p, s=strategy: s, params, base_seed=cfg.simulation.base_seed,
        )
    return results


def kappa_sensitivity_table(results: dict[float, ConfigurationResult]) -> pd.DataFrame:
    rows = []
    for kappa, r in results.items():
        rows.append({
            "true_kappa": kappa,
            "pnl_mean": r.pnl_mean,
            "pnl_se": r.pnl_se,
            "pnl_std": r.pnl_std,
            "sharpe": r.sharpe,
            "mean_fill_count": r.fill_count.mean(),
            "min_fill_count": r.min_fill_count,
            "below_fill_floor_count": r.below_fill_floor_count,
        })
    return pd.DataFrame(rows)


# --- sweep 4b: kappa misspecification, section 6.4 --------------------------


def kappa_misspecification_sweep(cfg: Config, true_kappa: float, ratios: list[float]) -> dict[float, ConfigurationResult]:
    """True kappa is held fixed at true_kappa (the simulator's fill
    intensity); the strategy's own belief about kappa is set independently
    to ratio * true_kappa. AvellanedaStoikov.kappa already governs only the
    strategy's spread formula and never the simulator's fill mechanics, so
    this needs no new parameter -- see DECISIONS.md."""
    params = base_sim_params(cfg, kappa=true_kappa)
    results = {}
    for ratio in ratios:
        assumed_kappa = ratio * true_kappa
        strategy = AvellanedaStoikov(
            gamma=cfg.strategy.gamma, sigma=cfg.price.sigma, kappa=assumed_kappa, T=cfg.price.T,
        )
        results[ratio] = run_configuration(
            f"assumed/true_kappa={ratio}", lambda p, s=strategy: s, params, base_seed=cfg.simulation.base_seed,
        )
    return results


def kappa_misspecification_table(results: dict[float, ConfigurationResult]) -> pd.DataFrame:
    rows = []
    for ratio, r in results.items():
        rows.append({
            "assumed_over_true_kappa": ratio,
            "pnl_mean": r.pnl_mean,
            "pnl_se": r.pnl_se,
            "pnl_std": r.pnl_std,
            "sharpe": r.sharpe,
            "mean_fill_count": r.fill_count.mean(),
            "min_fill_count": r.min_fill_count,
            "below_fill_floor_count": r.below_fill_floor_count,
        })
    return pd.DataFrame(rows)

"""Frozen configuration dataclasses and YAML loading/validation.

Config is loaded once at the start of a run and passed around as plain,
immutable data so that every module can rely on the same validated
parameter set rather than re-deriving or re-checking it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class PriceConfig:
    S0: float
    sigma: float
    dt: float
    T: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"price.sigma must be > 0, got {self.sigma}")
        if self.dt <= 0:
            raise ValueError(f"price.dt must be > 0, got {self.dt}")
        if self.T <= 0:
            raise ValueError(f"price.T must be > 0, got {self.T}")
        steps = self.T / self.dt
        if abs(steps - round(steps)) > 1e-9:
            raise ValueError(
                f"price.T / price.dt must be within 1e-9 of an integer, "
                f"got T={self.T}, dt={self.dt}, T/dt={steps}"
            )


@dataclass(frozen=True)
class FlowConfig:
    A: float
    kappa: float
    order_size: int
    phi_informed: float
    informed_horizon: int

    def __post_init__(self) -> None:
        if self.kappa <= 0:
            raise ValueError(f"flow.kappa must be > 0, got {self.kappa}")
        if not (0.0 <= self.phi_informed <= 1.0):
            raise ValueError(
                f"flow.phi_informed must be within [0, 1], got {self.phi_informed}"
            )


@dataclass(frozen=True)
class StrategyConfig:
    gamma: float
    q_max: int
    fixed_spread: float

    def __post_init__(self) -> None:
        if self.gamma <= 0:
            raise ValueError(f"strategy.gamma must be > 0, got {self.gamma}")
        if self.fixed_spread <= 0:
            raise ValueError(f"strategy.fixed_spread must be > 0, got {self.fixed_spread}")


@dataclass(frozen=True)
class SimulationConfig:
    n_paths: int
    base_seed: int
    tick_size: float
    adverse_selection_horizon: int
    liquidation_cost_multiplier: float

    def __post_init__(self) -> None:
        if self.tick_size <= 0:
            raise ValueError(f"simulation.tick_size must be > 0, got {self.tick_size}")
        if self.n_paths < 1:
            raise ValueError(f"simulation.n_paths must be >= 1, got {self.n_paths}")


@dataclass(frozen=True)
class Config:
    price: PriceConfig
    flow: FlowConfig
    strategy: StrategyConfig
    simulation: SimulationConfig


def load_config(path: str | Path) -> Config:
    """Read a YAML config file and return a validated, frozen Config.

    Validation is performed by each dataclass's __post_init__, so any
    invalid parameter raises ValueError before a simulation can start.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        price=PriceConfig(**raw["price"]),
        flow=FlowConfig(**raw["flow"]),
        strategy=StrategyConfig(**raw["strategy"]),
        simulation=SimulationConfig(**raw["simulation"]),
    )

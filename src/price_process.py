"""Mid-price path generation.

Arithmetic (not geometric) Brownian motion, no drift, per CLAUDE.md
section 2.2. Arithmetic is the Avellaneda-Stoikov convention and keeps
the closed-form quoting solution valid; see DECISIONS.md.
"""

from __future__ import annotations

import numpy as np


def generate_price_path(
    S0: float,
    sigma: float,
    dt: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate one arithmetic Brownian motion mid-price path.

    Takes an explicit numpy Generator (never the global RNG) so that
    callers control reproducibility deterministically per path.
    """
    n_steps = round(T / dt)
    increments = sigma * np.sqrt(dt) * rng.standard_normal(n_steps)
    path = np.empty(n_steps + 1, dtype=float)
    path[0] = S0
    np.cumsum(increments, out=increments)
    path[1:] = S0 + increments
    return path

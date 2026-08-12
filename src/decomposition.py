"""PnL attribution, per CLAUDE.md sections 5.2 and 5.3.

Reads only the StateRecord and FillRecord frames an episode produced; it
never re-runs the simulation (section 4.3). That constraint is why the
clamped-fill count is recomputed from the recorded fill times rather than
being returned by the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.simulator import pnl_identity_components


def signed_size(fills: pd.DataFrame) -> np.ndarray:
    """`+size` for a buy (bid) fill, `-size` for a sell (ask) fill.

    This one mapping fixes the sign convention for every attribution term,
    so it lives in a single place rather than being rewritten per formula.
    """
    return np.where(fills["side"] == "bid", fills["size"], -fills["size"]).astype(float)


def adverse_selection(fills: pd.DataFrame) -> float:
    """Section 5.2: `Σ_fills signed_size · (mid_{t+h} - mid_t)`.

    Negative when the counterparty was right — a buy followed by a falling
    mid, or a sell followed by a rising mid — so its magnitude is the price
    of trading against better-informed flow.

    `mid_after_horizon` already carries the end-of-path clamping the
    simulator applied (see DECISIONS.md): for a fill within `h` steps of the
    end, the forward mid is the final mid rather than the fill being dropped.
    """
    if fills.empty:
        return 0.0
    forward_move = fills["mid_after_horizon"].to_numpy() - fills["mid_at_fill"].to_numpy()
    return float(np.sum(signed_size(fills) * forward_move))


def count_clamped_fills(
    states: pd.DataFrame,
    fills: pd.DataFrame,
    adverse_selection_horizon: int,
) -> int:
    """Fills whose `h`-step-ahead mid ran past the end of the path and was
    therefore clamped to the final mid.

    Recovered from the recorded fill times because section 4.3 forbids
    re-running the simulation to recompute state.
    """
    if fills.empty:
        return 0
    n_steps = len(states) - 1
    dt = float(states["t"].iloc[1] - states["t"].iloc[0])
    step_index = np.rint(fills["t"].to_numpy() / dt).astype(int)
    return int(np.count_nonzero(step_index + adverse_selection_horizon > n_steps))


@dataclass(frozen=True)
class Decomposition:
    """One episode's section 5.3 breakdown.

    `spread_capture + adverse_selection + residual == total_pnl` by
    construction, since `residual` absorbs whatever inventory PnL the
    attribution horizon did not reach.
    """

    total_pnl: float
    spread_capture: float
    inventory_pnl: float
    adverse_selection: float
    residual: float
    fill_count: int
    clamped_fill_count: int


def decompose(
    states: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    adverse_selection_horizon: int,
) -> Decomposition:
    """Attribute one episode's mark-to-market PnL across sections 5.1-5.2."""
    spread_capture, inventory_pnl = pnl_identity_components(states, fills)
    adverse = adverse_selection(fills)
    return Decomposition(
        total_pnl=spread_capture + inventory_pnl,
        spread_capture=spread_capture,
        inventory_pnl=inventory_pnl,
        adverse_selection=adverse,
        # drift in inventory held beyond the attribution horizon; reported,
        # not hidden, because a dominant residual means h is mis-set
        residual=inventory_pnl - adverse,
        fill_count=len(fills),
        clamped_fill_count=count_clamped_fills(states, fills, adverse_selection_horizon),
    )


def standard_error(values: np.ndarray) -> float:
    """Standard error of the mean across Monte Carlo paths."""
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def format_breakdown(name: str, decompositions: list[Decomposition]) -> str:
    """Render the section 5.3 table with its reconciliation shown.

    The reconciliation is printed rather than merely asserted so a reader can
    see the identity close, which is the whole point of the section.
    """
    total = np.array([d.total_pnl for d in decompositions])
    spread = np.array([d.spread_capture for d in decompositions])
    adverse = np.array([d.adverse_selection for d in decompositions])
    residual = np.array([d.residual for d in decompositions])
    fills = np.array([d.fill_count for d in decompositions], dtype=float)
    clamped = np.array([d.clamped_fill_count for d in decompositions], dtype=float)

    reconciled = spread + adverse + residual
    max_residual = float(np.max(np.abs(reconciled - total)))
    adverse_pct = 100.0 * adverse.mean() / spread.mean() if spread.mean() != 0 else float("nan")

    lines = [
        f"=== {name} ({len(decompositions)} paths) ===",
        f"Total PnL          = {total.mean():+10.4f}  (SE {standard_error(total):.4f})",
        f"  Spread capture   = {spread.mean():+10.4f}  (SE {standard_error(spread):.4f})",
        f"  Adverse selection= {adverse.mean():+10.4f}  (SE {standard_error(adverse):.4f})",
        f"  Residual         = {residual.mean():+10.4f}  (SE {standard_error(residual):.4f})",
        f"reconciliation     : spread + adverse + residual = {reconciled.mean():+10.4f}"
        f"   max |diff| vs total across paths = {max_residual:.3e}",
        f"adverse selection as pct of spread capture: {adverse_pct:.2f}%",
        f"mean fill count    : {fills.mean():.2f}",
        f"mean clamped fills : {clamped.mean():.2f}",
    ]
    return "\n".join(lines)

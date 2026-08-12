"""Figure generation, per CLAUDE.md section 7. Every figure is 150 dpi PNG
with labelled axes and no seaborn default styling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_quote_skew_example(
    states: pd.DataFrame,
    save_path: str,
    *,
    seed: int,
    gamma: float,
) -> None:
    """Single-episode figure (section 7, item 2): mid, reservation price,
    bid, and ask on the upper axis; inventory on a lower axis sharing the
    time axis. Shows the quote centre pulling away from the mid as
    inventory builds. See DECISIONS.md for the seed and gamma choice."""
    fig, (ax_price, ax_inv) = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_price.plot(states["t"], states["mid"], label="mid", color="black", linewidth=1.2)
    ax_price.plot(
        states["t"], states["reservation_price"], label="reservation price",
        color="tab:purple", linewidth=1.2,
    )
    ax_price.plot(states["t"], states["bid"], label="bid", color="tab:blue", linewidth=0.9)
    ax_price.plot(states["t"], states["ask"], label="ask", color="tab:red", linewidth=0.9)
    ax_price.set_ylabel("price ($)")
    ax_price.legend(loc="upper left", frameon=False)
    ax_price.set_title(
        f"Avellaneda-Stoikov quote skew example (gamma={gamma}, seed={seed})"
    )

    ax_inv.plot(states["t"], states["inventory"], color="tab:green", linewidth=1.2)
    ax_inv.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
    ax_inv.set_ylabel("inventory (units)")
    ax_inv.set_xlabel("time")

    fig.text(
        0.5, -0.02,
        f"seed={seed}, gamma={gamma}, chosen to reach |inventory| >= 15 (see DECISIONS.md)",
        ha="center", va="top", fontsize=8, color="grey",
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

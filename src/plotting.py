"""Figure generation, per CLAUDE.md section 7. Every figure is 150 dpi PNG
with labelled axes and no seaborn default styling.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Consistent colour per strategy across every figure (section 7).
STRATEGY_COLORS = {
    "SymmetricFixedSpread": "tab:orange",
    "FixedSpreadWithInventoryLimit": "tab:red",
    "AvellanedaStoikov": "tab:blue",
    "AvellanedaStoikovWithLimit": "tab:cyan",
    "RandomQuoter": "tab:gray",
}


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


def plot_efficient_frontier(table: pd.DataFrame, save_path: str) -> None:
    """Section 7, item 3: mean PnL vs PnL standard deviation across the
    gamma sweep (section 6.3), each point annotated with gamma, plus Sharpe
    vs gamma to identify the peak. The project's signature figure."""
    fig, (ax_frontier, ax_sharpe) = plt.subplots(1, 2, figsize=(13, 5.5))

    ax_frontier.errorbar(
        table["pnl_mean"], table["pnl_std"],
        xerr=table["pnl_se"], fmt="o", color="tab:blue", capsize=3, markersize=6,
    )
    for _, row in table.iterrows():
        ax_frontier.annotate(
            f"γ={row['gamma']:g}", (row["pnl_mean"], row["pnl_std"]),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax_frontier.set_xlabel("mean terminal PnL ($)")
    ax_frontier.set_ylabel("terminal PnL standard deviation ($)")
    ax_frontier.set_title("Efficient frontier: risk vs return across gamma")

    ax_sharpe.errorbar(
        table["gamma"], table["sharpe"],
        yerr=table["sharpe_bootstrap_se"], fmt="o-", color="tab:blue", capsize=3, markersize=6,
    )
    ax_sharpe.set_xscale("log")
    ax_sharpe.set_xlabel("gamma (log scale)")
    ax_sharpe.set_ylabel("Sharpe (mean / std of terminal PnL)")
    ax_sharpe.set_title("Sharpe vs gamma")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_informed_flow_crossover(table: pd.DataFrame, save_path: str) -> None:
    """Section 7, item 5: spread capture and adverse selection (plotted as
    magnitude, so a crossover is visible as an intersection) against
    phi_informed (section 6.6)."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        table["phi_informed"], table["spread_capture_mean"],
        "o-", label="spread capture", color="tab:green",
    )
    ax.plot(
        table["phi_informed"], -table["adverse_selection_mean"],
        "o-", label="adverse selection (magnitude)", color="tab:red",
    )
    ax.set_xlabel("phi_informed")
    ax.set_ylabel("$ per episode")
    ax.set_title("Informed flow: spread capture vs adverse selection")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_kappa_misspecification(table: pd.DataFrame, save_path: str) -> None:
    """Section 7, item 7: Sharpe and mean PnL against the ratio of assumed
    to true kappa (section 6.4b), with the correctly specified point
    (ratio=1.0) marked."""
    fig, ax_sharpe = plt.subplots(figsize=(9, 5.5))
    ax_pnl = ax_sharpe.twinx()

    ax_sharpe.plot(
        table["assumed_over_true_kappa"], table["sharpe"],
        "o-", color="tab:blue", label="Sharpe",
    )
    ax_pnl.plot(
        table["assumed_over_true_kappa"], table["pnl_mean"],
        "s--", color="tab:orange", label="mean PnL",
    )

    correct = table[table["assumed_over_true_kappa"] == 1.0]
    if not correct.empty:
        ax_sharpe.axvline(1.0, color="grey", linewidth=0.8, linestyle=":")
        ax_sharpe.scatter(
            correct["assumed_over_true_kappa"], correct["sharpe"],
            color="tab:blue", s=140, zorder=5, edgecolors="black",
            label="correctly specified",
        )

    ax_sharpe.set_xlabel("assumed kappa / true kappa")
    ax_sharpe.set_ylabel("Sharpe", color="tab:blue")
    ax_pnl.set_ylabel("mean terminal PnL ($)", color="tab:orange")
    ax_sharpe.set_title("Kappa misspecification: Sharpe and mean PnL vs assumed/true ratio")

    lines1, labels1 = ax_sharpe.get_legend_handles_labels()
    lines2, labels2 = ax_pnl.get_legend_handles_labels()
    ax_sharpe.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_inventory_paths(
    fixed_states: list[pd.DataFrame],
    as_states: list[pd.DataFrame],
    save_path: str,
    *,
    n_paths: int = 20,
) -> None:
    """Section 7, item 1: inventory over time, AS vs fixed-spread, ~20
    sample paths each, side by side, same y-axis. The visual punchline: one
    is bounded, the other wanders."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, states_list, name in zip(
        axes, [fixed_states, as_states], ["SymmetricFixedSpread", "AvellanedaStoikov"]
    ):
        color = STRATEGY_COLORS[name]
        for states in states_list[:n_paths]:
            ax.plot(states["t"], states["inventory"], color=color, alpha=0.5, linewidth=0.8)
        ax.axhline(0.0, color="grey", linewidth=0.6, linestyle="--")
        ax.set_title(f"{name} ({min(n_paths, len(states_list))} sample paths)")
        ax.set_xlabel("time")

    axes[0].set_ylabel("inventory (units)")
    fig.suptitle("Inventory paths: fixed spread vs Avellaneda-Stoikov")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pnl_distribution(
    fixed_pnls: np.ndarray,
    as_pnls: np.ndarray,
    save_path: str,
) -> None:
    """Section 7, item 6: histograms of terminal PnL across paths, AS vs
    fixed-spread overlaid. The variance reduction should be visible at a
    glance."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    lo = min(fixed_pnls.min(), as_pnls.min())
    hi = max(fixed_pnls.max(), as_pnls.max())
    bins = np.linspace(lo, hi, 41)

    ax.hist(
        fixed_pnls, bins=bins, alpha=0.55, label="SymmetricFixedSpread",
        color=STRATEGY_COLORS["SymmetricFixedSpread"],
    )
    ax.hist(
        as_pnls, bins=bins, alpha=0.55, label="AvellanedaStoikov",
        color=STRATEGY_COLORS["AvellanedaStoikov"],
    )
    ax.axvline(fixed_pnls.mean(), color=STRATEGY_COLORS["SymmetricFixedSpread"], linewidth=1.2)
    ax.axvline(as_pnls.mean(), color=STRATEGY_COLORS["AvellanedaStoikov"], linewidth=1.2)

    ax.set_xlabel("terminal mark-to-market PnL ($)")
    ax.set_ylabel("path count")
    ax.set_title("Terminal PnL distribution: fixed spread vs Avellaneda-Stoikov")
    ax.legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

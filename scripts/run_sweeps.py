"""Run all four sweeps from CLAUDE.md sections 6.3-6.6. Run with
`uv run python scripts/run_sweeps.py`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, load_config
from src.experiments import (
    gamma_sweep,
    gamma_sweep_table,
    informed_flow_sweep,
    informed_flow_table,
    kappa_misspecification_sweep,
    kappa_misspecification_table,
    kappa_sensitivity_sweep,
    kappa_sensitivity_table,
    volatility_sweep,
    volatility_table,
)
from src.plotting import (
    plot_efficient_frontier,
    plot_informed_flow_crossover,
    plot_kappa_misspecification,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "base.yaml"
TABLES_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

GAMMA_VALUES = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]  # section 6.3
GAMMA_EXCLUDED = [0.5, 1.0, 5.0]  # diagnostic only, not on the published frontier
PHI_VALUES = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60, 0.80, 1.0]  # section 6.6, extended
SIGMA_MULTIPLIERS = [0.5, 1.0, 2.0, 4.0]  # section 6.5
KAPPA_SENSITIVITY_VALUES = [0.75, 1.0, 1.5, 2.25, 3.0]  # section 6.4a
KAPPA_MISSPEC_RATIOS = [0.5, 0.75, 1.0, 1.5, 2.0]  # section 6.4b, 1.0 = correctly specified


def _report_fill_floor_violations(table, param_col: str, floor_col: str, min_col: str) -> None:
    """Prints any configuration where fewer than MIN_FILLS_REQUIRED fills
    were observed on at least one path. The PnL identity is still verified
    (hard assert) for every such path; this is a visibility report, not a
    correctness failure -- see DECISIONS.md and src/experiments.py."""
    violations = table[table[floor_col] > 0]
    for _, row in violations.iterrows():
        print(
            f"  DEGENERATE CORNER: {param_col}={row[param_col]!s} -- "
            f"{int(row[floor_col])} of 2000 paths below the 20-fill floor "
            f"(min observed: {int(row[min_col])})"
        )


def run_sweep_1(cfg: Config) -> int:
    """Gamma sweep, section 6.3. Returns the minimum fill count observed."""
    print("=== sweep 1: gamma (section 6.3) ===")
    results = gamma_sweep(cfg, GAMMA_VALUES)
    table = gamma_sweep_table(results)
    min_fill = int(table["min_fill_count"].min())

    table.to_csv(TABLES_DIR / "gamma_sweep.csv", index=False)
    print(table.to_string(index=False))
    _report_fill_floor_violations(table, "gamma", "below_fill_floor_count", "min_fill_count")

    best_idx = int(table["sharpe"].idxmax())
    best_gamma = table.loc[best_idx, "gamma"]
    print(f"\nSharpe-maximising gamma: {best_gamma:g} (sharpe={table.loc[best_idx, 'sharpe']:.4f})")
    if best_idx in (0, len(table) - 1):
        print("  this is an ENDPOINT of the sweep range, not an interior maximum -- "
              "the true optimum may lie outside [0.0005, 0.1]")

    for i in range(len(table) - 1):
        s1, se1 = table.loc[i, "sharpe"], table.loc[i, "sharpe_bootstrap_se"]
        s2, se2 = table.loc[i + 1, "sharpe"], table.loc[i + 1, "sharpe_bootstrap_se"]
        if (s1 - se1 <= s2 + se2) and (s2 - se2 <= s1 + se1):
            g1, g2 = table.loc[i, "gamma"], table.loc[i + 1, "gamma"]
            print(f"  gamma={g1:g} and gamma={g2:g}: Sharpe standard errors overlap -- peak not resolved between them")

    plot_efficient_frontier(table, str(FIGURES_DIR / "efficient_frontier.png"))
    print("wrote results/figures/efficient_frontier.png")

    print("\n--- diagnostic only: gamma in [0.5, 1.0, 5.0], NOT on the published frontier (see DECISIONS.md) ---")
    diag_results = gamma_sweep(cfg, GAMMA_EXCLUDED)
    diag_table = gamma_sweep_table(diag_results)
    min_fill = min(min_fill, int(diag_table["min_fill_count"].min()))
    # min_fill_count / below_fill_floor_count are the whole point of this
    # table: they are the evidence that these points are degenerate rather
    # than merely low-Sharpe, so they must be readable from the CSV itself.
    diag_cols = ["gamma", "sharpe", "mean_fill_count", "min_fill_count", "below_fill_floor_count"]
    diag_table[diag_cols].to_csv(TABLES_DIR / "gamma_excluded_diagnostic.csv", index=False)
    print(diag_table[diag_cols].to_string(index=False))
    _report_fill_floor_violations(diag_table, "gamma", "below_fill_floor_count", "min_fill_count")
    print()
    return min_fill


def run_sweep_2(cfg: Config) -> int:
    """Informed flow sweep, section 6.6. Returns the minimum fill count observed."""
    print("=== sweep 2: informed flow (section 6.6) ===")
    results = informed_flow_sweep(cfg, PHI_VALUES)
    table = informed_flow_table(results)
    min_fill = int(table["min_fill_count"].min())

    table.to_csv(TABLES_DIR / "informed_flow_sweep.csv", index=False)
    print(table.to_string(index=False))
    _report_fill_floor_violations(table, "phi_informed", "below_fill_floor_count", "min_fill_count")

    plot_informed_flow_crossover(table, str(FIGURES_DIR / "informed_flow_crossover.png"))
    print("wrote results/figures/informed_flow_crossover.png")

    crosses = (table["spread_capture_mean"] <= -table["adverse_selection_mean"]).any()
    if not crosses:
        pct_at_1 = table.loc[table["phi_informed"] == 1.0, "adverse_selection_pct_of_spread_capture"].iloc[0]
        print(
            "\nNo crossover within phi_informed in [0, 1]: adverse selection never "
            "overtakes spread capture."
        )
        print(f"At phi_informed=1.0, adverse selection is {abs(pct_at_1):.2f}% of spread capture.")
        print(
            "Mechanism: an informed arrival trades the same size at the same rate as an "
            "uninformed one, so its edge per trade is bounded by the mid move over "
            "informed_horizon and cannot exceed the spread collected -- adverse selection "
            "can approach but not exceed spread capture under this fill model. Not "
            "extrapolated or curve-fit; see DECISIONS.md."
        )
    else:
        crossing_row = table[table["spread_capture_mean"] <= -table["adverse_selection_mean"]].iloc[0]
        print(f"\nCrossover found at or before phi_informed={crossing_row['phi_informed']}")
    print()
    return min_fill


def run_sweep_3(cfg: Config) -> int:
    """Volatility sweep, section 6.5. Returns the minimum fill count observed."""
    print("=== sweep 3: volatility (section 6.5) ===")
    results = volatility_sweep(cfg, SIGMA_MULTIPLIERS)
    table = volatility_table(results, SIGMA_MULTIPLIERS)
    min_fill = int(min(table["as_min_fill_count"].min(), table["fixed_min_fill_count"].min()))

    table.to_csv(TABLES_DIR / "volatility_sweep.csv", index=False)
    print(table.to_string(index=False))
    _report_fill_floor_violations(table, "sigma_multiplier", "as_below_fill_floor_count", "as_min_fill_count")
    _report_fill_floor_violations(table, "sigma_multiplier", "fixed_below_fill_floor_count", "fixed_min_fill_count")

    base_row = table[table["sigma_multiplier"] == 1.0].iloc[0]
    print(
        f"\nAS spread widening relative to 1x sigma (mean spread {base_row['as_mean_spread']:.4f}, "
        f"fixed baseline held constant at {cfg.strategy.fixed_spread:.4f}):"
    )
    for _, row in table.iterrows():
        widen_pct = 100.0 * (row["as_mean_spread"] - base_row["as_mean_spread"]) / base_row["as_mean_spread"]
        print(f"  sigma x{row['sigma_multiplier']:g}: AS mean spread = {row['as_mean_spread']:.4f} ({widen_pct:+.2f}%)")
    print()
    return min_fill


def run_sweep_4(cfg: Config) -> int:
    """Kappa sensitivity (6.4a) and misspecification (6.4b). Returns the
    minimum fill count observed across both."""
    print("=== sweep 4a: kappa sensitivity (section 6.4) ===")
    results_a = kappa_sensitivity_sweep(cfg, KAPPA_SENSITIVITY_VALUES)
    table_a = kappa_sensitivity_table(results_a)
    min_fill = int(table_a["min_fill_count"].min())
    table_a.to_csv(TABLES_DIR / "kappa_sensitivity.csv", index=False)
    print(table_a.to_string(index=False))
    _report_fill_floor_violations(table_a, "true_kappa", "below_fill_floor_count", "min_fill_count")
    print()

    print("=== sweep 4b: kappa misspecification (section 6.4) ===")
    results_b = kappa_misspecification_sweep(cfg, cfg.flow.kappa, KAPPA_MISSPEC_RATIOS)
    table_b = kappa_misspecification_table(results_b)
    min_fill = min(min_fill, int(table_b["min_fill_count"].min()))
    table_b.to_csv(TABLES_DIR / "kappa_misspecification.csv", index=False)
    print(table_b.to_string(index=False))
    _report_fill_floor_violations(table_b, "assumed_over_true_kappa", "below_fill_floor_count", "min_fill_count")

    plot_kappa_misspecification(table_b, str(FIGURES_DIR / "kappa_misspecification.png"))
    print("wrote results/figures/kappa_misspecification.png")

    correct_sharpe = table_b.loc[table_b["assumed_over_true_kappa"] == 1.0, "sharpe"].iloc[0]
    degraded_sharpe = table_b.loc[table_b["assumed_over_true_kappa"] == 2.0, "sharpe"].iloc[0]
    degradation_pct = 100.0 * (correct_sharpe - degraded_sharpe) / correct_sharpe
    print(
        f"\nSharpe degradation at 2x kappa error: {degradation_pct:+.2f}% "
        f"(correctly specified sharpe={correct_sharpe:.4f}, 2x error sharpe={degraded_sharpe:.4f})"
    )
    print()
    return min_fill


def main() -> None:
    t0 = time.time()
    cfg = load_config(CONFIG_PATH)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    min_fills = [
        run_sweep_1(cfg),
        run_sweep_2(cfg),
        run_sweep_3(cfg),
        run_sweep_4(cfg),
    ]

    elapsed = time.time() - t0
    print(f"total runtime: {elapsed:.1f}s")
    print(
        "minimum fill count observed across all configurations in all sweeps: "
        f"{min(min_fills)}"
    )
    if elapsed > 600:
        print("WARNING: runtime exceeded 10 minutes -- profile before optimising, log the bottleneck in DECISIONS.md")


if __name__ == "__main__":
    main()

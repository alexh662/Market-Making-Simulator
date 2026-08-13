# Inventory-Aware Market Making Simulator

## 1. Summary

This project implements the Avellaneda–Stoikov (AS) optimal market-making policy in a
simulated limit order market and measures what inventory-aware quoting actually buys you.
Against a fixed-spread baseline calibrated to quote at exactly the same average width
(1.3590), **AS reduced terminal PnL standard deviation by 31.03% (12.60 → 8.69) while
delivering mean PnL that is not statistically distinguishable from the baseline's**. The
comparison is paired — both strategies replay the same 2000 price paths and the same fill
draws — so the per-path difference is the correct test: **+0.2052 mean PnL in AS's favour,
standard error 0.1503, t = 1.36**, which is not significant. That is the result you want
from a risk-control policy: the same money, materially less variance in how you get it.

## 2. The problem

A market maker quotes a bid and an ask around a mid-price and earns the spread from
counterparties who trade for reasons unrelated to short-term price direction. That is a fee
business, and in expectation it is profitable. The difficulty is that fills arrive
stochastically and one-sided. A run of buyers leaves the maker short; a run of sellers
leaves it long. Inventory accumulates as a by-product of doing business, and once the maker
is holding a position, its PnL depends on where the price goes — it has been converted from
a market-neutral fee earner into a directional bet it never intended to take.

The second problem is that not all counterparties are uninformed. Some trade precisely
because they know something about the next price move. Against those, the maker
systematically buys just before the price falls and sells just before it rises. This is
adverse selection, and it is a direct subtraction from the spread revenue. A market maker's
economics are therefore the difference between two quantities: spread captured from
uninformed flow, and value surrendered to informed flow. This project measures both
separately rather than reporting only their sum.

## 3. Model

The mid-price follows driftless arithmetic Brownian motion, `S_{t+dt} = S_t + σ·√dt·Z`.
Arithmetic rather than geometric, because that is the convention under which the AS closed
form is derived.

**Reservation price** — the centre the maker quotes around:

```
r_t = S_t − q_t · γ · σ² · (T − t)
```

The quote centre sits *below* the mid when the maker is long and *above* it when short. The
shift grows with inventory `q`, with risk aversion `γ`, with volatility `σ`, and with time
remaining `(T − t)`. At `t → T` it vanishes, because there is no time left for inventory to
hurt you. This is the skew, and it is not a bolted-on heuristic — it falls out of the
optimal control solution.

**Optimal total spread** — how wide to quote:

```
spread_total = γ·σ²·(T − t)  +  (2/γ)·ln(1 + γ/κ)
```

Two terms with distinct jobs. The first is the **inventory risk term**: widen when
volatility is high and the session is long, because there is more time and more scope for
inventory to move against you. The second is the **market microstructure term**: it depends
only on `κ`, the rate at which fill probability decays as you quote further from the mid,
and is independent of both inventory and time. Quotes are placed symmetrically about `r_t`,
which makes them *asymmetric* about the mid — that asymmetry is the entire mechanism.

## 4. Simulator

Order arrivals are Poisson with intensity decaying exponentially in distance from the mid,
`λ(δ) = A·exp(−κ·δ)`, drawn independently on each side each step. Distances are measured
from the **mid**, not from the reservation price: a counterparty decides whether your quote
is attractive relative to the true market, and neither knows nor cares where your inventory
sits.

Informed flow is explicit. With probability `phi_informed`, an arrival is informed, and an
informed arrival only executes if it is on the side that turns out to be correct over the
next `informed_horizon` steps. This requires looking ahead at the pre-generated price path.
**That lookahead is confined entirely to the counterparty model.** The strategy receives
only a `MarketObservation` carrying `t`, `S_t`, `q_t` and its own fill history, and a test
asserts structurally that no future prices are reachable from it.

PnL decomposes exactly:

```
PnL_total = Σ_fills (mid_at_fill − execution_price)·signed_size   (A) spread capture
          + Σ_steps  q_i·(S_{i+1} − S_i)                          (B) inventory PnL
```

This identity is verified to within 1e-9 on every path of every configuration in every
sweep — roughly 150,000 episodes — and it is a hard assert that runs before any number is
written. Term (B) is split further into adverse selection (signed size times the mid move
over the attribution horizon) and a reported residual.

**What is not realistic** is set out in section 6. The most important item is that quotes
are recentred on the mid every single step, with no latency and no queue.

## 5. Results

All figures use 2000 Monte Carlo paths per configuration, with the same base seed across
every parameter value so that comparisons are paired rather than contaminated by seed noise.
Sharpe here means mean terminal PnL divided by the standard deviation of terminal PnL across
paths, not annualised: each path is one independent session, not a point on a time series.

### Headline comparison (`results/tables/headline_comparison.csv`)

| strategy | mean PnL | SE | PnL std | Sharpe | spread capture | adverse selection | mean \|q\| | max \|q\| | fills |
|---|---|---|---|---|---|---|---|---|---|
| SymmetricFixedSpread | 52.61 | 0.28 | 12.60 | 4.18 | 56.06 | −3.29 | 4.41 | 30 | 81.9 |
| FixedSpreadWithInventoryLimit | 52.08 | 0.26 | 11.62 | 4.48 | 55.20 | −3.08 | 4.03 | 11 | 80.7 |
| **AvellanedaStoikov** | **52.82** | **0.19** | **8.69** | **6.08** | 55.79 | −2.99 | 2.76 | 22 | 82.7 |
| AvellanedaStoikovWithLimit | 52.72 | 0.19 | 8.71 | 6.05 | 55.66 | −2.96 | 2.73 | 11 | 82.5 |
| RandomQuoter | 49.97 | 0.29 | 12.81 | 3.90 | 53.56 | −3.40 | 4.39 | 35 | 84.5 |

![inventory paths](results/figures/inventory_paths.png)

Inventory under the fixed-spread baseline wanders freely and reaches ±30 units, while AS
inventory is visibly pulled back toward zero — mean |q| of 2.76 against 4.41.

![efficient frontier](results/figures/efficient_frontier.png)

Across the swept γ range, higher risk aversion buys a large reduction in PnL standard
deviation (12.22 → 6.17) for almost no cost in mean PnL (52.53 → 51.17), so Sharpe rises
monotonically to 8.29 at γ = 0.1 and never turns within the plotted range.

![PnL distribution](results/figures/pnl_distribution.png)

The AS terminal PnL distribution is visibly tighter around the same centre — this is the
31.03% standard deviation reduction, seen directly.

![informed flow crossover](results/figures/informed_flow_crossover.png)

Spread capture falls and adverse selection rises as informed flow increases, but the two
lines do not meet anywhere in the admissible range: at `phi_informed = 1.0`, adverse
selection is 69.96% of spread capture, not 100%.

## 6. Walk-backs and limitations

**There is no endogenous adverse selection in this simulator.** All measured adverse
selection comes from the explicit informed-flow mechanism. The reason is structural: quotes
are recomputed and reposted every step, so there is never a stale resting quote to be picked
off. Analytically, the fill indicator at step `i` depends only on the path up to `i` and on
independent uniform draws at `i`, while the forward move is a sum of strictly later
increments of a driftless walk; the two are independent, so the expectation of
`signed_size × forward_move` is exactly zero. Empirically, with `phi_informed = 0` and
pooling roughly 22,000 fills per side, the mean forward mid move was +0.0022 after buys and
−0.0027 after sells, both with standard error 0.0041, and the sign of the 500-path mean
flipped positive in 4 of 8 base seeds. Adding quote latency — letting a posted quote persist
for more than one step before it can be replaced — is the single most valuable extension to
this model, because it is the thing that would make pick-off adverse selection real.

**There is no informed-flow crossover, and the expectation of one was not met.** The project
set out to find the `phi_informed` threshold beyond which adverse selection overwhelms
spread capture and market making becomes unprofitable. That threshold does not exist within
the admissible range. At `phi_informed = 1.0` — every single arrival informed — adverse
selection reached 69.96% of spread capture and never exceeded it. The mechanism is that an
informed arrival trades the same size at the same rate as an uninformed one; its per-trade
edge is bounded by the mid move over `informed_horizon` and cannot exceed the spread
collected on that trade. Under this fill model the business degrades severely (mean PnL
falls from 60.37 to 9.23, an 84.7% drop) but does not cross into structural
unprofitability from adverse selection alone. No curve was fitted and no crossover was
extrapolated past 1.0.

**The Sharpe-maximising γ is at the boundary of the swept range, not an interior peak.**
Sharpe rose monotonically across the whole sweep to 8.29 at γ = 0.1, the largest value
plotted. The turn was not observed within the plotted range. It does occur above it, in a
region excluded from the published figure because the points are degenerate rather than
merely low-Sharpe: at γ = 5.0, mean fill count collapses to 12.76 and **1921 of 2000 paths
fall below 20 fills**, with a minimum of 2 fills in a 200-step episode. A Sharpe computed
from 2 trades is not a market-making result. Those diagnostic points are in
`results/tables/gamma_excluded_diagnostic.csv` and deliberately not on the frontier plot.

**A sentinel value leaked into a published quantity and was caught by a range check.** The
two inventory-limit strategies withdraw a side by quoting it 1e4 away from the mid. That
sentinel was being recorded as a genuine quoted spread, so the average quoted spread for one
path came out as 751.32 instead of 1.3690, and the forced-liquidation cost derived from it
was two orders of magnitude too large — a forced-liquidation PnL of −707.25 against a
mark-to-market PnL of +44.07 on a terminal inventory of 2 units. It was caught before
reaching any published number, fixed at source by recording withdrawn steps as NaN, and a
regression test now asserts that no included spread exceeds 10 and that the mean sits within
10% of the configured width. The test was verified by reverting the fix and confirming it
fails (`assert 10000.69 < 10.0`).

**Simulated flow is not real flow.** Exponential arrival intensity in quote distance is a
convenient analytical assumption, not an empirical fact about any market.

**There is no queue position modelling.** Real fills depend on where you sit in the queue at
your price level, on how much size is ahead of you, and on cancellation dynamics. This
simulator ignores all of it: if your price is good, you fill with a probability that depends
only on distance.

**The informed trader model is a caricature.** Real informed flow is noisy, partially
informed, and informed over varying horizons. Here an informed trader is perfectly informed
over exactly `informed_horizon` steps and never wrong.

**The AS closed form assumes a terminal time and no drift.** Both are questionable for a
maker that operates continuously. The `(T − t)` factor means the policy deliberately stops
managing inventory as the session ends, which is correct for the model and wrong for a desk
that reopens tomorrow holding the same book.

**Spread monotonicity in γ holds at these parameters but is not a general property.** The
inventory term rises with γ while the microstructure term `(2/γ)ln(1 + γ/κ)` falls with it.
Which dominates depends on `σ`, `T` and `κ`. At the frozen defaults evaluated at `t = 0` the
inventory term wins across the whole sweep; near the horizon it does not. The test that
asserts monotonicity is scoped to the frozen defaults for exactly this reason.

## 7. Findings

1. **Inventory skewing cut terminal PnL standard deviation by 31.03% at no cost in mean
   PnL.** Against a fixed-spread baseline calibrated to the same 1.3590 average width, AS
   reduced PnL standard deviation from 12.60 to 8.69 and raised Sharpe from 4.18 to 6.08.
   The paired per-path mean PnL difference was +0.2052 with standard error 0.1503
   (t = 1.36, 2000 paths) — not statistically distinguishable from zero. AS is a
   risk-control policy, and it behaves like one.

2. **Under volatility stress the protection comes from skewing, not from quoting wider.**
   This is the most mechanistically informative result in the project, because it identifies
   which of the policy's two levers does the work. As σ went from 0.5× to 4× the default, AS
   widened its average spread only modestly — 1.3439 at 0.5× (slightly *narrower* than
   baseline), to 1.6605 at 4×, a rise of just 22.19%. Yet over the same range its PnL
   standard deviation grew from 7.21 to 11.41 while the fixed baseline's grew from 7.86 to
   **46.20**, a 4.05× gap at 4× σ. AS's max |q| actually *fell* as volatility rose (27 → 10)
   while the baseline's sat flat at 30 regardless. A 22% wider spread cannot explain a 4×
   difference in outcome variance. The reservation-price shift scales with σ², so as
   volatility rises the skew pulls inventory back toward zero far more aggressively — that,
   not width, is what contains the risk.

3. **Adverse selection consumed 5.36% of gross spread revenue at the frozen defaults, rising
   to 69.96% under fully informed flow.** At `phi_informed = 0.15`, AS captured 55.79 in
   spread and surrendered 2.99 to adverse selection. Sweeping informed flow to 1.0 raised
   that to 21.01 against 30.02 of spread capture, and cut mean PnL from 60.37 to 9.23.

4. **A 2× error in the assumed fill-intensity parameter κ cost 19.77% of Sharpe.** Feeding
   the strategy a κ estimate twice the true value dropped Sharpe from 6.08 to 4.88 and mean
   PnL from 52.82 to 38.89. Underestimating is worse: a 0.5× error cost 24.35% of Sharpe
   (6.08 → 4.60) and pushed mean fill count down to 33.14, because a strategy that believes
   fills decay slowly quotes too wide for the true intensity. Since κ is estimated from data
   with error in any real deployment, this is the practical cost of estimating it badly.

## 8. Reproducing

```bash
uv sync
uv run python scripts/make_figures.py     # quote_skew_example.png
uv run python scripts/run_baseline.py     # decomposition + headline comparison + 2 figures
uv run python scripts/run_sweeps.py       # 4 sweeps + 6 tables + 3 figures
uv run pytest -v                          # 44 tests
```

Everything is seeded from `simulation.base_seed` in `config/base.yaml`; runs are
bit-reproducible. The three scripts regenerate every table in `results/tables/` and every
figure in `results/figures/` from scratch, and are independent of the working directory they
are invoked from.

**Reproducibility was verified, not asserted.** `results/` was moved outside the repository,
the three scripts above were run end to end from a different working directory, and every
regenerated CSV was compared byte-for-byte against the original: **all seven pre-existing
tables came back byte-identical**, and all six figures regenerated. Total runtime for the
three scripts was **206 seconds** on one core (`run_sweeps.py` alone is ~180s), covering
roughly 150,000 simulated episodes.

Two genuine reproducibility bugs were found and fixed by this exercise: `make_figures.py`
wrote to a working-directory-relative path, and neither it nor `run_baseline.py` created
`results/figures/` when it was absent — so both would have failed on a clean checkout. Both
now resolve paths from the repository root and create their output directories.

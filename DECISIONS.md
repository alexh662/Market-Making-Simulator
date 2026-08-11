# Decisions

## 2026-08-11 Arithmetic vs geometric Brownian motion
**Decision:** Mid-price follows arithmetic Brownian motion with no drift, `S_{t+dt} = S_t + sigma * sqrt(dt) * Z`.
**Alternatives considered:** Geometric Brownian motion (log-normal price, proportional volatility).
**Reasoning:** CLAUDE.md section 2.2 mandates arithmetic BM as the Avellaneda-Stoikov convention; the closed-form quoting solution (reservation price, optimal spread) is derived under arithmetic BM and does not carry over cleanly to GBM. Arithmetic dynamics also keep inventory PnL linear in price increments, which is what the PnL decomposition in section 5 relies on.
**Consequence:** Prices can in principle go negative over long horizons or high sigma, but at the frozen defaults (S0=100, sigma=2.0, T=1.0) this is not a practical concern. No drift is added in the base case so PnL variance can be attributed to inventory risk rather than a directional bet; drift is reserved as an optional robustness check (section 6.7).

## 2026-08-11 Config validation rules
**Decision:** Config is represented as frozen dataclasses (`PriceConfig`, `FlowConfig`, `StrategyConfig`, `SimulationConfig`, `Config`) that validate their own fields in `__post_init__`, raising `ValueError` on: `sigma <= 0`, `dt <= 0`, `T <= 0`, `kappa <= 0`, `gamma <= 0`, `tick_size <= 0`, `n_paths < 1`, `phi_informed` outside `[0, 1]`, and `T / dt` not within `1e-9` of an integer.
**Alternatives considered:** A single flat config dataclass; validating with a schema library (e.g. `pydantic`); validating only at the point of use rather than at construction.
**Reasoning:** CLAUDE.md section 13 requires frozen dataclasses for config (no dictionaries passed as configuration) and forbids adding dependencies without logging them here — `pydantic` was not added for that reason. Validating in `__post_init__` means an invalid config fails immediately at construction, whether built from YAML or directly in tests, rather than surfacing as a confusing downstream numerical error (e.g. a `dt` that doesn't evenly divide `T` would silently produce a path one element short or long). The `T / dt` integer check uses a `1e-9` tolerance because floating-point division (e.g. `1.0 / 0.005`) is not exact.
**Consequence:** Each config section validates independently of the others, which is sufficient since none of the required rules are cross-section. If a future rule needs to compare fields across sections (e.g. price vs strategy), it will need to move into `Config.__post_init__` instead.

# CLAUDE.md — option-pricing-lab

## Project Identity

- **Repo name**: `option-pricing-lab`
- **Author**: Diogo Vilela (GitHub: diogovilela03)
- **Purpose**: Portfolio project demonstrating option pricing theory, numerical methods, vol surface calibration, and software engineering practice
- **Companion project**: `forward-curve-arbitrage` (same GitHub account, cross-linked on personal site as a distinct, complementary project — that project covers forward curve / Delta One mechanics; this project covers pricing engines, numerical methods, and vol surface calibration. Avoid overlapping content.)
- **Target audience**: Interviewers at Optiver, IMC, Flow Traders, QRT, Man AHL, Millennium, Robeco, Jane Street
- **Language**: Python 3.11+
- **OS**: Windows (ensure cross-platform path handling with `pathlib`)
- **Dashboard framework**: Streamlit (consistent with `forward-curve-arbitrage`)

---

## What This Project Does

A pricing and calibration toolkit covering three layers:

1. **Pricing engines** — Black-Scholes (European, closed-form), binomial tree (American), Monte Carlo (European, GBM and Heston dynamics, with antithetic and control-variate variance reduction).
2. **Vol surface calibration** — SVI (raw-SVI, Gatheral parameterization) fit per-expiry slice against a real SPY option chain snapshot, with explicit arbitrage-free checks across the surface.
3. **Interactive dashboard** — Streamlit app with two tabs: a manual parameter calculator (with a live price heatmap) and a market-data-driven calibration view.

### Explicit Non-Goals for v1 (Phase 2 backlog — do not build these now)

- Exotic payoffs (barriers, Asians, autocallables)
- SABR as an alternative vol parameterization
- Monte Carlo-based Greeks (bump-and-reprice, pathwise, likelihood ratio)
- Jointly calibrating Heston parameters to the same option chain via optimization (Heston is used only as an alternative simulatable dynamics for pricing/comparison against the SVI-implied smile — NOT fit to market data via an optimizer. This is a deliberate scope boundary; Heston calibration via optimization is notoriously fragile (local minima, Feller condition constraints) and is out of scope for v1.)
- Multi-day / time-evolving surface snapshots (single snapshot day only for v1)

---

## Core Modules

1. **`pricing/`** — Pricing engines
   - `black_scholes.py` — closed-form European price + Greeks (delta, gamma, vega, theta, rho) for call and put
   - `binomial.py` — Cox-Ross-Rubinstein binomial tree, American exercise only (this is the point of including it alongside BS — it should NOT also support European, since the comparison table's value comes from contrasting American early-exercise pricing against European BS pricing)
   - `monte_carlo.py` — European pricing via simulation. Must support:
     - GBM path generation
     - Heston path generation (Euler or Milstein discretization, enforce Feller condition warning if violated by chosen parameters)
     - Antithetic variates
     - Control variates (using BS closed-form as the control)
   - `base.py` — shared interfaces / abstract base class so all three engines expose a consistent `price(S, K, T, r, sigma, option_type)` style signature where feasible (binomial and MC will need additional params — accommodate via kwargs or engine-specific config objects, not by breaking the common interface)

2. **`vol_surface/`** — Calibration
   - `svi.py` — raw-SVI parameterization, per-expiry-slice fitting (5 parameters: a, b, rho, m, sigma)
   - `arbitrage_checks.py` — Roper-style no-calendar-arbitrage checks across expiry slices, Gatheral-Jacquier no-butterfly-arbitrage checks within a slice
   - `iv_inversion.py` — Black-Scholes IV inversion from observed option close price (Newton-Raphson or Brent's method) — this is how implied vols are derived since Massive's free tier does not provide pre-computed IV

3. **`data/`** — Massive API data layer
   - `massive_client.py` — thin wrapper around Massive REST calls (reference contracts endpoint + previous-day/aggregates endpoint). Respect the 5 calls/minute free-tier rate limit — add throttling/sleep between calls, do NOT parallelize requests.
   - `chain_builder.py` — orchestrates: pull reference contracts for SPY (universe of strikes/expiries) → pull EOD aggregates per contract → drop contracts with no qualifying trade (empty aggregate response) → assemble into a clean DataFrame (strike, expiry, close price, contract ticker)
   - This is a ONE-TIME pull. Build a standalone script (`scripts/fetch_snapshot.py`) that runs once and writes to SQLite. The dashboard reads only from SQLite, never calls Massive live.

4. **`storage/`** — SQLite persistence layer (single snapshot table: ticker, strike, expiration, close_price, fetched_at)

5. **`dashboard/`** — Streamlit app, two tabs (see Dashboard Specification below)

6. **`tests/`** — pytest, light touch (see Testing Specification below)

7. **`notebooks/`** — optional research notebook(s) documenting SVI fit quality, arbitrage check results, methodology writeup

---

## Dashboard Specification

### Tab 1: Manual Parameters

**Core inputs** (number inputs or sliders): S (spot), K (strike), T (time to maturity, years), σ (volatility), r (risk-free rate)

**Pricing comparison table**: side-by-side call and put prices from three engines on the same inputs:
- Black-Scholes (European)
- Binomial tree (American — this is intentional; the table is contrasting European vs American pricing, not three estimates of the same value. Consider showing "early exercise premium" = American price − European BS price as a derived column, since it's a natural and informative byproduct.)
- Monte Carlo (European; user can select GBM or Heston dynamics via a dropdown/radio)

**Greeks display**: BS Greeks only (delta, gamma, vega, theta, rho) for both call and put, shown below or beside the comparison table.

**Heatmap section**: two heatmaps side by side (call price, put price), BS pricing only (must stay fast/responsive on every slider drag — do not use binomial or MC here).
- X/Y axes: Spot price (S) and volatility (σ), swept across a grid
- K, T, r held fixed at the core input values above
- Grid resolution: 10×10 cells
- Axis ranges set via four sliders: min spot, max spot, min vol, max vol
- The single point (S, σ) from the core inputs above should be highlighted/marked as a distinct cell or marker on both heatmaps, visually tying the calculator to the heatmap
- Recompute the full grid on every slider movement (BS is cheap enough for this to be instant — this is exactly why MC/binomial are excluded from the heatmap)

### Tab 2: Market Data Snapshot

- Load the cached SPY chain snapshot from SQLite (built by `scripts/fetch_snapshot.py`, never fetched live in the dashboard)
- Display the raw chain (strikes, expiries, close prices) in a table
- Run BS IV inversion on each contract's close price
- Fit SVI per expiry slice, display fitted smile curves overlaid on the raw IV points (one chart per expiry, or a combined 3D/2D surface view)
- Display arbitrage-check results (no-calendar, no-butterfly) — pass/fail or violation magnitude per check, not just a boolean
- Clearly label this tab as using single-day end-of-day close prices as a price proxy (not true bid/ask mid), since Massive's free tier does not expose quotes — this is a deliberate, disclosed methodology choice, not a hidden limitation

---

## Testing Specification (light touch — not a full suite)

- `test_black_scholes.py`: BS price against known closed-form benchmark values (e.g., standard textbook examples); put-call parity check (C − P = S − K·e^(−rT))
- `test_monte_carlo.py`: MC price converges to BS closed-form price as number of paths → ∞ (use a generous tolerance and a reasonably large path count, not an exhaustive convergence study)
- `test_binomial.py`: binomial price converges to BS price as number of steps → ∞ when run in a European-equivalent limit, OR simply sanity-check against a known textbook American option value
- Keep this to a handful of well-chosen tests, not exhaustive coverage. This is a portfolio signal ("I write tests for numerical correctness"), not a production test suite.

---

## Data Layer Notes (Massive API)

- Free tier: "Options Basic" — 5 API calls/minute, end-of-day data, reference data, minute aggregates. NO real-time snapshots, NO quotes (bid/ask) endpoint — both return HTTP 403 on the free tier. Confirmed directly against the live API; do not attempt to use `/v3/snapshot/options/...` or `/v3/quotes/...` endpoints, they will fail.
- Usable endpoints: `/v3/reference/options/contracts` (contract universe — free, paginated), `/v2/aggs/ticker/{optionsTicker}/prev` or `/v2/aggs/ticker/{optionsTicker}/range/...` (EOD OHLCV per contract — free, but returns an empty/fieldless response if the contract had no qualifying trades that day — this is NOT an error, just means drop that contract from the dataset)
- Universe: single ticker (SPY), single snapshot day, a handful of near-term expiries with strikes spanning a reasonable range around spot to get enough slices for calendar-arbitrage checks to be meaningful
- Rate limit math: full snapshot pull is roughly 60-90 sequential calls (reference pagination + one aggregate call per contract). At 5 calls/minute this takes ~12-18 minutes. This is fine — it happens once, not repeatedly. Add a `time.sleep()` throttle in `massive_client.py` and do not attempt to parallelize or batch around the rate limit.
- Price proxy: daily close price (not bid/ask mid, not VWAP) is the agreed-upon single-price input for BS IV inversion. This is a disclosed simplification, not an oversight — document it in the dashboard UI and README.

---

## Engineering Practices (learning goals for this project)

- OOP structure throughout `pricing/` and `vol_surface/` — shared base classes/interfaces where sensible
- `.env` for the Massive API key (never commit it; `.env.example` with a placeholder should be committed)
- `requirements.txt` or `pyproject.toml`
- Git: feature branches per module (e.g., `feature/black-scholes`, `feature/svi-calibration`, `feature/dashboard`), meaningful commit messages
- pytest as described above
- README with project overview, methodology notes (especially the disclosed simplifications: close-price-as-IV-input, Heston-not-jointly-calibrated, single-snapshot-day), and setup instructions

@option-pricing-lab-spec.md

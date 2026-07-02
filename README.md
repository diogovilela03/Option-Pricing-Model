# option-pricing-lab

A pricing and vol-surface calibration toolkit built as a portfolio project, demonstrating option pricing theory, numerical methods, stochastic volatility modelling, exotic/structured product pricing, and software engineering practice.

---

## What it does

### Vanilla pricing engines

| Engine | Exercise style | Key feature |
|---|---|---|
| Black-Scholes | European | Closed-form price + 8 Greeks (delta, gamma, vega, theta, rho, volga, vanna, charm) |
| Binomial tree (CRR) | American | Early-exercise premium vs BS European baseline |
| Monte Carlo | European | GBM and Heston dynamics; antithetic + control-variate variance reduction |

The binomial tree is American-only by design — the point is to show the early-exercise premium against the BS baseline, not to reproduce the same number three ways.

### Exotic & structured products

| Category | Products | Notes |
|---|---|---|
| Path-dependent | Digital (cash-or-nothing, asset-or-nothing), Asian (geometric closed-form, arithmetic Monte Carlo), Barrier (knock-in/knock-out), Double Barrier, Quanto | Barrier pricer uses the Reiner-Rubinstein closed-form table, validated against Monte Carlo (with a Broadie-Glasserman-Kou continuity correction) and cross-checked against QuantLib's reference implementation |
| Structured products | Reverse Convertible, Barrier Reverse Convertible, Discount Certificate, Bonus Certificate, Airbag Certificate, Twin-Win Certificate | Built as compositions of the vanilla + barrier pricers |
| Autocallables (Monte Carlo) | Autocall Incremental, Phoenix, Phoenix Memory | Configurable observation frequency and DIP (protection barrier) style — European (checked only at maturity) or American (knocked in the first time the barrier is breached on any observation date). Reports equivalent expected maturity, equivalent zero-coupon-bond value, forward at maturity, probability of capital loss, and a per-observation-date maturity/coupon probability table |
| Multi-asset | Basket, Worst-of, Rainbow | Correlated-asset payoffs via Monte Carlo, with an editable, PSD-validated N×N correlation matrix (n ≥ 3 assets) alongside a simple uniform-ρ slider |
| Option strategies | Long call/put, bull call / bear put spread, straddle, strangle, risk reversal, butterfly, condor, iron condor | Analytic P&L diagrams via summed signed per-leg BS Greeks |

A shared finite-difference Greeks helper (`pricing/greeks_fd.py`) bumps-and-reprices delta/gamma/vega/theta/rho around any `price_fn(S, K, T, r, sigma)`, validated against analytic BS Greeks. It powers the Premium-vs-Spot and Greeks-vs-Spot charts across Tab 1 and Tab 3. For Monte Carlo-priced products where FD bumping is too noisy (e.g. arithmetic Asian), the dashboard falls back to a closed-form proxy and discloses the simplification rather than showing an unstable chart — see [Future improvements](#future-improvements).

### Volatility surface calibration

- **SVI** (raw-SVI / Gatheral parameterisation), fit per expiry slice
- **SSVI** (Gatheral-Jacquier power-law), globally consistent surface
- **Heston characteristic-function pricer** for closed-form Heston prices and calibration
- **SANOS** (Smooth Arbitrage-free Non-parametric Option Surfaces, Buehler et al. 2026) — per-expiry LP fit in pure price space; arbitrage-free by construction rather than by post-hoc check
- Arbitrage-free checks: no-calendar-arbitrage (across slices), no-butterfly-arbitrage (within a slice)
- Calibration target: a real SPY option chain snapshot sourced from yfinance or the Massive API

### Interactive Streamlit dashboard

**Tab 1 — Manual Parameters**
- Inputs: S, K, T, σ, r
- Side-by-side call/put price comparison: BS (European), binomial (American), Monte Carlo (GBM or Heston)
- Early-exercise premium column
- 8 BS Greeks for call and put
- Two live 10×10 heatmaps sweeping spot and vol (BS only, stays responsive); current (S, σ) point highlighted
- Premium-vs-Spot and P&L-vs-Spot charts

**Tab 2 — Market Data Snapshot**
- Cached single-day SPY chain loaded from SQLite (no live API calls from the dashboard)
- Self-derived implied vols via BS inversion on bid/ask mid price (falls back to close price when quotes are unavailable)
- SVI, SSVI, Heston, and SANOS smile curves overlaid on raw IV points per expiry
- Arbitrage-check results with violation magnitudes, not just pass/fail

**Tab 3 — Exotic Structures**

Five categories, switchable via the sidebar: Option Strategies, Path-Dependent, Structured Products, Autocallables, Multi-Asset (see table above for the full product list).

- Premium-vs-Spot and Greeks-vs-Spot panels for every path-dependent and structured product
- A genuine terminal payoff diagram (S_T vs. barrier convention) alongside the PV-vs-spot chart, kept visually distinct so the two aren't conflated
- Autocallable analytics: equivalent expected maturity, equivalent ZCB, forward at maturity, capital-loss probability, per-observation-date table
- Multi-asset correlation-sensitivity chart, driven by either a uniform-ρ slider or the editable N×N correlation matrix

---

## Methodology disclosures

**Bid/ask mid where available, close price otherwise** — the yfinance source now carries `bid`/`ask` fields alongside `lastPrice`; when both are present the mid is used as the IV-inversion input, since it's a materially better price proxy than a stale last trade. The Massive free tier still has no bid/ask quotes endpoint (HTTP 403), so contracts sourced from Massive fall back to daily close price. This distinction is disclosed in the dashboard UI rather than silently mixed.

**Heston is not jointly calibrated to the chain** — Heston is used as an alternative simulatable dynamics for pricing and smile comparison against SVI/SSVI. Jointly calibrating Heston parameters via optimisation against an option chain is a notoriously fragile numerical problem (local minima, Feller condition constraints) and is deliberately out of scope — not a backlog item, a permanent design boundary.

**Single-day snapshot** — the chain is a single end-of-day snapshot; there is no time-evolving surface currently.

**FD Greeks fall back to a closed-form proxy for noisy Monte Carlo products** — bump-and-reprice Greeks are unstable for products priced via arithmetic-average Monte Carlo (e.g. arithmetic Asian) at interactive path counts. The dashboard uses the closed-form geometric Asian as a stand-in for the Greeks chart in that case and says so in the UI, rather than silently showing noisy output.

---

## Project structure

```
pricing/
  black_scholes.py     closed-form price + 8 Greeks
  binomial.py          CRR binomial tree, American exercise
  monte_carlo.py        GBM + Heston dynamics, antithetic + control variates
  heston_cf.py          Heston characteristic-function pricer
  base.py                shared OptionPricer ABC
  digital.py             cash-or-nothing / asset-or-nothing digitals
  asian.py                geometric (closed-form) + arithmetic (MC) Asian options
  barrier.py             Reiner-Rubinstein closed-form barrier pricer + MC validator
  double_barrier.py     double knock-out/in barrier pricer
  quanto.py               quanto call/put
  structured.py          reverse convertible, discount/bonus/airbag/twin-win certificates
  autocall.py             autocall incremental, Phoenix, Phoenix Memory (Monte Carlo)
  multi_asset.py         basket, worst-of, rainbow options + correlation sensitivity
  greeks_fd.py            generic finite-difference Greeks, reusable across any pricer

vol_surface/
  svi.py                  raw-SVI per-slice calibration
  ssvi.py                 SSVI global surface
  heston_calibration.py  L-BFGS-B Heston calibration
  sanos.py                SANOS non-parametric LP fit (pure price space)
  iv_inversion.py        Newton-Raphson + Brent BS IV inversion
  arbitrage_checks.py    calendar + butterfly arbitrage checks

data/
  massive_client.py      Massive API thin wrapper (rate-limited, free tier)
  chain_builder.py       contract universe → EOD aggregates → DataFrame
  yfinance_client.py     yfinance chain builder (fast, no key required)

storage/
  db.py                   SQLite persistence (init, insert, load)

dashboard/
  app.py                  Streamlit entry point
  tab1_manual.py          manual parameter calculator
  tab2_market.py          market data snapshot view
  tab3_exotic.py          Tab 3 router across the 5 categories
  tab3_strategies.py     option strategies (P&L diagrams)
  tab3_path_dep.py       path-dependent + structured products
  tab3_autocall.py       autocallables
  tab3_multi.py           multi-asset options
  charts.py                shared Plotly chart helpers

scripts/
  fetch_snapshot.py      one-time data pull → writes to data/spy_chain.db

tests/
  16 files, 163 tests — see Tests below
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/diogovilela03/Option-Pricing-Model.git
cd Option-Pricing-Model
pip install -r requirements.txt
```

### 2. Environment variables (optional — only needed for Massive API source)

```bash
cp .env.example .env
# Edit .env and set MASSIVE_API_KEY=<your key>
```

The `yfinance` source requires no API key and is the default.

### 3. Fetch the option chain snapshot

```bash
# Fast (~seconds), no API key needed:
python scripts/fetch_snapshot.py --source yfinance

# Slow (~15-20 min, rate-limited to 5 calls/min):
python scripts/fetch_snapshot.py --source massive
```

This writes a SQLite file to `data/spy_chain.db`. The dashboard reads only from this file.

### 4. Launch the dashboard

```bash
streamlit run dashboard/app.py
```

---

## Tests

```bash
pytest
```

163 tests across 16 files, including:
- BS benchmark values against Hull (10th ed.), put-call parity, and Greek sign/magnitude sanity checks
- Binomial American call = European BS (no early exercise on non-dividend stock); American put > European BS put; convergence as steps increase
- MC GBM convergence to BS closed-form (plain, antithetic, control-variate); Heston non-negativity, low vol-of-vol limit near GBM, Feller condition warning
- Barrier pricer: in/out parity, non-negativity, boundary guards for spot already past the barrier at inception, and Monte Carlo regression guards across every barrier-type/option-type/strike-vs-barrier combination
- Digital, Asian, double-barrier, quanto: closed-form sanity checks and MC agreement where applicable
- Structured products and autocallables: capital-loss monotonicity, parameter-shift regression guards, DIP American/European style consistency
- Multi-asset: correlation-sensitivity direction, PSD validation
- SVI fit quality and arbitrage checks (calendar + butterfly)
- `greeks_fd.py`: finite-difference Greeks validated against analytic BS Greeks across ITM/ATM/OTM cases, plus edge cases (deep OTM, near-zero vol)

---

## Future improvements

- **SABR** as an alternative vol parameterisation, compared against SVI
- **Multi-day snapshots** showing the surface evolving over time, rather than a single end-of-day cross-section
- **Pathwise / likelihood-ratio Greeks** for Monte Carlo-priced exotics (arithmetic Asian, autocallables), replacing the current closed-form-proxy workaround used where bump-and-reprice is too noisy
- **Broader data source coverage** — try other tickers and vendors (e.g. a paid real-time/quote-level feed, or a different underlying beyond SPY) to see how calibration quality and coverage change with richer input data, rather than being limited to a single free-tier EOD snapshot

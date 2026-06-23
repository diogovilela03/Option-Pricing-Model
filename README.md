# option-pricing-lab

A pricing and vol-surface calibration toolkit built as a portfolio project, demonstrating option pricing theory, numerical methods, stochastic volatility modelling, and software engineering practice.

---

## What it does

### Pricing engines

| Engine | Exercise style | Key feature |
|---|---|---|
| Black-Scholes | European | Closed-form price + 8 Greeks (delta, gamma, vega, theta, rho, volga, vanna, charm) |
| Binomial tree (CRR) | American | Early-exercise premium vs BS European baseline |
| Monte Carlo | European | GBM and Heston dynamics; antithetic + control-variate variance reduction |

The binomial tree is American-only by design — the point is to show the early-exercise premium against the BS baseline, not to reproduce the same number three ways.

### Volatility surface calibration

- **SVI** (raw-SVI / Gatheral parameterisation), fit per expiry slice
- **SSVI** (Gatheral-Jacquier power-law), globally consistent surface
- **Heston characteristic-function pricer** for closed-form Heston prices and calibration
- Arbitrage-free checks: no-calendar-arbitrage (across slices), no-butterfly-arbitrage (within a slice)
- Calibration target: a real SPY option chain snapshot sourced from yfinance or the Massive API

### Interactive Streamlit dashboard

**Tab 1 — Manual Parameters**
- Inputs: S, K, T, σ, r
- Side-by-side call/put price comparison: BS (European), binomial (American), Monte Carlo (GBM or Heston)
- Early-exercise premium column
- 8 BS Greeks for call and put
- Two live 10×10 heatmaps sweeping spot and vol (BS only, stays responsive); current (S, σ) point highlighted

**Tab 2 — Market Data Snapshot**
- Cached single-day SPY chain loaded from SQLite (no live API calls from the dashboard)
- Self-derived implied vols via BS inversion on close prices
- SVI, SSVI, and Heston smile curves overlaid on raw IV points per expiry
- Arbitrage-check results with violation magnitudes, not just pass/fail

---

## Methodology disclosures

**Close price as IV input** — the Massive free tier has no bid/ask quotes endpoint (HTTP 403). The yfinance `lastPrice` field is similarly a last-trade price, not a true bid/ask mid. Daily close price is the agreed-upon single-price proxy for BS IV inversion. This is disclosed in the dashboard UI and is the stronger engineering story vs relying on a pre-computed vendor IV column.

**Heston is not jointly calibrated to the chain** — Heston is used as an alternative simulatable dynamics for pricing and smile comparison against SVI/SSVI. Jointly calibrating Heston parameters via optimisation against an option chain is a notoriously fragile numerical problem (local minima, Feller condition constraints) and is explicitly out of scope for v1.

**Single-day snapshot** — the chain is a single end-of-day snapshot; there is no time-evolving surface in v1.

---

## Project structure

```
pricing/
  black_scholes.py     closed-form price + 8 Greeks
  binomial.py          CRR binomial tree, American exercise
  monte_carlo.py       GBM + Heston dynamics, antithetic + control variates
  heston_cf.py         Heston characteristic-function pricer
  base.py              shared OptionPricer ABC

vol_surface/
  svi.py               raw-SVI per-slice calibration
  ssvi.py              SSVI global surface
  heston_calibration.py  L-BFGS-B Heston calibration
  iv_inversion.py      Newton-Raphson + Brent BS IV inversion
  arbitrage_checks.py  calendar + butterfly arbitrage checks

data/
  massive_client.py    Massive API thin wrapper (rate-limited, free tier)
  chain_builder.py     contract universe → EOD aggregates → DataFrame
  yfinance_client.py   yfinance chain builder (fast, no key required)

storage/
  db.py                SQLite persistence (init, insert, load)

dashboard/
  app.py               Streamlit entry point
  tab1_manual.py       manual parameter calculator
  tab2_market.py       market data snapshot view
  charts.py            shared Plotly chart helpers

scripts/
  fetch_snapshot.py    one-time data pull → writes to data/spy_chain.db

tests/
  test_black_scholes.py
  test_binomial.py
  test_monte_carlo.py
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/diogovilela03/option-pricing-lab.git
cd option-pricing-lab
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
pytest tests/test_black_scholes.py tests/test_binomial.py tests/test_monte_carlo.py -v
```

36 tests covering:
- BS benchmark values against Hull (10th ed.) and put-call parity
- BS Greek sign and magnitude sanity checks
- Binomial American call = European BS (no early exercise on non-dividend stock)
- Binomial American put > European BS put; convergence as steps increase
- MC GBM convergence to BS closed-form (plain, antithetic, control-variate)
- Heston pricing non-negativity; low vol-of-vol limit near GBM; Feller condition warning

---

## Explicitly deferred to v2

- Exotic payoffs (barriers, Asians, autocallables)
- SABR as an alternative vol parameterisation (comparison against SVI)
- Monte Carlo Greeks (bump-and-reprice, pathwise, likelihood ratio)
- Jointly-calibrated Heston via optimisation against the chain
- Multi-day snapshots showing the surface evolving over time

# option-pricing-lab — Project Specification

## Overview

`option-pricing-lab` is a portfolio project demonstrating option pricing theory, numerical methods, and volatility surface calibration. It is the second piece in a two-project portfolio alongside `forward-curve-arbitrage`: that project covers forward curve construction and Delta One mechanics, while this project covers pricing engines, Monte Carlo methods, and arbitrage-free vol surface calibration. The two are deliberately scoped not to overlap, so together they tell a broader story about derivatives pricing and market microstructure than either would alone.

## Motivation

Built as a demonstration of:
- Closed-form and numerical option pricing (Black-Scholes, binomial trees, Monte Carlo with variance reduction)
- Stochastic volatility modeling (Heston dynamics) as a comparison tool against market-implied smiles
- Arbitrage-free volatility surface construction (SVI parameterization, calendar and butterfly arbitrage checks) — directly relevant to market-making and vol-trading roles
- Software engineering practice: OOP design, testing for numerical correctness, environment/secrets management, git workflow

Target audience: quantitative research, quant trading, and quant developer interviewers at market-making and systematic trading firms (Optiver, IMC, Flow Traders, QRT, Man AHL, and similar).

## Scope (v1)

### Pricing Engines
| Engine | Exercise Style | Purpose |
|---|---|---|
| Black-Scholes | European | Closed-form baseline, Greeks |
| Binomial Tree (CRR) | American | Demonstrates early-exercise premium vs BS |
| Monte Carlo | European | GBM and Heston dynamics, antithetic + control variates |

Binomial is intentionally American-only — the value of including it is contrasting early-exercise pricing against the European BS baseline, not reproducing the same number three ways.

### Volatility Surface
- SVI (raw-SVI / Gatheral parameterization), fit per expiry slice
- Arbitrage-free checks: no-calendar arbitrage (Roper-style, across expiry slices), no-butterfly arbitrage (Gatheral-Jacquier, within a slice)
- Calibration target: a single real SPY option chain snapshot (one day), sourced from the Massive API free tier
- Implied vols are self-derived via Black-Scholes inversion on daily close prices — Massive's free tier does not expose bid/ask quotes or pre-computed IV/Greeks, so this project computes both directly rather than consuming a vendor field. This is a disclosed methodology choice and a stronger engineering story than relying on a pre-computed IV column.
- Heston is used as an alternative simulatable dynamics for pricing comparison against the SVI-implied smile — it is NOT calibrated to the chain via optimization. Joint Heston calibration is a notoriously fragile numerical problem (local minima, Feller condition constraints) and is explicitly deferred to phase 2 to keep v1 deliverable.

### Dashboard (Streamlit)

**Tab 1 — Manual Parameters**
- Inputs: S, K, T, σ, r
- Output: side-by-side call/put price comparison across BS (European), binomial (American), and Monte Carlo (European, GBM or Heston selectable), plus the implied early-exercise premium
- BS Greeks (delta, gamma, vega, theta, rho) for call and put
- Two live 10×10 heatmaps (call price, put price) sweeping spot and volatility, with K/T/r fixed, computed via BS only for responsiveness; axis ranges set by sliders; the calculator's current (S, σ) point is highlighted on the grid

**Tab 2 — Market Data Snapshot**
- Cached single-day SPY chain (no live API calls from the dashboard itself)
- Self-derived implied vols, SVI fits per expiry, overlaid smile charts
- Arbitrage-check results displayed explicitly (not hidden as a pass/fail badge — show violation magnitude where relevant)

## Data Layer

**Source**: Massive API, free "Options Basic" tier only.

**What's available for free**: contract reference data (strikes, expiries, exercise style — unlimited within the 5 calls/minute rate limit), end-of-day OHLCV aggregates per contract.

**What's gated (paid tier)**: real-time chain snapshots, historical quotes (bid/ask).

**Implication**: option price input to IV inversion is the daily close price, not a bid/ask mid. This is a known, disclosed limitation rather than a hidden one.

**Pull strategy**: a one-time script builds the contract universe from the reference endpoint, then pulls EOD aggregates per contract (dropping any contract with no qualifying trade that day), and writes the result to SQLite. The Streamlit dashboard reads only from SQLite — it never calls Massive live. Given the 5 calls/minute limit, a full snapshot pull (roughly 60-90 calls across reference pagination and per-contract aggregates) takes on the order of 15 minutes; this happens once, not on every dashboard load.

## Testing

A light pytest suite focused on numerical correctness, not exhaustive coverage:
- Black-Scholes price against known textbook benchmark values
- Put-call parity holds for BS outputs
- Monte Carlo price converges to the BS closed-form price as path count grows
- Binomial price is sanity-checked against a known benchmark / converges appropriately

## Explicitly Deferred to Phase 2

- Exotic payoffs (barriers, Asians, autocallables)
- SABR as an alternative vol surface parameterization (comparison against SVI)
- Monte Carlo-based Greeks (bump-and-reprice, pathwise, likelihood ratio methods)
- Jointly-calibrated Heston parameters via optimization against the option chain
- Multi-day snapshots showing the surface evolving over time

## Delivery

- Standalone GitHub repo: `option-pricing-lab`
- Cross-linked with `forward-curve-arbitrage` as a second, distinct project on personal portfolio site
- Build workflow: Claude Code, driven by `CLAUDE.md`, same pattern as `forward-curve-arbitrage`
- Engineering practices as learning goals: OOP module structure, `.env` for API key management, git feature branching, pytest

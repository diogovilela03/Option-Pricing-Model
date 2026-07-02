"""Generic finite-difference Greeks, reusable across any pricer.

Every exotic/structured pricer in this project has a different signature
(barrier options need a barrier level, digitals need a payoff style, etc.),
so instead of writing a bespoke Greeks function per product, callers close
over the product-specific parameters into a plain `price_fn(S, K, T, r,
sigma) -> float` and this module bumps-and-reprices around it.
"""
from typing import Callable

PriceFn = Callable[[float, float, float, float, float], float]

_EPS_S = 1e-3       # relative bump on spot
_EPS_SIGMA = 1e-4   # absolute bump on vol
_EPS_R = 1e-4        # absolute bump on rate
_EPS_T_DAYS = 1.0    # theta horizon: 1 calendar day


def fd_greeks(price_fn: PriceFn, S: float, K: float, T: float, r: float, sigma: float) -> dict[str, float]:
    """Bump-and-reprice delta, gamma, vega, theta, rho around (S,K,T,r,sigma).

    vega/rho are scaled "per 1% move" (matching pricing/black_scholes.py's
    analytic convention); theta is "per calendar day" (price decay as T
    shrinks by one day, holding S/r/sigma fixed).
    """
    p0 = price_fn(S, K, T, r, sigma)

    dS = S * _EPS_S
    p_S_up = price_fn(S + dS, K, T, r, sigma)
    p_S_dn = price_fn(S - dS, K, T, r, sigma)
    delta = (p_S_up - p_S_dn) / (2 * dS)
    gamma = (p_S_up - 2 * p0 + p_S_dn) / (dS ** 2)

    p_vega_up = price_fn(S, K, T, r, sigma + _EPS_SIGMA)
    p_vega_dn = price_fn(S, K, T, r, max(sigma - _EPS_SIGMA, 1e-6))
    vega = (p_vega_up - p_vega_dn) / (2 * _EPS_SIGMA) / 100.0

    p_rho_up = price_fn(S, K, T, r + _EPS_R, sigma)
    p_rho_dn = price_fn(S, K, T, r - _EPS_R, sigma)
    rho = (p_rho_up - p_rho_dn) / (2 * _EPS_R) / 100.0

    dt_years = _EPS_T_DAYS / 365.0
    T_dn = max(T - dt_years, 1e-6)
    theta = price_fn(S, K, T_dn, r, sigma) - p0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}


def fd_greeks_profile(
    price_fn: PriceFn, S_grid, K: float, T: float, r: float, sigma: float,
) -> dict[str, list[float]]:
    """fd_greeks evaluated across a grid of spot values, transposed into
    per-Greek lists (delta[], gamma[], vega[], theta[], rho[]) for charting."""
    profile: dict[str, list[float]] = {"delta": [], "gamma": [], "vega": [], "theta": [], "rho": []}
    for s in S_grid:
        g = fd_greeks(price_fn, float(s), K, T, r, sigma)
        for name in profile:
            profile[name].append(g[name])
    return profile

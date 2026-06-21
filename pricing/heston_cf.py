"""Heston (1993) closed-form European option pricer via characteristic functions.

Uses the 'little trap' formulation (Albrecher et al. 2007) to avoid branch-cut
discontinuities that appear in the original Heston CF at certain parameter values.

Parameters
----------
v0    : initial variance (σ² at t=0)
kappa : mean-reversion speed
theta : long-run variance
xi    : vol-of-vol
rho   : spot–vol correlation ∈ (−1, 1)

Feller condition: 2·κ·θ > ξ²  →  variance process stays strictly positive.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad


def _cf_risk_neutral(
    u: complex,
    S: float, v0: float, kappa: float, theta: float, xi: float, rho: float,
    r: float, T: float,
) -> complex:
    """Heston CF of ln(S_T) under the risk-neutral measure (little trap)."""
    iu = 1j * u
    d = np.sqrt((iu * rho * xi - kappa) ** 2 + xi ** 2 * (iu + u ** 2))
    g = (kappa - iu * rho * xi - d) / (kappa - iu * rho * xi + d)

    C = r * iu * T + kappa * theta / xi ** 2 * (
        (kappa - iu * rho * xi - d) * T
        - 2 * np.log((1 - g * np.exp(-d * T)) / (1 - g))
    )
    D = (kappa - iu * rho * xi - d) / xi ** 2 * (
        (1 - np.exp(-d * T)) / (1 - g * np.exp(-d * T))
    )
    # r·T already included in C; only log(S) here to avoid double-counting
    return np.exp(C + D * v0 + iu * math.log(S))


def heston_price(
    S: float,
    K: float,
    T: float,
    r: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    option_type: str = "call",
) -> float:
    """European option price under the Heston model.

    Computes P1 (stock-measure prob) and P2 (risk-neutral prob) via numerical
    integration of the characteristic function, then applies C = S·P1 − K·e^{-rT}·P2.
    """
    log_K = math.log(K)
    disc = math.exp(-r * T)

    cf_minus_i = _cf_risk_neutral(-1j, S, v0, kappa, theta, xi, rho, r, T)

    def _p1_integrand(u: float) -> float:
        cf = _cf_risk_neutral(u - 1j, S, v0, kappa, theta, xi, rho, r, T) / cf_minus_i
        return float(np.real(np.exp(-1j * u * log_K) * cf / (1j * u)))

    def _p2_integrand(u: float) -> float:
        cf = _cf_risk_neutral(u, S, v0, kappa, theta, xi, rho, r, T)
        return float(np.real(np.exp(-1j * u * log_K) * cf / (1j * u)))

    p1_int, _ = quad(_p1_integrand, 1e-9, 100.0, limit=100, epsabs=1e-4, epsrel=1e-4)
    p2_int, _ = quad(_p2_integrand, 1e-9, 100.0, limit=100, epsabs=1e-4, epsrel=1e-4)

    P1 = 0.5 + p1_int / math.pi
    P2 = 0.5 + p2_int / math.pi

    intrinsic = max(S - K * disc, 0.0)
    call = max(S * P1 - K * disc * P2, intrinsic)

    if option_type == "call":
        return float(call)
    return float(max(call - S + K * disc, 0.0))

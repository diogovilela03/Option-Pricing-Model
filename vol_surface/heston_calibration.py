"""Calibrate Heston parameters to a market option chain.

Strategy
--------
1. Sample up to `n_sample` representative contracts from the chain.
2. Run `n_restarts` local L-BFGS-B optimisations from random starting points.
3. Return the best result: params, RMSE, and Feller condition status.

Objective: minimise mean-squared error in implied-vol space,
           IV_heston(K,T) vs IV_market(K,T).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pricing.heston_cf import heston_price
from vol_surface.iv_inversion import implied_vol

_BOUNDS = [
    (1e-4, 1.0),    # v0
    (1e-2, 10.0),   # kappa
    (1e-4, 1.0),    # theta
    (1e-2, 2.0),    # xi  (vol-of-vol)
    (-0.98, 0.98),  # rho
]


def _heston_iv(
    S: float, K: float, T: float, r: float,
    v0: float, kappa: float, theta: float, xi: float, rho: float,
    option_type: str,
) -> float:
    """Heston implied vol for one contract; NaN on failure."""
    try:
        price = heston_price(S, K, T, r, v0, kappa, theta, xi, rho, option_type)
        return implied_vol(S, K, T, r, price, option_type)
    except Exception:
        return float("nan")


def calibrate_heston(
    market_df,
    S: float,
    r: float,
    n_sample: int = 40,
    n_restarts: int = 5,
    seed: int = 42,
) -> dict:
    """Fit Heston to market IVs.

    Parameters
    ----------
    market_df : DataFrame with columns strike, T, iv, option_type (pre-filtered)
    S         : spot price
    r         : risk-free rate
    n_sample  : max contracts used (random sample for speed)
    n_restarts: number of random L-BFGS-B restarts

    Returns
    -------
    dict: v0, kappa, theta, xi, rho, rmse, feller_satisfied, feller_lhs
    """
    df = market_df.dropna(subset=["iv"]).copy()
    df = df[df["iv"].between(0.01, 1.5)]
    if df.empty:
        raise ValueError("No valid market IVs for Heston calibration")

    if len(df) > n_sample:
        df = df.sample(n_sample, random_state=seed)

    strikes  = df["strike"].values
    Ts       = df["T"].values
    iv_mkt   = df["iv"].values
    otypes   = df["option_type"].values

    def objective(params: np.ndarray) -> float:
        v0, kappa, theta, xi, rho = params
        sq_errors = []
        for K, T, iv_m, ot in zip(strikes, Ts, iv_mkt, otypes):
            iv_h = _heston_iv(S, K, max(T, 1e-4), r, v0, kappa, theta, xi, rho, ot)
            if not np.isnan(iv_h):
                sq_errors.append((iv_h - iv_m) ** 2)
        return float(np.mean(sq_errors)) if sq_errors else 1.0

    rng = np.random.default_rng(seed)
    best_res, best_val = None, np.inf

    for _ in range(n_restarts):
        x0 = [
            rng.uniform(0.01, 0.5),   # v0
            rng.uniform(0.5,  5.0),   # kappa
            rng.uniform(0.01, 0.5),   # theta
            rng.uniform(0.1,  1.0),   # xi
            rng.uniform(-0.8, 0.0),   # rho (negative skew prior)
        ]
        try:
            res = minimize(
                objective, x0, method="L-BFGS-B", bounds=_BOUNDS,
                options={"maxiter": 200, "ftol": 1e-9, "gtol": 1e-6},
            )
            if res.fun < best_val:
                best_val, best_res = res.fun, res
        except Exception:
            pass

    if best_res is None:
        raise RuntimeError("Heston calibration failed for all restarts")

    v0, kappa, theta, xi, rho = best_res.x
    feller_lhs = 2 * kappa * theta - xi ** 2

    return {
        "v0":    float(v0),
        "kappa": float(kappa),
        "theta": float(theta),
        "xi":    float(xi),
        "rho":   float(rho),
        "rmse":  float(np.sqrt(best_val)),
        "feller_satisfied": bool(feller_lhs > 0),
        "feller_lhs": float(feller_lhs),
    }

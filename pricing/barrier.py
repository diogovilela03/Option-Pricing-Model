"""Barrier option pricing.

Analytical: Reiner-Rubinstein (1991) closed-form for single-barrier options.
Monte Carlo: full GBM path simulation with continuous monitoring.

Parity: In + Out = Vanilla (exact, holds to float precision).
"""
import math
import warnings
from typing import Literal

import numpy as np
from scipy.stats import norm

from pricing.base import OptionPricer, OptionType
from pricing.asian import _simulate_gbm_paths

BarrierType = Literal["down-and-out", "down-and-in", "up-and-out", "up-and-in"]


def _rr_aux(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float,
    H: float,
    R: float,
    phi: int,
    eta: int,
) -> dict:
    """Reiner-Rubinstein auxiliary building blocks A–F."""
    b = r - q
    mu_ = (b - 0.5 * sigma ** 2) / sigma ** 2
    lam = math.sqrt(mu_ ** 2 + 2 * r / sigma ** 2)
    sT = sigma * math.sqrt(T)

    x1 = math.log(S / K) / sT + (1 + mu_) * sT
    x2 = math.log(S / H) / sT + (1 + mu_) * sT
    y1 = math.log(H ** 2 / (S * K)) / sT + (1 + mu_) * sT
    y2 = math.log(H / S) / sT + (1 + mu_) * sT
    z  = math.log(H / S) / sT + lam * sT

    disc = math.exp(-r * T)
    ebT  = math.exp((b - r) * T)
    HS   = H / S

    A = phi * S * ebT * norm.cdf(phi * x1) - phi * K * disc * norm.cdf(phi * (x1 - sT))
    B = phi * S * ebT * norm.cdf(phi * x2) - phi * K * disc * norm.cdf(phi * (x2 - sT))
    C = (phi * S * ebT * HS ** (2 * (mu_ + 1)) * norm.cdf(eta * y1)
         - phi * K * disc * HS ** (2 * mu_) * norm.cdf(eta * (y1 - sT)))
    D = (phi * S * ebT * HS ** (2 * (mu_ + 1)) * norm.cdf(eta * y2)
         - phi * K * disc * HS ** (2 * mu_) * norm.cdf(eta * (y2 - sT)))
    F = R * (HS ** (mu_ + lam) * norm.cdf(eta * z)
             + HS ** (mu_ - lam) * norm.cdf(eta * (z - 2 * lam * sT)))

    return {"A": A, "B": B, "C": C, "D": D, "F": F}


class BarrierOption(OptionPricer):
    """Reiner-Rubinstein (1991) barrier option pricer (closed-form)."""

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        barrier: float,
        barrier_type: BarrierType,
        rebate: float = 0.0,
        q: float = 0.0,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        if barrier_type not in ("down-and-out", "down-and-in", "up-and-out", "up-and-in"):
            raise ValueError(f"Unknown barrier_type: {barrier_type!r}")

        H = barrier
        if abs(S / H - 1) < 1e-4:
            warnings.warn(
                f"Spot S={S} is within 0.01% of barrier H={H}. "
                "Reiner-Rubinstein formula is numerically sensitive near the barrier.",
                UserWarning,
                stacklevel=2,
            )

        phi = 1 if option_type == "call" else -1
        eta = 1 if barrier_type.startswith("down") else -1

        aux = _rr_aux(S, K, T, r, sigma, q, H, rebate, phi, eta)
        A, B, C, D, F = aux["A"], aux["B"], aux["C"], aux["D"], aux["F"]

        K_above_H = K >= H
        # RR formula table is derived for calls; puts swap the K≥H / K<H cases
        above = K_above_H if option_type == "call" else not K_above_H

        if barrier_type == "down-and-out":
            val = (A - C + F) if above else (B - D + F)
        elif barrier_type == "down-and-in":
            val = (C - F) if above else (A - B + D - F)
        elif barrier_type == "up-and-out":
            val = F if above else (A - B + D - F)
        else:  # up-and-in
            val = (A - F) if above else (B - D + F)

        return max(val, 0.0)


def barrier_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    barrier: float,
    barrier_type: BarrierType,
    paths: int = 50_000,
    steps: int = 252,
    seed: int | None = None,
) -> float:
    """MC barrier pricer using full GBM path simulation (continuous monitoring)."""
    rng = np.random.default_rng(seed)
    path_arr = _simulate_gbm_paths(S, T, r, sigma, paths, steps, rng)  # (steps+1, paths)

    H = barrier
    sign = 1.0 if option_type == "call" else -1.0
    terminal = path_arr[-1]

    if barrier_type in ("down-and-out", "down-and-in"):
        path_extreme = path_arr.min(axis=0)
        breached = path_extreme <= H
    else:
        path_extreme = path_arr.max(axis=0)
        breached = path_extreme >= H

    if barrier_type in ("down-and-out", "up-and-out"):
        survived = ~breached
        payoffs = survived * np.maximum(sign * (terminal - K), 0.0)
    else:  # knock-in
        payoffs = breached * np.maximum(sign * (terminal - K), 0.0)

    return max(math.exp(-r * T) * float(np.mean(payoffs)), 0.0)


def barrier_delta_profile(
    S_grid: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: BarrierType,
    option_type: OptionType,
    epsilon: float = 0.005,
) -> np.ndarray:
    """Numerical delta via central finite difference across a spot grid.

    Reveals the delta discontinuity/explosion near the barrier.
    """
    pricer = BarrierOption()
    deltas = np.zeros(len(S_grid))
    for i, s in enumerate(S_grid):
        s_up = s * (1 + epsilon)
        s_dn = s * (1 - epsilon)
        p_up = pricer.price(s_up, K, T, r, sigma, option_type, barrier, barrier_type)
        p_dn = pricer.price(s_dn, K, T, r, sigma, option_type, barrier, barrier_type)
        deltas[i] = (p_up - p_dn) / (s_up - s_dn)
    return deltas


def get_mc_paths_for_display(
    S: float,
    T: float,
    r: float,
    sigma: float,
    barrier: float,
    barrier_type: BarrierType,
    n_display: int = 60,
    steps: int = 126,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (time_grid, paths, knocked_mask) for visualization."""
    rng = np.random.default_rng(seed)
    paths = _simulate_gbm_paths(S, T, r, sigma, n_display, steps, rng)
    time_grid = np.linspace(0, T, steps + 1)

    if barrier_type.startswith("down"):
        knocked = paths.min(axis=0) <= barrier
    else:
        knocked = paths.max(axis=0) >= barrier

    return time_grid, paths, knocked

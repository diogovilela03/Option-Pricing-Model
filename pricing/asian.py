"""Asian option pricing.

Geometric Asian: Kemna-Vorst (1990) closed-form via modified Black-Scholes.
Arithmetic Asian: Monte Carlo with geometric Asian as control variate.
"""
import math

import numpy as np
from scipy.stats import norm

from pricing.base import OptionPricer, OptionType


def _simulate_gbm_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    paths: int,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full GBM path simulation with antithetic variates.

    Returns shape (steps+1, paths): row 0 is S0, last row is S_T.
    """
    dt = T / steps
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * math.sqrt(dt)
    half = paths // 2
    Z = rng.standard_normal((steps, half))
    Z = np.concatenate([Z, -Z], axis=1)
    log_inc = drift + vol * Z
    log_paths = np.cumsum(log_inc, axis=0)
    return S * np.exp(np.vstack([np.zeros(paths), log_paths]))


class GeometricAsian(OptionPricer):
    """Closed-form geometric Asian option (Kemna-Vorst 1990).

    Replaces Black-Scholes sigma and drift with averaging-adjusted equivalents.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        n_periods: int | None = None,
        q: float = 0.0,
        **kwargs,
    ) -> float:
        """
        n_periods: discrete observation count (None = continuous averaging).
        """
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        if n_periods is None or n_periods == 0:
            sigma_G = sigma / math.sqrt(3)
            b = 0.5 * (r - q - sigma ** 2 / 6)
        else:
            n = n_periods
            sigma_G = sigma * math.sqrt((n + 1) * (2 * n + 1) / (6 * n ** 2))
            b = (r - q - 0.5 * sigma ** 2) * (n + 1) / (2 * n) + 0.5 * sigma_G ** 2

        sT = sigma_G * math.sqrt(T)
        if sT < 1e-12:
            intrinsic = max(S * math.exp((b - r) * T) - K * math.exp(-r * T), 0.0)
            return intrinsic if option_type == "call" else max(K * math.exp(-r * T) - S * math.exp((b - r) * T), 0.0)

        d1 = (math.log(S / K) + (b + 0.5 * sigma_G ** 2) * T) / sT
        d2 = d1 - sT
        disc = math.exp(-r * T)

        if option_type == "call":
            return S * math.exp((b - r) * T) * norm.cdf(d1) - K * disc * norm.cdf(d2)
        return K * disc * norm.cdf(-d2) - S * math.exp((b - r) * T) * norm.cdf(-d1)


def asian_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    averaging: str = "arithmetic",
    paths: int = 50_000,
    steps: int = 252,
    use_control_variate: bool = True,
    seed: int | None = None,
    q: float = 0.0,
) -> float:
    """Monte Carlo Asian option pricer.

    averaging: 'arithmetic' or 'geometric'.
    When arithmetic + use_control_variate=True, uses geometric Asian CF as control.
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    if averaging not in ("arithmetic", "geometric"):
        raise ValueError(f"averaging must be 'arithmetic' or 'geometric'")

    rng = np.random.default_rng(seed)
    path_arr = _simulate_gbm_paths(S, T, r, sigma, paths, steps, rng)  # (steps+1, paths)

    disc = math.exp(-r * T)
    sign = 1.0 if option_type == "call" else -1.0

    if averaging == "geometric":
        log_avg = np.log(path_arr[1:]).mean(axis=0)
        avg = np.exp(log_avg)
    else:
        avg = path_arr[1:].mean(axis=0)

    payoffs = np.maximum(sign * (avg - K), 0.0)
    raw_price = disc * float(np.mean(payoffs))

    if averaging == "arithmetic" and use_control_variate:
        # Geometric Asian on same paths as control variate
        log_avg_geo = np.log(path_arr[1:]).mean(axis=0)
        geo_avg = np.exp(log_avg_geo)
        geo_payoffs = np.maximum(sign * (geo_avg - K), 0.0)

        geo_cf_price = GeometricAsian().price(S, K, T, r, sigma, option_type, q=q)
        cov = np.cov(payoffs, geo_payoffs)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0
        raw_price = raw_price - beta * (disc * float(np.mean(geo_payoffs)) - geo_cf_price)

    return max(raw_price, 0.0)


def asian_running_avg_paths(
    S: float,
    T: float,
    r: float,
    sigma: float,
    n_display_paths: int = 20,
    steps: int = 252,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (time_grid, price_paths, avg_paths) for visualization.

    price_paths: (steps+1, n_display_paths)
    avg_paths:   (steps+1, n_display_paths) — cumulative arithmetic mean
    """
    rng = np.random.default_rng(seed)
    path_arr = _simulate_gbm_paths(S, T, r, sigma, n_display_paths, steps, rng)
    time_grid = np.linspace(0, T, steps + 1)

    cum_sum = np.cumsum(path_arr, axis=0)
    counts = np.arange(1, steps + 2)[:, None]
    avg_paths = cum_sum / counts

    return time_grid, path_arr, avg_paths

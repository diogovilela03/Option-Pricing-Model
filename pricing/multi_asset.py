"""Multi-asset option pricers via correlated GBM Monte Carlo.

Uses Cholesky decomposition of the correlation matrix to generate
correlated Brownian motions for each asset.

Products:
    BasketOption  — payoff on weighted average of asset prices
    WorstOfOption — payoff on the worst-performing asset
    RainbowOption — payoff on the best-performing asset
"""
import math

import numpy as np

from pricing.base import OptionType


def simulate_multi_asset_paths(
    spots: list[float],
    T: float,
    r: float,
    vols: list[float],
    corr_matrix: np.ndarray,
    paths: int,
    steps: int = 52,
    seed: int | None = None,
) -> np.ndarray:
    """Correlated multi-asset GBM paths via Cholesky decomposition.

    Returns shape (steps+1, n_assets, paths).
    """
    n_assets = len(spots)
    if corr_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"corr_matrix must be ({n_assets},{n_assets})")

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr_matrix)   # (n_assets, n_assets)

    dt = T / steps
    result = np.empty((steps + 1, n_assets, paths))

    log_S = np.array([[math.log(s)] * paths for s in spots], dtype=float)  # (n_assets, paths)
    result[0] = np.exp(log_S)

    drifts = np.array([(r - 0.5 * v ** 2) * dt for v in vols])          # (n_assets,)
    vols_dt = np.array([v * math.sqrt(dt) for v in vols])                # (n_assets,)

    for step in range(steps):
        Z = rng.standard_normal((n_assets, paths))     # (n_assets, paths)
        W = L @ Z                                       # (n_assets, paths) — correlated
        log_S += drifts[:, None] + vols_dt[:, None] * W
        result[step + 1] = np.exp(log_S)

    return result


class BasketOption:
    """European basket option: payoff on weighted average of terminal prices."""

    def price(
        self,
        spots: list[float],
        weights: list[float],
        K: float,
        T: float,
        r: float,
        vols: list[float],
        corr_matrix: np.ndarray,
        option_type: OptionType,
        paths: int = 50_000,
        steps: int = 52,
        seed: int | None = None,
    ) -> float:
        weights = np.array(weights)
        weights = weights / weights.sum()

        paths_arr = simulate_multi_asset_paths(spots, T, r, vols, corr_matrix, paths, steps, seed)
        terminal = paths_arr[-1]  # (n_assets, paths)
        basket = (weights[:, None] * terminal).sum(axis=0)  # (paths,)

        sign = 1.0 if option_type == "call" else -1.0
        payoffs = np.maximum(sign * (basket - K), 0.0)
        return max(math.exp(-r * T) * float(np.mean(payoffs)), 0.0)


class WorstOfOption:
    """Worst-of option: payoff on the minimum-performing asset (relative to spot)."""

    def price(
        self,
        spots: list[float],
        K_relative: float,
        T: float,
        r: float,
        vols: list[float],
        corr_matrix: np.ndarray,
        option_type: OptionType,
        paths: int = 50_000,
        steps: int = 52,
        seed: int | None = None,
    ) -> float:
        paths_arr = simulate_multi_asset_paths(spots, T, r, vols, corr_matrix, paths, steps, seed)
        terminal = paths_arr[-1]  # (n_assets, paths)
        spots_arr = np.array(spots)[:, None]
        perf = terminal / spots_arr   # relative performance (paths,) per asset

        worst_perf = perf.min(axis=0)  # (paths,) — worst performing asset

        sign = 1.0 if option_type == "call" else -1.0
        payoffs = np.maximum(sign * (worst_perf - K_relative), 0.0)
        return max(math.exp(-r * T) * float(np.mean(payoffs)), 0.0)


class RainbowOption:
    """Rainbow option: payoff on the best-performing asset (relative to spot)."""

    def price(
        self,
        spots: list[float],
        K_relative: float,
        T: float,
        r: float,
        vols: list[float],
        corr_matrix: np.ndarray,
        option_type: OptionType,
        paths: int = 50_000,
        steps: int = 52,
        seed: int | None = None,
    ) -> float:
        paths_arr = simulate_multi_asset_paths(spots, T, r, vols, corr_matrix, paths, steps, seed)
        terminal = paths_arr[-1]
        spots_arr = np.array(spots)[:, None]
        perf = terminal / spots_arr

        best_perf = perf.max(axis=0)

        sign = 1.0 if option_type == "call" else -1.0
        payoffs = np.maximum(sign * (best_perf - K_relative), 0.0)
        return max(math.exp(-r * T) * float(np.mean(payoffs)), 0.0)


def correlation_sensitivity(
    spots: list[float],
    K: float,
    T: float,
    r: float,
    vols: list[float],
    option_type: OptionType,
    product: str = "basket",
    rho_grid: np.ndarray | None = None,
    weights: list[float] | None = None,
    paths: int = 20_000,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute option price across a range of uniform correlations.

    Returns (rho_grid, prices). Assumes a 2-asset or uniform n-asset correlation.
    """
    if rho_grid is None:
        rho_grid = np.linspace(-0.9, 0.9, 30)

    n = len(spots)
    prices = []

    for rho in rho_grid:
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        try:
            np.linalg.cholesky(corr)  # check PSD
        except np.linalg.LinAlgError:
            prices.append(float("nan"))
            continue

        if product == "basket":
            w = weights or [1.0 / n] * n
            p = BasketOption().price(spots, w, K, T, r, vols, corr, option_type, paths, seed=seed)
        elif product == "worst-of":
            p = WorstOfOption().price(spots, K, T, r, vols, corr, option_type, paths, seed=seed)
        else:
            p = RainbowOption().price(spots, K, T, r, vols, corr, option_type, paths, seed=seed)
        prices.append(p)

    return rho_grid, np.array(prices)

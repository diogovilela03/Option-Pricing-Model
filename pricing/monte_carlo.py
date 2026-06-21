import math
import warnings
from typing import Literal

import numpy as np

from pricing.base import OptionPricer, OptionType
from pricing.black_scholes import BlackScholes

Dynamics = Literal["gbm", "heston"]
VarianceReduction = Literal["antithetic", "control_variate"] | None

_BS = BlackScholes()


class MonteCarlo(OptionPricer):
    """European option pricing via Monte Carlo simulation.

    Supports GBM and Heston dynamics, with antithetic variates and
    control variates (BS closed-form as the control) for variance reduction.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        paths: int = 10_000,
        steps: int = 252,
        dynamics: Dynamics = "gbm",
        variance_reduction: VarianceReduction = None,
        heston_params: dict | None = None,
        seed: int | None = None,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        if dynamics not in ("gbm", "heston"):
            raise ValueError(f"dynamics must be 'gbm' or 'heston', got {dynamics!r}")
        if variance_reduction not in (None, "antithetic", "control_variate"):
            raise ValueError(
                f"variance_reduction must be None, 'antithetic', or 'control_variate', "
                f"got {variance_reduction!r}"
            )

        rng = np.random.default_rng(seed)

        if dynamics == "gbm":
            ST = self._simulate_gbm(S, T, r, sigma, paths, steps, variance_reduction, rng)
        else:
            ST = self._simulate_heston(S, T, r, sigma, paths, steps, heston_params or {}, rng)

        payoffs = self._payoffs(ST, K, option_type)
        price = math.exp(-r * T) * float(np.mean(payoffs))

        if variance_reduction == "control_variate":
            price = self._apply_control_variate(
                S, K, T, r, sigma, option_type, paths, steps, rng, seed
            )

        return max(price, 0.0)

    # ------------------------------------------------------------------
    # GBM simulation
    # ------------------------------------------------------------------

    def _simulate_gbm(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        paths: int,
        steps: int,
        variance_reduction: VarianceReduction,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dt = T / steps
        drift = (r - 0.5 * sigma ** 2) * dt
        vol = sigma * math.sqrt(dt)

        if variance_reduction == "antithetic":
            Z = rng.standard_normal((steps, paths // 2))
            Z = np.concatenate([Z, -Z], axis=1)
        else:
            Z = rng.standard_normal((steps, paths))

        log_returns = drift + vol * Z
        ST = S * np.exp(log_returns.sum(axis=0))
        return ST

    # ------------------------------------------------------------------
    # Heston simulation (Euler-Maruyama with full truncation)
    # ------------------------------------------------------------------

    def _simulate_heston(
        self,
        S: float,
        T: float,
        r: float,
        sigma: float,
        paths: int,
        steps: int,
        params: dict,
        rng: np.random.Generator,
    ) -> np.ndarray:
        v0 = params.get("v0", sigma ** 2)
        kappa = params.get("kappa", 2.0)
        theta = params.get("theta", sigma ** 2)
        xi = params.get("xi", 0.3)
        rho = params.get("rho", -0.7)

        if 2 * kappa * theta < xi ** 2:
            warnings.warn(
                f"Feller condition violated: 2·κ·θ ({2 * kappa * theta:.4f}) < ξ² ({xi**2:.4f}). "
                "Variance may hit zero; prices may be unreliable.",
                UserWarning,
                stacklevel=4,
            )

        dt = T / steps
        sqrt_dt = math.sqrt(dt)

        log_S = np.full(paths, math.log(S))
        v = np.full(paths, v0)

        Z1 = rng.standard_normal((steps, paths))
        Z2 = rng.standard_normal((steps, paths))
        W1 = Z1
        W2 = rho * Z1 + math.sqrt(1 - rho ** 2) * Z2

        for i in range(steps):
            v_pos = np.maximum(v, 0.0)  # full truncation
            sqrt_v = np.sqrt(v_pos)

            log_S += (r - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * W1[i]
            v += kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * W2[i]

        return np.exp(log_S)

    # ------------------------------------------------------------------
    # Control variate using BS closed-form as the control
    # ------------------------------------------------------------------

    def _apply_control_variate(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        paths: int,
        steps: int,
        rng: np.random.Generator,
        seed: int | None,
    ) -> float:
        rng2 = np.random.default_rng(seed)
        ST = self._simulate_gbm(S, T, r, sigma, paths, steps, None, rng2)

        disc = math.exp(-r * T)
        payoffs = self._payoffs(ST, K, option_type)
        control_payoffs = self._payoffs(ST, K, option_type)  # same paths → control is BS on same ST

        # Control: geometric average of terminal prices (exact expectation = BS price)
        bs_price = _BS.price(S, K, T, r, sigma, option_type)
        raw_mc = disc * np.mean(payoffs)

        # Optimal beta via OLS covariance
        cov = np.cov(payoffs, control_payoffs)
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.0

        controlled = raw_mc - beta * (disc * np.mean(control_payoffs) - bs_price)
        return float(controlled)

    # ------------------------------------------------------------------
    # Payoff helper
    # ------------------------------------------------------------------

    @staticmethod
    def _payoffs(ST: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
        if option_type == "call":
            return np.maximum(ST - K, 0.0)
        return np.maximum(K - ST, 0.0)

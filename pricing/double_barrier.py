"""Double barrier option pricing.

Analytical: CRR binomial tree with barrier absorption at each node.
Monte Carlo: full GBM path simulation for validation.

Parity: DKI + DKO = Vanilla.
"""
import math
from typing import Literal

import numpy as np

from pricing.base import OptionPricer, OptionType
from pricing.asian import _simulate_gbm_paths
from pricing.black_scholes import BlackScholes

DoubleBarrierType = Literal["double-knock-out", "double-knock-in"]

_BS = BlackScholes()


class DoubleBarrierOption(OptionPricer):
    """CRR binomial tree pricer for double-barrier options."""

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        H_lower: float,
        H_upper: float,
        barrier_type: DoubleBarrierType = "double-knock-out",
        n_steps: int = 200,
        q: float = 0.0,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        if H_lower >= H_upper:
            raise ValueError(f"H_lower ({H_lower}) must be < H_upper ({H_upper})")
        if not (H_lower < S < H_upper):
            return 0.0

        dko = self._dko_price(S, K, T, r, sigma, q, H_lower, H_upper, n_steps, option_type)

        if barrier_type == "double-knock-out":
            return max(dko, 0.0)
        else:
            vanilla = _BS.price(S, K, T, r, sigma, option_type)
            return max(vanilla - dko, 0.0)

    def _dko_price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        q: float,
        L: float,
        U: float,
        n_steps: int,
        option_type: OptionType,
    ) -> float:
        """Double knock-out via CRR binomial tree with barrier absorption."""
        b = r - q
        dt = T / n_steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = max(0.0, min(1.0, (math.exp(b * dt) - d) / (u - d)))
        phi = 1 if option_type == "call" else -1

        # Terminal payoffs — knock out if outside corridor
        vals = []
        for j in range(n_steps + 1):
            s = S * u ** (n_steps - 2 * j)
            vals.append(max(phi * (s - K), 0.0) if L < s < U else 0.0)

        # Backward induction with barrier check
        for step in range(n_steps - 1, -1, -1):
            new_vals = []
            for j in range(step + 1):
                s = S * u ** (step - 2 * j)
                v = disc * (p * vals[j] + (1 - p) * vals[j + 1]) if L < s < U else 0.0
                new_vals.append(v)
            vals = new_vals

        return max(vals[0], 0.0)


def double_barrier_mc_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType,
    H_lower: float,
    H_upper: float,
    barrier_type: DoubleBarrierType = "double-knock-out",
    paths: int = 50_000,
    steps: int = 252,
    seed: int | None = None,
) -> float:
    """MC double barrier pricer via full GBM path simulation."""
    rng = np.random.default_rng(seed)
    path_arr = _simulate_gbm_paths(S, T, r, sigma, paths, steps, rng)

    sign = 1.0 if option_type == "call" else -1.0
    terminal = path_arr[-1]
    path_min = path_arr.min(axis=0)
    path_max = path_arr.max(axis=0)
    breached = (path_min <= H_lower) | (path_max >= H_upper)

    if barrier_type == "double-knock-out":
        payoffs = (~breached) * np.maximum(sign * (terminal - K), 0.0)
    else:
        payoffs = breached * np.maximum(sign * (terminal - K), 0.0)

    return max(math.exp(-r * T) * float(np.mean(payoffs)), 0.0)

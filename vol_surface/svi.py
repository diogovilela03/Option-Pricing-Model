"""
Raw-SVI (Gatheral) parameterization and per-expiry-slice fitting.

Total implied variance:
    w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))

where k = log(K/F) is log-moneyness and w = sigma_impl^2 * T.
"""
import numpy as np
from scipy.optimize import minimize


def svi_total_var(k: np.ndarray, params: dict) -> np.ndarray:
    """Evaluate SVI total variance at log-moneyness array k."""
    a, b, rho, m, sigma = (
        params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    )
    diff = k - m
    return a + b * (rho * diff + np.sqrt(diff ** 2 + sigma ** 2))


def fit_slice(
    log_moneyness: np.ndarray,
    total_var: np.ndarray,
    n_restarts: int = 5,
) -> dict:
    """Fit SVI to a single expiry slice via least-squares.

    Parameters
    ----------
    log_moneyness : log(K/F) for each strike
    total_var     : market total implied variance (sigma_impl^2 * T) per strike
    n_restarts    : number of random restarts to avoid local minima

    Returns
    -------
    dict with keys a, b, rho, m, sigma
    """
    def objective(x):
        a, b, rho, m, sigma = x
        params = {"a": a, "b": b, "rho": rho, "m": m, "sigma": sigma}
        residuals = svi_total_var(log_moneyness, params) - total_var
        return float(np.sum(residuals ** 2))

    # Constraints: b >= 0, |rho| < 1, sigma > 0, a + b*sigma*sqrt(1-rho^2) >= 0
    constraints = [
        {"type": "ineq", "fun": lambda x: x[1]},                          # b >= 0
        {"type": "ineq", "fun": lambda x: 1 - abs(x[2]) - 1e-6},          # |rho| < 1
        {"type": "ineq", "fun": lambda x: x[4] - 1e-6},                   # sigma > 0
        {"type": "ineq", "fun": lambda x: x[0] + x[1] * x[4] * np.sqrt(np.maximum(1 - x[2]**2, 0))},  # w >= 0
    ]

    best_result = None
    best_value = np.inf
    rng = np.random.default_rng(0)

    # Initial guess anchored to data
    w_mean = float(np.mean(total_var))
    seeds = [
        [w_mean * 0.5, 0.1, -0.3, 0.0, 0.2],
        [w_mean * 0.5, 0.05, 0.0, 0.0, 0.3],
        [w_mean * 0.3, 0.15, -0.5, 0.05, 0.15],
    ]
    for _ in range(n_restarts - len(seeds)):
        seeds.append([
            rng.uniform(0, w_mean),
            rng.uniform(0, 0.5),
            rng.uniform(-0.9, 0.9),
            rng.uniform(-0.3, 0.3),
            rng.uniform(0.01, 0.5),
        ])

    for x0 in seeds:
        result = minimize(
            objective,
            x0,
            method="SLSQP",
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if result.success and result.fun < best_value:
            best_value = result.fun
            best_result = result

    if best_result is None:
        raise RuntimeError("SVI fit failed to converge on all restarts.")

    a, b, rho, m, sigma = best_result.x
    return {"a": float(a), "b": float(b), "rho": float(rho), "m": float(m), "sigma": float(sigma)}

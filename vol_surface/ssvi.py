"""SSVI surface model — Gatheral-Jacquier (2014), power-law parameterization.

w(k; θ) = θ/2 × (1 + ρ·φ(θ)·k + √((φ(θ)·k + ρ)² + 1 − ρ²))
φ(θ)    = η / (θ^γ × (1+θ)^(1−γ))

Global params: rho ∈ (−1,1), eta > 0, gamma ∈ (0,1)
Per-expiry:   theta_t = ATM total variance (fixed from market data, not optimised)

Calendar-arb-free by construction when theta_t is non-decreasing in T.
Butterfly-arb-free sufficient condition: eta × (1 + |rho|) ≤ 4.
"""
import numpy as np
from scipy.optimize import minimize


def ssvi_total_var(
    k: np.ndarray,
    theta: float,
    rho: float,
    eta: float,
    gamma: float,
) -> np.ndarray:
    """Evaluate SSVI total variance at log-moneyness k for a single expiry slice."""
    phi = eta / (np.maximum(theta, 1e-8) ** gamma * (1 + theta) ** (1 - gamma))
    inner = phi * k + rho
    return theta / 2 * (1 + rho * phi * k + np.sqrt(np.maximum(inner ** 2 + 1 - rho ** 2, 0)))


def fit_ssvi(slices: list[dict], n_restarts: int = 10) -> dict:
    """Calibrate global SSVI parameters (rho, eta, gamma) across multiple expiry slices.

    Each slice dict must have:
        log_moneyness : np.ndarray   log(K / F)
        total_var     : np.ndarray   IV² × T from market
        theta         : float        ATM total variance for this expiry

    Returns dict: rho, eta, gamma, rmse
    """
    k_all     = np.concatenate([s["log_moneyness"] for s in slices])
    w_all     = np.concatenate([s["total_var"]     for s in slices])
    theta_rep = np.concatenate([
        np.full(len(s["log_moneyness"]), s["theta"]) for s in slices
    ])

    def objective(p: np.ndarray) -> float:
        rho, eta, gamma = p
        phi = eta / (np.maximum(theta_rep, 1e-8) ** gamma * (1 + theta_rep) ** (1 - gamma))
        inner = phi * k_all + rho
        w_hat = theta_rep / 2 * (1 + rho * phi * k_all + np.sqrt(np.maximum(inner ** 2 + 1 - rho ** 2, 0)))
        return float(np.mean((w_hat - w_all) ** 2))

    constraints = [
        {"type": "ineq", "fun": lambda p:  1 - abs(p[0]) - 1e-4},     # |rho| < 1
        {"type": "ineq", "fun": lambda p:  p[1] - 1e-4},               # eta > 0
        {"type": "ineq", "fun": lambda p:  p[2] - 1e-4},               # gamma > 0
        {"type": "ineq", "fun": lambda p:  1 - p[2] - 1e-4},           # gamma < 1
        {"type": "ineq", "fun": lambda p:  4 - p[1] * (1 + abs(p[0]))}, # butterfly-free
    ]

    rng = np.random.default_rng(42)
    best_res, best_val = None, np.inf

    for _ in range(n_restarts):
        x0 = [
            rng.uniform(-0.8, 0.8),
            rng.uniform(0.1, 2.0),
            rng.uniform(0.1, 0.9),
        ]
        try:
            res = minimize(
                objective, x0, method="SLSQP",
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if res.success and res.fun < best_val:
                best_val, best_res = res.fun, res
        except Exception:
            pass

    if best_res is None:
        raise RuntimeError("SSVI calibration failed for all restarts")

    rho, eta, gamma = best_res.x
    return {"rho": float(rho), "eta": float(eta), "gamma": float(gamma),
            "rmse": float(np.sqrt(best_val))}

"""
Arbitrage checks for SVI volatility surfaces.

Butterfly (within-slice): Gatheral-Jacquier density condition g(k) >= 0.
Calendar (across slices):  Total variance non-decreasing in T for each k.
"""
import numpy as np
from vol_surface.svi import svi_total_var


def check_butterfly(k: np.ndarray, params: dict, tol: float = -1e-6) -> dict:
    """Check no-butterfly-arbitrage (Gatheral-Jacquier) within a single slice.

    Computes g(k) = (1 - k*w'/(2w))^2 - (w')^2/4*(1/w + 1/4) + w''/2
    A slice is arbitrage-free iff g(k) >= 0 for all k.

    Returns
    -------
    dict with keys:
        arbitrage_free  : bool
        min_g           : float — minimum value of g across the grid
        violation_count : int   — number of grid points where g < 0
        g               : np.ndarray — g values across k
    """
    w, dw, ddw = _svi_derivatives(k, params)

    term1 = (1 - k * dw / (2 * w)) ** 2
    term2 = (dw ** 2 / 4) * (1 / w + 0.25)
    term3 = ddw / 2

    g = term1 - term2 + term3

    violations = int(np.sum(g < tol))
    return {
        "arbitrage_free": violations == 0,
        "min_g": float(np.min(g)),
        "violation_count": violations,
        "g": g,
    }


def check_calendar(k: np.ndarray, slices: list[dict], tol: float = -1e-6) -> dict:
    """Check no-calendar-spread-arbitrage across expiry slices.

    slices must be a list of dicts, each with keys 'T' and 'params',
    sorted by ascending T (or this function will sort them).

    Total variance w(k, T) = SVI(k; params) must be non-decreasing in T
    for every k. (Note: the SVI params already encode total variance, so
    no T multiplication is needed — the fit is done in total-variance space.)

    Returns
    -------
    dict with keys:
        arbitrage_free  : bool
        violation_count : int
        pair_results    : list of dicts, one per consecutive (T_i, T_{i+1}) pair
    """
    sorted_slices = sorted(slices, key=lambda s: s["T"])

    if len(sorted_slices) <= 1:
        return {"arbitrage_free": True, "violation_count": 0, "pair_results": []}

    pair_results = []
    total_violations = 0

    for i in range(len(sorted_slices) - 1):
        s1, s2 = sorted_slices[i], sorted_slices[i + 1]
        w1 = svi_total_var(k, s1["params"])
        w2 = svi_total_var(k, s2["params"])
        diff = w2 - w1  # must be >= 0 everywhere
        v_count = int(np.sum(diff < tol))
        total_violations += v_count
        pair_results.append({
            "T1": s1["T"],
            "T2": s2["T"],
            "arbitrage_free": v_count == 0,
            "violation_count": v_count,
            "min_diff": float(np.min(diff)),
        })

    return {
        "arbitrage_free": total_violations == 0,
        "violation_count": total_violations,
        "pair_results": pair_results,
    }


def _svi_derivatives(
    k: np.ndarray, params: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return w, w', w'' for the SVI parameterization."""
    a, b, rho, m, sigma = (
        params["a"], params["b"], params["rho"], params["m"], params["sigma"]
    )
    diff = k - m
    sq = np.sqrt(diff ** 2 + sigma ** 2)

    w = a + b * (rho * diff + sq)
    dw = b * (rho + diff / sq)
    ddw = b * sigma ** 2 / sq ** 3

    return w, dw, ddw

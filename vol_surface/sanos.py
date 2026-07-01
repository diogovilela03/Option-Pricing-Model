"""
SANOS: Smooth Strictly Arbitrage-Free Non-Parametric Option Surfaces.

Reference: Buehler et al. (2026), arXiv:2601.11209

Per-expiry LP fit — each slice is calibrated independently (no calendar
arbitrage enforcement across expiries in v1).

Pure price convention (Buehler et al.):
    F         = S * exp(r * T)        forward price
    DF        = exp(-r * T)           discount factor
    F * DF    = S                     (simplification used throughout)
    pure_strike  K = K_real / F
    pure_call    C = C_real / S

The fitted pure call price at any pure strike K is a mixture of
Black-Scholes kernels over a discrete density q:

    C_fit(K) = sum_j  BS(K / x_j, x_j * sigma_j * sqrtT) * x_j * q_j

where x_j are model strikes and sigma_j = vol_fac * atm_vol (constant,
default vol_fac=0.25). This is linear in q, making the problem a convex LP.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy.stats import norm

try:
    import cvxpy as cp
except ImportError as exc:
    raise ImportError(
        "cvxpy is required for SANOS. Install with: pip install cvxpy"
    ) from exc

from vol_surface.iv_inversion import implied_vol


# ---------------------------------------------------------------------------
# Black-Scholes kernel (pure price domain, spot = 1)
# ---------------------------------------------------------------------------

def _bs_call(k: np.ndarray, sqrtVar: np.ndarray) -> np.ndarray:
    """
    Vectorised BS call price with spot=1.

    k       : (...) array of strike / model_strike ratios
    sqrtVar : (...) array of sigma * sqrtT  (same shape or broadcastable)
    Returns : (...) array of call prices
    """
    intrinsic = np.maximum(1.0 - k, 0.0)
    sv = np.asarray(sqrtVar)
    mask = sv > 1e-10
    if not np.any(mask):
        return intrinsic
    d1 = np.where(mask, -np.log(np.maximum(k, 1e-12)) / np.where(mask, sv, 1.0) + 0.5 * sv, 0.0)
    d2 = d1 - sv
    bs = norm.cdf(d1) - k * norm.cdf(d2)
    return np.where(mask, bs, intrinsic)


def _kernel(K: np.ndarray, xstrikes: np.ndarray, xvols_sqrtT: np.ndarray) -> np.ndarray:
    """
    Kernel matrix of shape (len(K), len(xstrikes)).

    Entry [i, j] = BS_call(K[i]/x[j], xvols_sqrtT[j]) * x[j]

    The fitted call vector is kernel @ q.
    """
    K = np.asarray(K)
    k_ratio = K[:, None] / xstrikes[None, :]          # (M, N)
    sv      = xvols_sqrtT[None, :]                     # (1, N)
    return _bs_call(k_ratio, sv) * xstrikes[None, :]   # (M, N)


# ---------------------------------------------------------------------------
# Extended strike grid
# ---------------------------------------------------------------------------

def _build_xstrikes(
    strikes: np.ndarray,
    min_k: float,
    max_k: float,
    max_dx: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build model strike grid covering [min_k, max_k] with market strikes
    embedded and inter-market gaps filled to at most max_dx spacing.

    Returns (xstrikes, market_ix) where xstrikes[market_ix[i]] == strikes[i].
    """
    outer_dx = min(0.25, max_dx * 10.0)

    # left boundary -> first market strike
    left = strikes[0] - max_dx
    if left > min_k:
        n = max(2, 1 + math.floor((left - min_k) / outer_dx))
        out = np.linspace(min_k, left, n).tolist()
    else:
        out = [float(min_k)]

    mkt_ix = [len(out)]
    out.append(float(strikes[0]))

    for k in strikes[1:]:
        dx = k - out[-1]
        if dx > max_dx:
            n = max(1, int(dx / max_dx)) + 2
            for x in np.linspace(out[-1], k, n)[1:]:
                out.append(float(x))
            mkt_ix.append(len(out) - 1)
        else:
            mkt_ix.append(len(out))
            out.append(float(k))

    # last market strike -> right boundary
    right = strikes[-1] + max_dx
    if right < max_k:
        n = max(2, 1 + math.floor((max_k - right) / outer_dx))
        for x in np.linspace(right, max_k, n):
            out.append(float(x))
    else:
        out.append(float(max_k))

    xstrikes = np.array(out, dtype=np.float64)
    market_ix = np.array(mkt_ix, dtype=np.int32)

    assert np.all(np.abs(xstrikes[market_ix] - strikes) < 1e-8), \
        "Strike mapping error in _build_xstrikes"
    return xstrikes, market_ix


# ---------------------------------------------------------------------------
# Pure price conversion
# ---------------------------------------------------------------------------

def to_pure_prices(
    strikes_real: np.ndarray,
    prices_real: np.ndarray,
    option_types: np.ndarray,
    S: float,
    r: float,
    T: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert real option prices to pure call prices.

    Pure convention:
        pure_strike = K_real / F,   F = S * exp(r*T)
        pure_call   = C_real / S    (since F * DF = S)

    Puts are converted to calls via put-call parity in pure space:
        C_pure = P_pure + 1 - K_pure
    """
    F = S * math.exp(r * T)
    pure_K = strikes_real / F
    pure_P = prices_real / S

    is_put = np.asarray(option_types) == "put"
    pure_C = np.where(is_put, pure_P + 1.0 - pure_K, pure_P)
    return pure_K, pure_C


# ---------------------------------------------------------------------------
# Core per-slice fit
# ---------------------------------------------------------------------------

def fit_sanos_slice(
    strikes: np.ndarray,
    mids: np.ndarray,
    sqrtT: float,
    *,
    bids: np.ndarray | None = None,
    asks: np.ndarray | None = None,
    atm_vol: float | None = None,
    vol_fac: float = 0.25,
    floor_vol: float = 0.01,
    min_k: float | None = None,
    max_k: float | None = None,
    max_dx: float = 0.05,
    bid_ask_mode: str = "ignore",
    spread_weighted: bool = True,
    max_iweight: float = 100.0,
    solver: str = "CLARABEL",
) -> dict:
    """
    Fit SANOS to one expiry slice (pure price domain).

    Parameters
    ----------
    strikes : pure strikes (K/F), strictly increasing, must surround 1.0
    mids    : pure mid call prices (c/S)
    sqrtT   : sqrt of time to expiry in years
    bids, asks : pure bid/ask call prices (optional)
    atm_vol : ATM implied vol; estimated from mids at K=1 if None
    vol_fac : model vols = vol_fac * atm_vol  (controls smoothness)
    floor_vol : BS floor vol for the density floor constraint
    min_k, max_k : extended grid boundaries (pure strikes)
    max_dx  : maximum spacing between model strikes
    bid_ask_mode : 'ignore' | 'penalty' | 'constraint'
    spread_weighted : weight objective by 1/spread
    solver  : cvxpy solver name

    Returns
    -------
    dict with keys: xstrikes, xdensity, xvols, fitted_calls, fitted_vols,
                    market_vols, atm_vol, atm_call, rmse, status
    """
    strikes = np.asarray(strikes, dtype=np.float64)
    mids    = np.asarray(mids,    dtype=np.float64)

    if len(strikes) < 3:
        raise ValueError("Need at least 3 strikes per slice")
    if not (strikes[0] < 1.0 < strikes[-1]):
        raise ValueError("Pure strikes must straddle 1.0 (ATM)")
    if not np.all(np.diff(strikes) > 0):
        raise ValueError("Strikes must be strictly increasing")

    T = sqrtT ** 2
    # Default model grid: ±10% around market strike range, keeping the
    # grid roughly symmetric around 1.0 so the martingale condition
    # (mean of density = 1) doesn't force density into the tails.
    if min_k is None:
        min_k = max(strikes[0] - 0.05, strikes[0] * 0.90)
    if max_k is None:
        max_k = min(strikes[-1] + 0.05, strikes[-1] * 1.10)

    # ATM vol estimate (interpolate mids at K=1, use BS ATM approximation)
    if atm_vol is None:
        atm_mid = float(np.interp(1.0, strikes, mids))
        # BS ATM: C ≈ sigma * sqrtT / sqrt(2*pi)
        atm_vol = max(atm_mid * math.sqrt(2 * math.pi) / sqrtT, floor_vol)

    # Extended model strike grid
    xstrikes, market_ix = _build_xstrikes(strikes, min_k, max_k, max_dx)
    N = len(xstrikes)

    xvols       = np.full(N, vol_fac * atm_vol)
    xvols_sqrtT = xvols * sqrtT

    # Kernel matrices (linear in q)
    K_mkt   = _kernel(strikes,  xstrikes, xvols_sqrtT)  # (M, N)
    K_model = _kernel(xstrikes, xstrikes, xvols_sqrtT)  # (N, N)

    # cvxpy variable and fitted call expressions
    q       = cp.Variable(N)
    C_fit   = K_mkt   @ q   # (M,)  at market strikes
    C_model = K_model @ q   # (N,)  at all model strikes

    # Floor: minimum call price at each model strike.
    # In pure space (spot=1), the lower bound on a call at strike x_j is:
    #   BS_call(spot=1, strike=x_j, vol=floor_vol)
    #   ≈ max(1 - x_j, 0) for x_j < 1 (intrinsic floor for ITM calls)
    #   ≈ 0                for x_j > 1 (OTM calls have no intrinsic floor)
    # Buehler et al. use the same formula. We must NOT compute floor as
    # BS_call(1.0, …) * x_j (which treats every model strike as ATM and forces
    # excessive density at the far tail, inflating all fitted call prices).
    floor = _bs_call(xstrikes, np.full(N, sqrtT * floor_vol))

    constraints = [
        q >= 0.0,
        cp.sum(q) == 1.0,
        q @ xstrikes == 1.0,   # martingale: E[pure strike] = 1
        C_model >= floor,      # butterfly no-arb floor
    ]

    if bid_ask_mode == "constraint" and bids is not None and asks is not None:
        constraints += [C_fit >= bids, C_fit <= asks]

    # Objective weights
    if spread_weighted and bids is not None and asks is not None:
        spreads = np.maximum(asks - bids, 1e-8)
        w = np.minimum(max_iweight, 1.0 / spreads)
        w /= w.sum()
    else:
        w = np.ones(len(strikes)) / len(strikes)

    # Objective: weighted L1 distance to mid prices
    err = cp.multiply(w, cp.abs(C_fit - mids))
    if bid_ask_mode == "penalty" and bids is not None and asks is not None:
        err = (
            err
            + cp.multiply(w, cp.maximum(0.0, C_fit - asks))
            + cp.multiply(w, cp.maximum(0.0, bids - C_fit))
        )
    objective = cp.Minimize(cp.sum(err))

    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=solver)
    except cp.SolverError as exc:
        raise RuntimeError(f"SANOS solver error: {exc}") from exc

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"SANOS: solver status '{prob.status}'")
    if prob.status == "optimal_inaccurate":
        warnings.warn("SANOS: 'optimal_inaccurate' — results may be unreliable", UserWarning)

    density = np.maximum(0.0, np.array(q.value, dtype=np.float64))

    # Fitted calls and vols at market strikes
    fitted_calls = K_mkt @ density

    # ATM fitted call
    K_atm    = _kernel(np.array([1.0]), xstrikes, xvols_sqrtT)
    atm_call = float((K_atm @ density)[0])

    # IV back-out: pure space -> implied_vol(S=1, K=K_pure, T=T, r=0, price)
    fitted_vols = _pure_calls_to_vols(strikes, fitted_calls, T)
    market_vols = _pure_calls_to_vols(strikes, mids, T)

    valid = np.isfinite(fitted_vols) & np.isfinite(market_vols)
    rmse = float(np.sqrt(np.mean((fitted_vols[valid] - market_vols[valid]) ** 2))) \
           if valid.any() else float("nan")

    return {
        "xstrikes":    xstrikes,
        "xdensity":    density,
        "xvols":       xvols,
        "fitted_calls": np.asarray(fitted_calls),
        "fitted_vols":  fitted_vols,
        "market_vols":  market_vols,
        "atm_vol":     float(atm_vol),
        "atm_call":    atm_call,
        "rmse":        rmse,
        "status":      prob.status,
    }


def _pure_calls_to_vols(
    pure_strikes: np.ndarray,
    pure_calls: np.ndarray,
    T: float,
) -> np.ndarray:
    """Back out implied vols from pure call prices (spot=1, r=0)."""
    vols = np.full(len(pure_strikes), np.nan)
    for i, (K, C) in enumerate(zip(pure_strikes, pure_calls)):
        try:
            vols[i] = implied_vol(1.0, float(K), T, 0.0, float(C), "call")
        except Exception:
            pass
    return vols


# ---------------------------------------------------------------------------
# Surface fit (per-slice, independent)
# ---------------------------------------------------------------------------

def fit_sanos_surface(
    slices: list[dict],
    S: float,
    r: float,
    **slice_kwargs,
) -> list[dict]:
    """
    Fit SANOS independently to each expiry slice.

    Parameters
    ----------
    slices : list of dicts, each with:
        expiry     : str (YYYY-MM-DD)
        T          : float  (time to expiry in years)
        strikes    : np.ndarray  (real strikes $)
        mids       : np.ndarray  (real mid call prices $, mixed calls/puts ok)
        option_types: np.ndarray (str array, 'call'/'put' per row)
        bids, asks : np.ndarray | None  (real prices, optional)
    S : spot price
    r : risk-free rate
    **slice_kwargs : forwarded to fit_sanos_slice

    Returns
    -------
    List of result dicts (one per expiry that succeeded), each with an
    added 'expiry', 'T', and 'pure_strikes' key.
    """
    # Extract surface-level kwargs before the loop so the dict isn't mutated
    # on the first iteration and then empty for subsequent slices.
    lo_k             = slice_kwargs.pop("min_moneyness",  0.0)
    hi_k             = slice_kwargs.pop("max_moneyness",  10.0)
    min_time_value   = slice_kwargs.pop("min_time_value", 5e-5)  # pure space

    results = []
    for sl in slices:
        T     = sl["T"]
        sqrtT = math.sqrt(max(T, 1e-4))

        pure_K, pure_C = to_pure_prices(
            sl["strikes"], sl["mids"], sl["option_types"], S, r, T
        )

        pure_bids = pure_asks = None
        if sl.get("bids") is not None and sl.get("asks") is not None:
            _, pure_bids = to_pure_prices(sl["bids"], sl["bids"], sl["option_types"], S, r, T)
            _, pure_asks = to_pure_prices(sl["asks"], sl["asks"], sl["option_types"], S, r, T)

        # Sort by pure strike
        order   = np.argsort(pure_K)
        pure_K  = pure_K[order]
        pure_C  = pure_C[order]
        if pure_bids is not None:
            pure_bids = pure_bids[order]
            pure_asks = pure_asks[order]

        # Deduplicate (same strike may have both call and put; take mean)
        unique_K, inv = np.unique(pure_K.round(8), return_inverse=True)
        unique_C = np.array([pure_C[inv == i].mean() for i in range(len(unique_K))])
        if pure_bids is not None:
            unique_bids = np.array([pure_bids[inv == i].mean() for i in range(len(unique_K))])
            unique_asks = np.array([pure_asks[inv == i].mean() for i in range(len(unique_K))])
        else:
            unique_bids = unique_asks = None

        # Moneyness filter: liquid region around ATM only.
        mask = (unique_K >= lo_k) & (unique_K <= hi_k)
        unique_K = unique_K[mask]
        unique_C = unique_C[mask]
        if unique_bids is not None:
            unique_bids = unique_bids[mask]
            unique_asks = unique_asks[mask]

        # Time-value filter: drop near-intrinsic prices (pure call price must
        # exceed intrinsic by at least min_time_value). Near-intrinsic prices
        # carry no vol information and distort LP weights.
        intrinsic = np.maximum(1.0 - unique_K, 0.0)
        tv_mask   = (unique_C - intrinsic) >= min_time_value
        unique_K  = unique_K[tv_mask]
        unique_C  = unique_C[tv_mask]
        if unique_bids is not None:
            unique_bids = unique_bids[tv_mask]
            unique_asks = unique_asks[tv_mask]

        if len(unique_K) < 3 or not (unique_K[0] < 1.0 < unique_K[-1]):
            warnings.warn(
                f"Expiry {sl['expiry']}: fewer than 3 valid strikes after filters — skipping"
            )
            continue

        try:
            result = fit_sanos_slice(
                unique_K, unique_C, sqrtT,
                bids=unique_bids, asks=unique_asks,
                **slice_kwargs,
            )
            result["expiry"]       = sl["expiry"]
            result["T"]            = T
            result["pure_strikes"] = unique_K
            results.append(result)
        except Exception as exc:
            warnings.warn(f"Expiry {sl['expiry']}: SANOS failed — {exc}")

    return results

import math
from scipy.optimize import brentq
from pricing.black_scholes import BlackScholes
from pricing.base import OptionType

_BS = BlackScholes()

_MAX_ITER = 100
_TOL = 1e-8
_SIGMA_LOW = 1e-6
_SIGMA_HIGH = 10.0


def implied_vol(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: OptionType,
) -> float:
    """Return the implied volatility that reprices market_price under BS.

    Uses Newton-Raphson with a Brent's method fallback.
    Raises ValueError if no solution exists (price below intrinsic, zero price, bad type).
    """
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

    disc = math.exp(-r * T)
    if option_type == "call":
        intrinsic = max(S - K * disc, 0.0)
    else:
        intrinsic = max(K * disc - S, 0.0)

    if market_price <= 0.0:
        raise ValueError(f"market_price must be positive, got {market_price}")
    if market_price < intrinsic:
        raise ValueError(
            f"market_price {market_price:.4f} is below intrinsic value {intrinsic:.4f}; "
            "no valid implied vol exists."
        )

    sigma = _newton_raphson(S, K, T, r, market_price, option_type)
    if sigma is not None:
        return sigma

    # Fallback: Brent's method on [_SIGMA_LOW, _SIGMA_HIGH]
    def objective(s: float) -> float:
        return _BS.price(S, K, T, r, s, option_type) - market_price

    try:
        return brentq(objective, _SIGMA_LOW, _SIGMA_HIGH, xtol=_TOL, maxiter=_MAX_ITER)
    except ValueError:
        raise ValueError(
            f"Could not find implied vol for market_price={market_price:.4f}, "
            f"S={S}, K={K}, T={T}, r={r}, option_type={option_type!r}"
        )


def _newton_raphson(
    S: float,
    K: float,
    T: float,
    r: float,
    market_price: float,
    option_type: OptionType,
) -> float | None:
    sigma = _initial_guess(S, K, T, market_price)

    for _ in range(_MAX_ITER):
        price = _BS.price(S, K, T, r, sigma, option_type)
        # vega in price units per unit vol (undo the /100 scaling used for display)
        vega = _BS.greeks(S, K, T, r, sigma, option_type)["vega"] * 100.0

        if abs(vega) < 1e-12:
            return None

        diff = price - market_price
        sigma -= diff / vega

        if sigma <= 0:
            return None
        if abs(diff) < _TOL:
            return sigma

    return None


def _initial_guess(S: float, K: float, T: float, market_price: float) -> float:
    # Brenner-Subrahmanyam ATM approximation: sigma ≈ price * sqrt(2π/T) / S
    guess = market_price * math.sqrt(2 * math.pi / T) / S
    return max(min(guess, _SIGMA_HIGH - 0.01), _SIGMA_LOW + 0.01)

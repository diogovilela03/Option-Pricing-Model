"""Digital (binary) option pricing via Black-Scholes closed-form.

Vanilla decomposition:
    vanilla_call = asset_or_nothing_call − K·e^{-rT}·cash_or_nothing_call
"""
import math

from scipy.stats import norm

from pricing.base import OptionPricer, OptionType
from pricing.black_scholes import BlackScholes

_BS = BlackScholes()


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


class DigitalOption(OptionPricer):
    """Cash-or-nothing and asset-or-nothing options (Black-Scholes closed-form).

    Vanilla decomposition identity:
        vanilla_call = asset_or_nothing_call − K·e^{-rT}·cash_or_nothing_call
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        digital_type: str = "cash-or-nothing",
        notional: float = 1.0,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
        if digital_type not in ("cash-or-nothing", "asset-or-nothing"):
            raise ValueError(f"digital_type must be 'cash-or-nothing' or 'asset-or-nothing'")

        d1, d2 = _d1_d2(S, K, T, r, sigma)
        disc = math.exp(-r * T)

        if digital_type == "cash-or-nothing":
            if option_type == "call":
                return notional * disc * norm.cdf(d2)
            return notional * disc * norm.cdf(-d2)
        else:  # asset-or-nothing
            if option_type == "call":
                return notional * S * norm.cdf(d1)
            return notional * S * norm.cdf(-d1)

    def delta(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        digital_type: str = "cash-or-nothing",
    ) -> float:
        """Analytical delta. Diverges as T→0 at S=K (model risk)."""
        d1, d2 = _d1_d2(S, K, T, r, sigma)
        sT = sigma * math.sqrt(T)
        disc = math.exp(-r * T)
        sign = 1.0 if option_type == "call" else -1.0

        if digital_type == "cash-or-nothing":
            return sign * disc * norm.pdf(d2) / (S * sT)
        else:
            return sign * norm.pdf(d1) / (S * sT) + norm.cdf(sign * d1)


def call_spread_approximation(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    dK: float,
) -> float:
    """Replicate a cash-or-nothing digital call via a call spread of width dK.

    As dK → 0 this converges to the closed-form digital call price.
    """
    return (
        _BS.price(S, K - dK / 2, T, r, sigma, "call")
        - _BS.price(S, K + dK / 2, T, r, sigma, "call")
    ) / dK

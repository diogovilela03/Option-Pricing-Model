"""Quanto option pricing via modified Black-Scholes.

A quanto option pays in domestic currency but the underlying moves in foreign currency.
The correlation between spot and FX rate introduces a drift adjustment.

Adjusted drift: b = r_d - r_f - rho * sigma_S * sigma_FX
"""
import math

from scipy.stats import norm

from pricing.base import OptionPricer, OptionType


class QuantoOption(OptionPricer):
    """Quanto option pricer (modified BS with FX-adjusted drift).

    Parameters (beyond standard S, K, T, sigma, option_type):
        r_d      : domestic risk-free rate
        r_f      : foreign risk-free rate
        sigma_S  : volatility of the underlying in foreign currency
        sigma_FX : volatility of the FX rate (foreign/domestic)
        rho      : correlation between underlying and FX rate
        Q0       : current FX rate (domestic per foreign), default 1.0
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        r_d: float | None = None,
        r_f: float | None = None,
        sigma_S: float | None = None,
        sigma_FX: float = 0.10,
        rho: float = -0.30,
        Q0: float = 1.0,
        **kwargs,
    ) -> float:
        """r and sigma are aliases for r_d and sigma_S for ABC compatibility."""
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        r_d = r_d if r_d is not None else r
        sigma_S = sigma_S if sigma_S is not None else sigma
        r_f = r_f if r_f is not None else 0.0

        b = r_d - r_f - rho * sigma_S * sigma_FX
        sT = sigma_S * math.sqrt(T)

        if sT < 1e-12:
            intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
            return Q0 * math.exp(-r_d * T) * intrinsic

        d1 = (math.log(S / K) + (b + 0.5 * sigma_S ** 2) * T) / sT
        d2 = d1 - sT
        disc = math.exp(-r_d * T)
        sign = 1 if option_type == "call" else -1

        price = Q0 * (
            sign * S * math.exp((b - r_d) * T) * norm.cdf(sign * d1)
            - sign * K * disc * norm.cdf(sign * d2)
        )
        return max(price, 0.0)

    def decompose(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        r_f: float = 0.0,
        sigma_FX: float = 0.10,
        rho: float = -0.30,
        Q0: float = 1.0,
        **kwargs,
    ) -> list[dict]:
        """Show the FX drift adjustment and its effect on the price."""
        from pricing.black_scholes import BlackScholes
        bs_price = BlackScholes().price(S, K, T, r, sigma, option_type)
        quanto_price = self.price(S, K, T, r, sigma, option_type,
                                  r_d=r, r_f=r_f, sigma_S=sigma,
                                  sigma_FX=sigma_FX, rho=rho, Q0=Q0)
        drift_adj = rho * sigma * sigma_FX
        return [
            {"component": f"Vanilla {option_type.capitalize()}",
             "formula": "Standard BS",
             "value": bs_price},
            {"component": "Drift adjustment",
             "formula": f"−ρ·σ_S·σ_FX = −{rho:.2f}×{sigma:.2f}×{sigma_FX:.2f} = {-drift_adj:.4f}",
             "value": quanto_price - bs_price},
            {"component": f"Quanto {option_type.capitalize()}",
             "formula": f"b = r_d − r_f − ρσ_Sσ_FX",
             "value": quanto_price},
        ]

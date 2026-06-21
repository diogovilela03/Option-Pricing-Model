import math
from scipy.stats import norm
from pricing.base import OptionPricer, OptionType


class BlackScholes(OptionPricer):
    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        d1, d2 = self._d1_d2(S, K, T, r, sigma)

        if option_type == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def greeks(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> dict[str, float]:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        disc = math.exp(-r * T)
        phi_d1 = norm.pdf(d1)

        sign = 1.0 if option_type == "call" else -1.0

        delta = sign * norm.cdf(sign * d1)
        gamma = phi_d1 / (S * sigma * math.sqrt(T))
        vega = S * phi_d1 * math.sqrt(T) / 100.0  # per 1% move in vol
        theta = (
            -(S * phi_d1 * sigma) / (2.0 * math.sqrt(T))
            - sign * r * K * disc * norm.cdf(sign * d2)
        ) / 365.0  # per calendar day
        rho = sign * K * T * disc * norm.cdf(sign * d2) / 100.0  # per 1% move in r

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

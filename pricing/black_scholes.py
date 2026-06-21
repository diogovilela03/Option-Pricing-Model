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

        sqrt_T = math.sqrt(T)
        raw_vega = S * phi_d1 * sqrt_T  # ∂V/∂σ

        delta = sign * norm.cdf(sign * d1)
        gamma = phi_d1 / (S * sigma * sqrt_T)
        vega  = raw_vega / 100.0                                          # per 1% σ
        theta = (
            -(S * phi_d1 * sigma) / (2.0 * sqrt_T)
            - sign * r * K * disc * norm.cdf(sign * d2)
        ) / 365.0                                                          # per calendar day
        rho   = sign * K * T * disc * norm.cdf(sign * d2) / 100.0        # per 1% r

        # Second-order Greeks
        volga = raw_vega * d1 * d2 / (sigma * 100.0)                     # ∂²V/∂σ², per 1% σ
        vanna = -phi_d1 * d2 / (sigma * 100.0)                           # ∂²V/∂S∂σ, per 1% σ
        # ∂Delta/∂T_remaining per calendar day (positive → delta grows with more time)
        charm = phi_d1 * (2 * r * T - d2 * sigma * sqrt_T) / (2 * T * sigma * sqrt_T) / 365.0

        return {
            "delta": delta, "gamma": gamma, "vega": vega,
            "theta": theta, "rho": rho,
            "volga": volga, "vanna": vanna, "charm": charm,
        }

    @staticmethod
    def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2

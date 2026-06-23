import math
import numpy as np
from pricing.base import OptionPricer, OptionType


class BinomialTree(OptionPricer):
    """Cox-Ross-Rubinstein binomial tree, American exercise only.

    American-only by design: the value of this engine is demonstrating
    the early-exercise premium relative to the European BS baseline.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
        steps: int = 200,
        **kwargs,
    ) -> float:
        if option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")

        dt = T / steps
        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp(r * dt) - d) / (u - d)  # risk-neutral up probability

        # Terminal stock prices
        j = np.arange(steps + 1)
        ST = S * u ** (steps - j) * d ** j

        # Terminal option values
        if option_type == "call":
            values = np.maximum(ST - K, 0.0)
        else:
            values = np.maximum(K - ST, 0.0)

        # Backward induction with early-exercise check at each node
        for i in range(steps - 1, -1, -1):
            j = np.arange(i + 1)
            S_node = S * u ** (i - j) * d ** j
            continuation = disc * (p * values[:i + 1] + (1 - p) * values[1:i + 2])
            if option_type == "call":
                intrinsic = np.maximum(S_node - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_node, 0.0)
            values = np.maximum(continuation, intrinsic)

        return float(values[0])

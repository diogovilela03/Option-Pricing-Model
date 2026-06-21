from abc import ABC, abstractmethod
from typing import Literal


OptionType = Literal["call", "put"]


class OptionPricer(ABC):
    @abstractmethod
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
        """Return the option price.

        S: spot price
        K: strike price
        T: time to maturity in years
        r: continuously compounded risk-free rate
        sigma: volatility (annualised)
        option_type: 'call' or 'put'
        """

"""Structured product pricers with analytical decomposition.

Each class exposes .price() returning a dict and .decompose() returning
a list of building blocks, matching the "show decomposition" UI pattern.

Products:
    ReverseConvertible       — Bond + Short Put
    BarrierReverseConvertible— Bond + Short Down-and-In Put
    DiscountCertificate      — Long Stock − Short Call (covered call)
    BonusCertificate         — Long Stock + Long Down-and-Out Put
    AirbagCertificate        — Partial capital protection + capped participation
    TwinWinCertificate       — Benefits from both up and down moves until barrier
"""
import math

from pricing.black_scholes import BlackScholes
from pricing.barrier import BarrierOption

_BS = BlackScholes()
_Barrier = BarrierOption()


def _bond(notional: float, coupon_rate: float, T: float, r: float) -> float:
    """Present value of a zero-coupon bond paying notional*(1+coupon_rate) at T."""
    return notional * (1 + coupon_rate) * math.exp(-r * T)


class ReverseConvertible:
    """Reverse Convertible = Bond + Short Put (at K ≈ S0).

    Investor sells downside risk in exchange for enhanced coupon.
    At maturity:
        If S_T >= K: receive notional + coupon
        If S_T <  K: receive (S_T/K) * notional (capital loss)
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        coupon_rate: float,
        notional: float = 1000.0,
    ) -> dict:
        bond_val = _bond(notional, coupon_rate, T, r)
        short_put = -_BS.price(S, K, T, r, sigma, "put") * (notional / K)
        total = bond_val + short_put
        return {
            "price": total,
            "bond": bond_val,
            "short_put": short_put,
            "yield": coupon_rate,
            "notional": notional,
        }

    def decompose(self, S, K, T, r, sigma, coupon_rate, notional=1000.0) -> list[dict]:
        d = self.price(S, K, T, r, sigma, coupon_rate, notional)
        return [
            {"component": f"Bond (notional × (1+c)·e^{{-rT}})",
             "value": d["bond"], "pct": d["bond"] / notional * 100},
            {"component": f"Short Put (K={K:.0f})",
             "value": d["short_put"], "pct": d["short_put"] / notional * 100},
            {"component": "RC Price",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class BarrierReverseConvertible:
    """Barrier Reverse Convertible = Bond + Short Down-and-In Put.

    Capital loss only if S touches barrier H during the life of the product.
    Higher coupon than plain RC due to extra protection (barrier condition).
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        coupon_rate: float,
        barrier: float,
        notional: float = 1000.0,
    ) -> dict:
        bond_val = _bond(notional, coupon_rate, T, r)
        di_put = _Barrier.price(S, K, T, r, sigma, "put", barrier, "down-and-in")
        short_di_put = -di_put * (notional / K)
        total = bond_val + short_di_put
        return {
            "price": total,
            "bond": bond_val,
            "short_di_put": short_di_put,
            "yield": coupon_rate,
            "barrier": barrier,
            "notional": notional,
        }

    def decompose(self, S, K, T, r, sigma, coupon_rate, barrier, notional=1000.0) -> list[dict]:
        d = self.price(S, K, T, r, sigma, coupon_rate, barrier, notional)
        return [
            {"component": f"Bond (notional × (1+c)·e^{{-rT}})",
             "value": d["bond"], "pct": d["bond"] / notional * 100},
            {"component": f"Short Down-and-In Put (K={K:.0f}, H={barrier:.0f})",
             "value": d["short_di_put"], "pct": d["short_di_put"] / notional * 100},
            {"component": "BRC Price",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class DiscountCertificate:
    """Discount Certificate = Long Stock − Short Call (covered call).

    Investor buys the stock at a discount (cap level = strike).
    Upside is capped at K; downside is 1-for-1 below purchase price.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        notional: float = 1000.0,
    ) -> dict:
        n_shares = notional / S
        long_stock = n_shares * S * math.exp(-0 * T)  # undiscounted stock position
        short_call = -n_shares * _BS.price(S, K, T, r, sigma, "call")
        total = long_stock + short_call
        discount_pct = (S - total / n_shares) / S * 100
        return {
            "price": total,
            "long_stock": long_stock,
            "short_call": short_call,
            "cap_level": K,
            "discount_pct": discount_pct,
            "notional": notional,
        }

    def decompose(self, S, K, T, r, sigma, notional=1000.0) -> list[dict]:
        d = self.price(S, K, T, r, sigma, notional)
        return [
            {"component": f"Long Stock ({notional/S:.2f} shares @ {S:.2f})",
             "value": d["long_stock"], "pct": d["long_stock"] / notional * 100},
            {"component": f"Short Call (cap K={K:.0f})",
             "value": d["short_call"], "pct": d["short_call"] / notional * 100},
            {"component": f"Discount Certificate (discount {d['discount_pct']:.1f}%)",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class BonusCertificate:
    """Bonus Certificate = Long Stock + Long Down-and-Out Put.

    If barrier not hit: receive max(S_T, K_bonus) — floors return at bonus level.
    If barrier is hit: loses put protection, behaves like long stock.
    """

    def price(
        self,
        S: float,
        K_bonus: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        notional: float = 1000.0,
    ) -> dict:
        n_shares = notional / S
        long_stock = n_shares * S
        long_dop = n_shares * _Barrier.price(S, K_bonus, T, r, sigma, "put", barrier, "down-and-out")
        total = long_stock + long_dop
        return {
            "price": total,
            "long_stock": long_stock,
            "long_dop": long_dop,
            "bonus_level": K_bonus,
            "barrier": barrier,
            "notional": notional,
        }

    def decompose(self, S, K_bonus, T, r, sigma, barrier, notional=1000.0) -> list[dict]:
        d = self.price(S, K_bonus, T, r, sigma, barrier, notional)
        return [
            {"component": f"Long Stock ({notional/S:.2f} shares @ {S:.2f})",
             "value": d["long_stock"], "pct": d["long_stock"] / notional * 100},
            {"component": f"Long Down-and-Out Put (bonus={K_bonus:.0f}, H={barrier:.0f})",
             "value": d["long_dop"], "pct": d["long_dop"] / notional * 100},
            {"component": "Bonus Certificate",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class AirbagCertificate:
    """Airbag Certificate = Long Stock + Long Put (floor) − Short Call (cap).

    Provides partial downside protection (airbag) within the floor level.
    Above cap: participation is capped. Below floor: loss reduced by airbag factor.
    """

    def price(
        self,
        S: float,
        K_cap: float,
        K_floor: float,
        T: float,
        r: float,
        sigma: float,
        participation: float = 1.0,
        notional: float = 1000.0,
    ) -> dict:
        n_shares = notional / S
        long_stock = n_shares * S
        long_put = n_shares * participation * _BS.price(S, K_floor, T, r, sigma, "put")
        short_call = -n_shares * _BS.price(S, K_cap, T, r, sigma, "call")
        total = long_stock + long_put + short_call
        return {
            "price": total,
            "long_stock": long_stock,
            "long_put": long_put,
            "short_call": short_call,
            "cap_level": K_cap,
            "floor_level": K_floor,
            "participation": participation,
            "notional": notional,
        }

    def decompose(self, S, K_cap, K_floor, T, r, sigma, participation=1.0, notional=1000.0) -> list[dict]:
        d = self.price(S, K_cap, K_floor, T, r, sigma, participation, notional)
        return [
            {"component": f"Long Stock ({notional/S:.2f} shares @ {S:.2f})",
             "value": d["long_stock"], "pct": d["long_stock"] / notional * 100},
            {"component": f"Long Put floor (K={K_floor:.0f}, {participation:.0%} participation)",
             "value": d["long_put"], "pct": d["long_put"] / notional * 100},
            {"component": f"Short Call cap (K={K_cap:.0f})",
             "value": d["short_call"], "pct": d["short_call"] / notional * 100},
            {"component": "Airbag Certificate",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class TwinWinCertificate:
    """Twin-Win Certificate = Long Stock + Long Down-and-Out Put (convert put to positive return).

    Profits from both upward and downward moves — as long as barrier is not touched.
    If barrier is hit: loses twin-win, behaves like long stock.
    Payoff: S + |S - K| if barrier not touched, else S.
    """

    def price(
        self,
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        barrier: float,
        notional: float = 1000.0,
    ) -> dict:
        n_shares = notional / S
        long_stock = n_shares * S
        # Twin-win: add 2× knock-out put (the put mirrors downside to upside)
        twin_put = 2 * n_shares * _Barrier.price(S, K, T, r, sigma, "put", barrier, "down-and-out")
        total = long_stock + twin_put
        return {
            "price": total,
            "long_stock": long_stock,
            "twin_put": twin_put,
            "barrier": barrier,
            "notional": notional,
        }

    def decompose(self, S, K, T, r, sigma, barrier, notional=1000.0) -> list[dict]:
        d = self.price(S, K, T, r, sigma, barrier, notional)
        return [
            {"component": f"Long Stock ({notional/S:.2f} shares @ {S:.2f})",
             "value": d["long_stock"], "pct": d["long_stock"] / notional * 100},
            {"component": f"2× Long Down-and-Out Put (K={K:.0f}, H={barrier:.0f})",
             "value": d["twin_put"], "pct": d["twin_put"] / notional * 100},
            {"component": "Twin-Win Certificate",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]

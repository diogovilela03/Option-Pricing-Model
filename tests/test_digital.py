"""Tests for digital option pricing (cash-or-nothing, asset-or-nothing)."""
import math
import pytest
from scipy.stats import norm

from pricing.digital import DigitalOption, call_spread_approximation
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
_dig = DigitalOption()
_bs = BlackScholes()


def test_cash_or_nothing_call_plus_put_equals_discount():
    """call + put = e^{-rT} (they partition the payoff space)."""
    c = _dig.price(S, K, T, r, sigma, "call", "cash-or-nothing")
    p = _dig.price(S, K, T, r, sigma, "put",  "cash-or-nothing")
    assert abs(c + p - math.exp(-r * T)) < 1e-10


def test_asset_minus_cash_equals_vanilla():
    """asset_or_nothing_call − K·cash_or_nothing_call = BS call.

    cash-or-nothing already includes e^{-rT}, so no extra discounting needed:
        vanilla = S·N(d1) - K·e^{-rT}·N(d2) = aon - K·con
    """
    aon  = _dig.price(S, K, T, r, sigma, "call", "asset-or-nothing")
    con  = _dig.price(S, K, T, r, sigma, "call", "cash-or-nothing")
    vanilla = _bs.price(S, K, T, r, sigma, "call")
    assert abs(aon - K * con - vanilla) < 1e-10


def test_deep_itm_cash_or_nothing_approaches_discount():
    """Deep ITM call → e^{-rT}."""
    c = _dig.price(200.0, K, T, r, sigma, "call", "cash-or-nothing")
    assert abs(c - math.exp(-r * T)) < 0.01


def test_deep_otm_cash_or_nothing_near_zero():
    """Deep OTM call → 0."""
    c = _dig.price(10.0, K, T, r, sigma, "call", "cash-or-nothing")
    assert c < 0.01


def test_notional_scales_price():
    c1 = _dig.price(S, K, T, r, sigma, "call", notional=1.0)
    c5 = _dig.price(S, K, T, r, sigma, "call", notional=5.0)
    assert abs(c5 - 5 * c1) < 1e-10


def test_call_spread_converges_to_digital():
    """Tight call spread converges to cash-or-nothing call price."""
    digital = _dig.price(S, K, T, r, sigma, "call", "cash-or-nothing")
    spread  = call_spread_approximation(S, K, T, r, sigma, dK=0.001)
    assert abs(spread - digital) < 0.02


def test_non_negative_prices():
    for dt in ("cash-or-nothing", "asset-or-nothing"):
        for ot in ("call", "put"):
            assert _dig.price(S, K, T, r, sigma, ot, dt) >= 0.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        _dig.price(S, K, T, r, sigma, "other")


def test_invalid_digital_type_raises():
    with pytest.raises(ValueError):
        _dig.price(S, K, T, r, sigma, "call", "unknown")

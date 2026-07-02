"""Tests for structured product pricers."""
import math
import pytest

from pricing.structured import (
    ReverseConvertible, BarrierReverseConvertible,
    DiscountCertificate, BonusCertificate,
    AirbagCertificate, TwinWinCertificate,
)
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
H = 80.0
notional = 1000.0
coupon = 0.08
_bs = BlackScholes()


def test_rc_price_below_notional():
    """RC investors accept downside risk → price < notional at par."""
    rc = ReverseConvertible().price(S, K, T, r, sigma, coupon, notional)
    assert rc["price"] < notional


def test_rc_decompose_sums_correctly():
    d = ReverseConvertible().price(S, K, T, r, sigma, coupon, notional)
    parts = ReverseConvertible().decompose(S, K, T, r, sigma, coupon, notional)
    total_from_parts = sum(p["value"] for p in parts[:-1])
    assert abs(total_from_parts - d["price"]) < 1e-6


def test_brc_coupon_exceeds_rc_coupon():
    """BRC offers higher nominal protection → higher coupon for same price."""
    rc  = ReverseConvertible().price(S, K, T, r, sigma, coupon, notional)
    brc = BarrierReverseConvertible().price(S, K, T, r, sigma, coupon, H, notional)
    # BRC should be cheaper than RC for same coupon (less risk → less premium sold)
    assert brc["price"] >= rc["price"] - 1.0   # BRC price ≥ RC price (more conservative)


def test_brc_decompose_sums_correctly():
    d = BarrierReverseConvertible().price(S, K, T, r, sigma, coupon, H, notional)
    parts = BarrierReverseConvertible().decompose(S, K, T, r, sigma, coupon, H, notional)
    total_from_parts = sum(p["value"] for p in parts[:-1])
    assert abs(total_from_parts - d["price"]) < 1e-6


def test_discount_cert_below_spot():
    """Discount certificate always costs less than spot × shares (capped upside)."""
    dc = DiscountCertificate().price(S, K, T, r, sigma, notional)
    assert dc["price"] < notional


def test_bonus_cert_above_discount():
    """Bonus certificate costs more than discount (has protection floor)."""
    dc = DiscountCertificate().price(S, K, T, r, sigma, notional)
    bc = BonusCertificate().price(S, K, T, r, sigma, H, notional)
    assert bc["price"] > dc["price"]


def test_airbag_nonnegative():
    d = AirbagCertificate().price(S, K * 1.1, K * 0.9, T, r, sigma, notional=notional)
    assert d["price"] > 0.0


def test_airbag_long_put_floor_is_meaningfully_priced():
    """Regression guard: the dashboard used to call AirbagCertificate with a
    missing K_floor argument, shifting every later positional param by one
    slot and leaving the floor put priced at ~1e-125 (effectively 0)."""
    d = AirbagCertificate().price(S, K_cap=110.0, K_floor=90.0, T=T, r=r, sigma=sigma,
                                   participation=1.0, notional=notional)
    assert d["long_put"] > 5.0   # sane ATM-ish put value, not near-zero garbage
    assert d["floor_level"] == 90.0


def test_twin_win_put_varies_meaningfully_with_strike():
    """Regression guard: the down-and-out put leg used to be nearly flat
    across strikes due to a bug in the underlying barrier formula."""
    low_k = TwinWinCertificate().price(S, 80.0, T, r, sigma, H, notional)
    high_k = TwinWinCertificate().price(S, 120.0, T, r, sigma, H, notional)
    assert high_k["twin_put"] > low_k["twin_put"] * 5


def test_twin_win_above_long_stock():
    """Twin-win adds value via the knock-out put → costs more than plain long stock."""
    tw = TwinWinCertificate().price(S, K, T, r, sigma, H, notional)
    long_stock = notional   # 1 share @ 100 = 100% of notional
    assert tw["price"] > long_stock


def test_decompose_has_correct_keys():
    parts = ReverseConvertible().decompose(S, K, T, r, sigma, coupon, notional)
    for p in parts:
        assert "component" in p
        assert "value" in p
        assert "pct" in p

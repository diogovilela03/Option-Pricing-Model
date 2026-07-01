"""Tests for double barrier option pricing (Ikeda-Kunitomo)."""
import pytest

from pricing.double_barrier import DoubleBarrierOption, double_barrier_mc_price
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
L, U = 85.0, 120.0
_dbl = DoubleBarrierOption()
_bs  = BlackScholes()


def test_dko_plus_dki_equals_vanilla():
    """DKO + DKI = Vanilla (parity)."""
    dko = _dbl.price(S, K, T, r, sigma, "call", L, U, "double-knock-out")
    dki = _dbl.price(S, K, T, r, sigma, "call", L, U, "double-knock-in")
    vanilla = _bs.price(S, K, T, r, sigma, "call")
    assert abs(dko + dki - vanilla) < 0.05   # series truncation causes small error


def test_dko_cheaper_than_vanilla():
    """Double knock-out always <= vanilla (has extra knock-out feature)."""
    dko = _dbl.price(S, K, T, r, sigma, "call", L, U)
    vanilla = _bs.price(S, K, T, r, sigma, "call")
    assert dko <= vanilla + 1e-6


def test_dko_with_wide_barriers_near_vanilla():
    """Barriers very wide → DKO ≈ vanilla (rarely triggered)."""
    dko = _dbl.price(S, K, T, r, sigma, "call", 1.0, 1e6)
    vanilla = _bs.price(S, K, T, r, sigma, "call")
    assert abs(dko - vanilla) < 0.10


def test_spot_outside_corridor_is_zero():
    """S outside [L, U] → DKO is 0."""
    assert _dbl.price(80.0, K, T, r, sigma, "call", L, U) == 0.0
    assert _dbl.price(130.0, K, T, r, sigma, "call", L, U) == 0.0


def test_lower_gte_upper_raises():
    with pytest.raises(ValueError):
        _dbl.price(S, K, T, r, sigma, "call", 120.0, 90.0)


def test_mc_close_to_analytical():
    """MC vs Ikeda-Kunitomo within 10% (wider tolerance for series approx)."""
    analytical = _dbl.price(S, K, T, r, sigma, "call", L, U)
    mc = double_barrier_mc_price(S, K, T, r, sigma, "call", L, U, paths=50_000, seed=42)
    assert abs(mc - analytical) / (analytical + 1e-6) < 0.10


def test_nonnegative():
    assert _dbl.price(S, K, T, r, sigma, "call", L, U) >= 0.0
    assert _dbl.price(S, K, T, r, sigma, "put",  L, U) >= 0.0

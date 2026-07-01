"""Tests for Asian option pricing (geometric closed-form + arithmetic MC)."""
import math
import pytest

from pricing.asian import GeometricAsian, asian_mc_price
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
_geo = GeometricAsian()
_bs  = BlackScholes()


def test_geometric_asian_call_cheaper_than_vanilla():
    """Averaging always reduces the effective volatility → cheaper than vanilla."""
    asian = _geo.price(S, K, T, r, sigma, "call")
    vanilla = _bs.price(S, K, T, r, sigma, "call")
    assert asian < vanilla


def test_geometric_asian_put_cheaper_than_vanilla():
    asian = _geo.price(S, K, T, r, sigma, "put")
    vanilla = _bs.price(S, K, T, r, sigma, "put")
    assert asian < vanilla


def test_geometric_asian_nonnegative():
    for ot in ("call", "put"):
        assert _geo.price(S, K, T, r, sigma, ot) >= 0.0


def test_continuous_vs_large_n_discrete_close():
    """Continuous formula ≈ discrete with n=252 (daily monitoring)."""
    cont = _geo.price(S, K, T, r, sigma, "call", n_periods=None)
    disc = _geo.price(S, K, T, r, sigma, "call", n_periods=252)
    assert abs(cont - disc) / cont < 0.01   # within 1%


def test_geometric_put_call_parity():
    """Modified put-call parity: C_geo − P_geo = S·e^{(b-r)T} − K·e^{-rT}."""
    sigma_G = sigma / math.sqrt(3)
    b = 0.5 * (r - sigma ** 2 / 6)
    c = _geo.price(S, K, T, r, sigma, "call")
    p = _geo.price(S, K, T, r, sigma, "put")
    rhs = S * math.exp((b - r) * T) - K * math.exp(-r * T)
    assert abs((c - p) - rhs) < 1e-8


def test_arithmetic_mc_nonnegative():
    price = asian_mc_price(S, K, T, r, sigma, "call", averaging="arithmetic",
                           paths=10_000, seed=42)
    assert price >= 0.0


def test_arithmetic_mc_near_geometric_at_low_vol():
    """At near-zero vol, arithmetic and geometric averages coincide."""
    sig_low = 0.001
    arith = asian_mc_price(S, K, T, r, sig_low, "call", averaging="arithmetic",
                           paths=20_000, seed=42, use_control_variate=False)
    geo   = _geo.price(S, K, T, r, sig_low, "call")
    assert abs(arith - geo) < 0.10   # loose tolerance: MC noise at low vol


def test_arithmetic_mc_with_cv_close_to_without():
    """Control variate and raw MC should give similar results."""
    cv  = asian_mc_price(S, K, T, r, sigma, "call", paths=20_000, seed=1,
                         use_control_variate=True)
    raw = asian_mc_price(S, K, T, r, sigma, "call", paths=20_000, seed=1,
                         use_control_variate=False)
    assert abs(cv - raw) < 0.5   # both are estimates of the same value


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        _geo.price(S, K, T, r, sigma, "other")


def test_invalid_averaging_raises():
    with pytest.raises(ValueError):
        asian_mc_price(S, K, T, r, sigma, "call", averaging="invalid")

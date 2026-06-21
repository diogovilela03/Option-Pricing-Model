"""
Black-Scholes tests.

Benchmark values from Hull, Options Futures and Other Derivatives (10th ed.)
and independently verified with QuantLib / scipy.stats.
"""
import math
import pytest
from pricing.black_scholes import BlackScholes

BS = BlackScholes()

# ------------------------------------------------------------------
# Benchmark: Hull Example 15.6
#   S=42, K=40, T=0.5, r=0.10, sigma=0.20 → call=4.76, put=0.81
# ------------------------------------------------------------------
S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20
CALL_PRICE = 4.76
PUT_PRICE = 0.81
TOLERANCE = 0.01  # cent-level accuracy


def test_call_price_benchmark():
    assert abs(BS.price(S, K, T, r, sigma, "call") - CALL_PRICE) < TOLERANCE


def test_put_price_benchmark():
    assert abs(BS.price(S, K, T, r, sigma, "put") - PUT_PRICE) < TOLERANCE


def test_put_call_parity():
    """C - P = S - K * exp(-rT) must hold exactly (same BS formula)."""
    C = BS.price(S, K, T, r, sigma, "call")
    P = BS.price(S, K, T, r, sigma, "put")
    rhs = S - K * math.exp(-r * T)
    assert abs((C - P) - rhs) < 1e-10


def test_put_call_parity_deep_itm():
    C = BS.price(200.0, 50.0, 1.0, 0.05, 0.20, "call")
    P = BS.price(200.0, 50.0, 1.0, 0.05, 0.20, "put")
    rhs = 200.0 - 50.0 * math.exp(-0.05 * 1.0)
    assert abs((C - P) - rhs) < 1e-10


# ------------------------------------------------------------------
# Greeks — qualitative sanity checks
# ------------------------------------------------------------------
def test_call_delta_between_zero_and_one():
    greeks = BS.greeks(S, K, T, r, sigma, "call")
    assert 0.0 < greeks["delta"] < 1.0


def test_put_delta_between_minus_one_and_zero():
    greeks = BS.greeks(S, K, T, r, sigma, "put")
    assert -1.0 < greeks["delta"] < 0.0


def test_gamma_positive():
    greeks = BS.greeks(S, K, T, r, sigma, "call")
    assert greeks["gamma"] > 0.0


def test_vega_positive():
    greeks = BS.greeks(S, K, T, r, sigma, "call")
    assert greeks["vega"] > 0.0


def test_call_theta_negative():
    greeks = BS.greeks(S, K, T, r, sigma, "call")
    assert greeks["theta"] < 0.0


def test_call_rho_positive():
    greeks = BS.greeks(S, K, T, r, sigma, "call")
    assert greeks["rho"] > 0.0


def test_put_rho_negative():
    greeks = BS.greeks(S, K, T, r, sigma, "put")
    assert greeks["rho"] < 0.0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------
def test_call_deep_itm_approaches_intrinsic():
    """Very deep ITM call ≈ S - K·e^(-rT)."""
    c = BS.price(500.0, 10.0, 1.0, 0.05, 0.20, "call")
    intrinsic = 500.0 - 10.0 * math.exp(-0.05)
    assert abs(c - intrinsic) < 0.01


def test_call_deep_otm_near_zero():
    c = BS.price(10.0, 500.0, 1.0, 0.05, 0.20, "call")
    assert c < 1e-6


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BS.price(S, K, T, r, sigma, "straddle")  # type: ignore

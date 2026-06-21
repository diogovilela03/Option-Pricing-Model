"""
IV inversion tests.

Core property: round-trip consistency — feed a BS price back through
the inverter and recover the original sigma.
"""
import math
import pytest
from pricing.black_scholes import BlackScholes
from vol_surface.iv_inversion import implied_vol

BS = BlackScholes()

S, K, T, r = 42.0, 40.0, 0.5, 0.10


# ------------------------------------------------------------------
# Round-trip: BS price → implied_vol → original sigma
# ------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
def test_call_round_trip(sigma):
    price = BS.price(S, K, T, r, sigma, "call")
    recovered = implied_vol(S, K, T, r, price, "call")
    assert abs(recovered - sigma) < 1e-6


@pytest.mark.parametrize("sigma", [0.10, 0.20, 0.30, 0.50, 0.80])
def test_put_round_trip(sigma):
    price = BS.price(S, K, T, r, sigma, "put")
    recovered = implied_vol(S, K, T, r, price, "put")
    assert abs(recovered - sigma) < 1e-6


# ------------------------------------------------------------------
# ATM / ITM / OTM strikes
# ------------------------------------------------------------------

@pytest.mark.parametrize("strike", [30.0, 40.0, 42.0, 50.0, 60.0])
def test_various_strikes_call(strike):
    sigma = 0.25
    price = BS.price(S, strike, T, r, sigma, "call")
    recovered = implied_vol(S, strike, T, r, price, "call")
    assert abs(recovered - sigma) < 1e-5


@pytest.mark.parametrize("strike", [30.0, 40.0, 42.0, 50.0, 60.0])
def test_various_strikes_put(strike):
    sigma = 0.25
    price = BS.price(S, strike, T, r, sigma, "put")
    recovered = implied_vol(S, strike, T, r, price, "put")
    assert abs(recovered - sigma) < 1e-5


# ------------------------------------------------------------------
# Edge / error cases
# ------------------------------------------------------------------

def test_price_below_intrinsic_raises():
    """Price below intrinsic value has no valid IV."""
    intrinsic = max(S - K, 0.0)
    with pytest.raises(ValueError, match="intrinsic"):
        implied_vol(S, K, T, r, intrinsic - 1.0, "call")


def test_zero_price_raises():
    with pytest.raises(ValueError):
        implied_vol(S, K, T, r, 0.0, "call")


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        implied_vol(S, K, T, r, 3.0, "straddle")  # type: ignore

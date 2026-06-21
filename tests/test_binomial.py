"""
Binomial tree (CRR, American exercise) tests.

Key properties verified:
- American call on non-dividend-paying stock == BS European call (no early exercise)
- American put > BS European put (early exercise premium exists)
- Convergence to BS as steps increase
- Intrinsic value floor holds
- Invalid option type raises
"""
import math
import pytest
from pricing.binomial import BinomialTree
from pricing.black_scholes import BlackScholes

BT = BinomialTree()
BS = BlackScholes()

S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20


def test_american_call_equals_european_on_no_dividend_stock():
    """American call on non-dividend-paying stock has no early exercise premium."""
    american_call = BT.price(S, K, T, r, sigma, "call", steps=500)
    european_call = BS.price(S, K, T, r, sigma, "call")
    assert abs(american_call - european_call) < 0.01


def test_american_put_exceeds_european_put():
    """American put is worth more than European put due to early exercise."""
    american_put = BT.price(S, K, T, r, sigma, "put", steps=500)
    european_put = BS.price(S, K, T, r, sigma, "put")
    assert american_put > european_put


def test_american_put_convergence_to_bs():
    """Binomial American put converges as steps increase (prices should stabilise)."""
    put_100 = BT.price(S, K, T, r, sigma, "put", steps=100)
    put_500 = BT.price(S, K, T, r, sigma, "put", steps=500)
    assert abs(put_100 - put_500) < 0.05


def test_intrinsic_value_floor_call():
    """American call price >= intrinsic value max(S-K, 0)."""
    price = BT.price(S, K, T, r, sigma, "call", steps=200)
    intrinsic = max(S - K, 0.0)
    assert price >= intrinsic - 1e-9


def test_intrinsic_value_floor_put():
    """American put price >= intrinsic value max(K-S, 0)."""
    price = BT.price(S, K, T, r, sigma, "put", steps=200)
    intrinsic = max(K - S, 0.0)
    assert price >= intrinsic - 1e-9


def test_deep_itm_put_approaches_intrinsic():
    """Deep ITM American put should be close to its intrinsic value."""
    price = BT.price(10.0, 100.0, 1.0, 0.05, 0.20, "put", steps=500)
    intrinsic = 100.0 - 10.0
    assert abs(price - intrinsic) < 1.0


def test_option_price_non_negative():
    assert BT.price(S, K, T, r, sigma, "call", steps=100) >= 0.0
    assert BT.price(S, K, T, r, sigma, "put", steps=100) >= 0.0


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        BT.price(S, K, T, r, sigma, "straddle", steps=100)  # type: ignore

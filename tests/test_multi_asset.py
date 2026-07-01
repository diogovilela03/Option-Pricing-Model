"""Tests for multi-asset option pricers."""
import numpy as np
import pytest

from pricing.multi_asset import BasketOption, WorstOfOption, RainbowOption, correlation_sensitivity

spots = [100.0, 100.0]
vols  = [0.20, 0.25]
K     = 100.0
K_rel = 1.0   # relative strike (1.0 = ATM relative)
T, r  = 0.5, 0.05
corr  = np.array([[1.0, 0.3], [0.3, 1.0]])


def test_basket_nonnegative():
    p = BasketOption().price(spots, [0.5, 0.5], K, T, r, vols, corr, "call",
                             paths=5_000, seed=42)
    assert p >= 0.0


def test_worst_of_nonnegative():
    p = WorstOfOption().price(spots, K_rel, T, r, vols, corr, "call", paths=5_000, seed=42)
    assert p >= 0.0


def test_rainbow_nonnegative():
    p = RainbowOption().price(spots, K_rel, T, r, vols, corr, "call", paths=5_000, seed=42)
    assert p >= 0.0


def test_worst_of_cheaper_than_best_single():
    """Worst-of must be <= any single-asset option (it's the weakest outcome)."""
    from pricing.black_scholes import BlackScholes
    _bs = BlackScholes()
    wo  = WorstOfOption().price(spots, K_rel, T, r, vols, corr, "call", paths=20_000, seed=42)
    # Single asset call at K_rel (i.e. strike = spot * K_rel = spot)
    s1_call = _bs.price(spots[0], spots[0] * K_rel, T, r, vols[0], "call")
    s2_call = _bs.price(spots[1], spots[1] * K_rel, T, r, vols[1], "call")
    assert wo <= max(s1_call, s2_call) + 0.50   # MC tolerance


def test_rainbow_higher_than_worst_of():
    """Best-performing always >= worst-performing → Rainbow >= Worst-of."""
    wo = WorstOfOption().price(spots, K_rel, T, r, vols, corr, "call", paths=20_000, seed=42)
    rb = RainbowOption().price(spots, K_rel, T, r, vols, corr, "call", paths=20_000, seed=42)
    assert rb >= wo - 0.50   # MC tolerance


def test_correlation_sensitivity_returns_array():
    rho_grid, prices = correlation_sensitivity(
        spots, K, T, r, vols, "call", product="basket",
        rho_grid=np.linspace(-0.8, 0.8, 5), paths=3_000, seed=42
    )
    assert len(prices) == 5
    assert all(p >= 0 for p in prices if not np.isnan(p))


def test_invalid_corr_shape_raises():
    bad_corr = np.eye(3)  # 3×3 for 2 assets
    with pytest.raises(ValueError):
        BasketOption().price(spots, [0.5, 0.5], K, T, r, vols, bad_corr, "call", paths=100)

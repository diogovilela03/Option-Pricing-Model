"""Tests for autocallable structured products."""
import pytest

from pricing.autocall import AutocallIncremental, Phoenix, PhoenixMemory

S0, T, r, sigma = 100.0, 2.0, 0.05, 0.20
obs = [0.5, 1.0, 1.5, 2.0]
notional = 1000.0
coupon = 0.08


def test_autocall_price_in_reasonable_range():
    """Price should be between 50% and 150% of notional."""
    d = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                    notional=notional, paths=5_000, seed=42)
    assert 0.5 * notional < d["price"] < 1.5 * notional


def test_phoenix_price_in_reasonable_range():
    d = Phoenix().price(S0, T, r, sigma, observation_dates=obs,
                        notional=notional, paths=5_000, seed=42)
    assert 0.5 * notional < d["price"] < 1.5 * notional


def test_phoenix_memory_price_gte_phoenix():
    """Memory is valuable: Phoenix Memory >= Phoenix (all else equal)."""
    d_mem = PhoenixMemory().price(S0, T, r, sigma, observation_dates=obs,
                                   notional=notional, paths=10_000, seed=42)
    d_phx = Phoenix().price(S0, T, r, sigma, observation_dates=obs,
                             notional=notional, paths=10_000, seed=42)
    # Memory slightly increases price (missed coupons not lost)
    # Allow small tolerance for MC noise
    assert d_mem["price"] >= d_phx["price"] - 10.0


def test_autocall_probability_between_0_and_1():
    d = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                    notional=notional, paths=5_000, seed=42)
    assert 0.0 <= d["autocall_probability"] <= 1.0


def test_high_autocall_level_low_autocall_probability():
    """Autocall level 2× spot → rarely triggered."""
    d = AutocallIncremental().price(S0, T, r, sigma,
                                    autocall_level=2.0, observation_dates=obs,
                                    notional=notional, paths=10_000, seed=42)
    assert d["autocall_probability"] < 0.15


def test_decompose_contains_price():
    d = AutocallIncremental().decompose(S0, T, r, sigma, observation_dates=obs,
                                        notional=notional, paths=5_000, seed=42)
    prices = [c["value"] for c in d]
    assert any(p > 0 for p in prices)


def test_phoenix_result_has_expected_keys():
    d = Phoenix().price(S0, T, r, sigma, observation_dates=obs,
                        notional=notional, paths=5_000, seed=42)
    for key in ("price", "yield", "autocall_probability", "notional"):
        assert key in d

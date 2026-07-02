"""Tests for autocallable structured products."""
import pytest

from pricing.autocall import AutocallIncremental, Phoenix, PhoenixMemory, default_observation_dates

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


def test_result_has_new_analytics_keys():
    d = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                    notional=notional, paths=5_000, seed=42)
    for key in ("capital_loss_probability", "expected_exit_time",
               "equivalent_zcb", "forward_at_maturity", "obs_table"):
        assert key in d
    assert len(d["obs_table"]) == len(obs)
    for row in d["obs_table"]:
        assert 0.0 <= row["maturity_probability"] <= 1.0
        assert 0.0 <= row["coupon_probability"] <= 1.0


def test_obs_table_maturity_probabilities_sum_to_one():
    """Every path exits on exactly one observation date (called or matures)."""
    d = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                    notional=notional, paths=20_000, seed=42)
    total = sum(row["maturity_probability"] for row in d["obs_table"])
    assert abs(total - 1.0) < 1e-9


def test_american_dip_more_conservative_than_european():
    """American-style protection (checked at every observation) can only be
    breached earlier/more often than European (checked at maturity only),
    so it should never be worth more and should show >= capital-loss risk."""
    d_eur = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                        protection_barrier=0.60, dip_style="european",
                                        notional=notional, paths=30_000, seed=11)
    d_amer = AutocallIncremental().price(S0, T, r, sigma, observation_dates=obs,
                                         protection_barrier=0.60, dip_style="american",
                                         notional=notional, paths=30_000, seed=11)
    assert d_amer["capital_loss_probability"] >= d_eur["capital_loss_probability"]
    assert d_amer["price"] <= d_eur["price"] + 1.0  # small MC tolerance


def test_default_observation_dates_frequency():
    assert default_observation_dates(2.0, "annual") == [1.0, 2.0]
    assert default_observation_dates(2.0, "semi-annual") == [0.5, 1.0, 1.5, 2.0]
    assert len(default_observation_dates(2.0, "quarterly")) == 8
    assert len(default_observation_dates(2.0, "monthly")) == 24

"""Tests for the shared finite-difference Greeks helper (pricing/greeks_fd.py).

Validated against pricing/black_scholes.py's analytic Greeks, since BS is the
one pricer in this project with closed-form Greeks to check against. All
Greeks-vs-Spot charts in the dashboard route exotic/structured payoffs
through fd_greeks, so this file is the only thing standing between a bad
bump-size change and every one of those charts silently going wrong.
"""
import pytest
from pricing.black_scholes import BlackScholes
from pricing.greeks_fd import fd_greeks, fd_greeks_profile

BS = BlackScholes()

# (S, K, T, r, sigma, option_type)
CASES = [
    (42.0, 40.0, 0.5, 0.10, 0.20, "call"),
    (42.0, 40.0, 0.5, 0.10, 0.20, "put"),
    (100.0, 100.0, 1.0, 0.05, 0.25, "call"),
    (100.0, 100.0, 1.0, 0.05, 0.25, "put"),
]

# Deep OTM, short-dated: Greeks are tiny in absolute terms, so only an
# absolute (not relative) tolerance is meaningful here.
OTM_CASE = (100.0, 120.0, 0.1, 0.02, 0.15, "call")

ABS_TOL = {"delta": 1e-4, "gamma": 1e-4, "vega": 1e-4, "theta": 5e-5, "rho": 1e-4}


def _bs_price_fn(option_type):
    return lambda s, k, t, r, sigma: BS.price(s, k, t, r, sigma, option_type)


@pytest.mark.parametrize("S,K,T,r,sigma,option_type", CASES)
def test_fd_greeks_match_analytic_bs(S, K, T, r, sigma, option_type):
    analytic = BS.greeks(S, K, T, r, sigma, option_type)
    fd = fd_greeks(_bs_price_fn(option_type), S, K, T, r, sigma)

    for greek, tol in ABS_TOL.items():
        assert abs(fd[greek] - analytic[greek]) < tol, (
            f"{greek} mismatch: fd={fd[greek]!r} analytic={analytic[greek]!r}"
        )


def test_fd_greeks_otm_short_dated_absolute_tolerance():
    """Deep OTM / short-dated Greeks are tiny; only check absolute error,
    since relative error blows up for near-zero analytic values."""
    S, K, T, r, sigma, option_type = OTM_CASE
    analytic = BS.greeks(S, K, T, r, sigma, option_type)
    fd = fd_greeks(_bs_price_fn(option_type), S, K, T, r, sigma)

    for greek, tol in ABS_TOL.items():
        assert abs(fd[greek] - analytic[greek]) < tol


def test_fd_gamma_positive_for_vanilla():
    """Sanity check independent of the analytic comparison: gamma must be
    positive for a vanilla option (convex payoff)."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    fd = fd_greeks(_bs_price_fn("call"), S, K, T, r, sigma)
    assert fd["gamma"] > 0.0


def test_fd_greeks_profile_shape_and_consistency():
    """fd_greeks_profile must return one value per grid point per Greek,
    and each point must match a standalone fd_greeks call at that spot."""
    K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    s_grid = [80.0, 100.0, 120.0]
    profile = fd_greeks_profile(_bs_price_fn("call"), s_grid, K, T, r, sigma)

    assert set(profile.keys()) == {"delta", "gamma", "vega", "theta", "rho"}
    for greek_values in profile.values():
        assert len(greek_values) == len(s_grid)

    for i, s in enumerate(s_grid):
        pointwise = fd_greeks(_bs_price_fn("call"), s, K, T, r, sigma)
        for greek in profile:
            assert profile[greek][i] == pytest.approx(pointwise[greek])


def test_fd_greeks_handles_near_zero_sigma_bump_without_error():
    """The vega bump-down clamps sigma to a floor (max(sigma-eps, 1e-6)) so
    it never evaluates the pricer at sigma <= 0; must not raise even when
    sigma itself is already tiny."""
    fd = fd_greeks(_bs_price_fn("call"), 100.0, 100.0, 1.0, 0.05, 1e-5)
    assert all(v == v for v in fd.values())  # no NaNs

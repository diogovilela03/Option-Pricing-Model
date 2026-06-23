"""
SVI calibration tests.

Core property: fit synthetic data generated from known SVI params,
then verify the fitted params reproduce the input total variances closely.
"""
import numpy as np
import pytest
from vol_surface.svi import svi_total_var, fit_slice

# Known arbitrage-free SVI params (Gatheral 2004 style)
TRUE_PARAMS = {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.2}

K_GRID = np.linspace(-0.5, 0.5, 21)  # log-moneyness grid


def _synthetic_slice(params=TRUE_PARAMS, noise_std=0.0, seed=0):
    w = svi_total_var(K_GRID, params)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        w = w + rng.normal(0, noise_std, size=w.shape)
        w = np.maximum(w, 1e-8)
    return w


# ------------------------------------------------------------------
# svi_total_var
# ------------------------------------------------------------------

def test_total_var_non_negative():
    w = svi_total_var(K_GRID, TRUE_PARAMS)
    assert np.all(w >= 0)


def test_total_var_shape():
    w = svi_total_var(K_GRID, TRUE_PARAMS)
    assert w.shape == K_GRID.shape


def test_total_var_atm_equals_formula():
    """At k=0, w = a + b*(rho*(-m) + sqrt(m^2 + sigma^2))."""
    p = TRUE_PARAMS
    k0 = np.array([0.0])
    expected = p["a"] + p["b"] * (p["rho"] * (0 - p["m"]) + np.sqrt(p["m"]**2 + p["sigma"]**2))
    result = svi_total_var(k0, p)
    assert abs(result[0] - expected) < 1e-12


# ------------------------------------------------------------------
# fit_slice — noiseless round-trip
# ------------------------------------------------------------------

def test_fit_recovers_total_variance():
    """Fitted params reproduce the input total variances (noiseless)."""
    w_market = _synthetic_slice()
    fitted = fit_slice(K_GRID, w_market)
    w_fitted = svi_total_var(K_GRID, fitted)
    np.testing.assert_allclose(w_fitted, w_market, atol=1e-4)


def test_fit_returns_required_keys():
    w_market = _synthetic_slice()
    fitted = fit_slice(K_GRID, w_market)
    assert set(fitted.keys()) == {"a", "b", "rho", "m", "sigma"}


def test_fit_params_within_constraints():
    """b >= 0, |rho| < 1, sigma > 0."""
    w_market = _synthetic_slice()
    fitted = fit_slice(K_GRID, w_market)
    assert fitted["b"] >= 0
    assert abs(fitted["rho"]) < 1
    assert fitted["sigma"] > 0


def test_fit_noisy_data_reasonable():
    """With small noise, fit still produces non-negative total variance."""
    w_market = _synthetic_slice(noise_std=0.001)
    fitted = fit_slice(K_GRID, w_market)
    w_fitted = svi_total_var(K_GRID, fitted)
    assert np.all(w_fitted >= 0)

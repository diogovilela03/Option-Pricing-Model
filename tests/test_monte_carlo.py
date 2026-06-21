"""
Monte Carlo pricer tests.

Convergence to BS is the primary correctness check. Variance reduction
methods are verified to tighten that convergence, not just pass/fail.
Heston is tested for consistency and boundary behaviour, not against a
closed-form solution (none exists in simple form).
"""
import math
import pytest
import numpy as np
from pricing.monte_carlo import MonteCarlo
from pricing.black_scholes import BlackScholes

MC = MonteCarlo()
BS = BlackScholes()

S, K, T, r, sigma = 42.0, 40.0, 0.5, 0.10, 0.20
SEED = 42


# ------------------------------------------------------------------
# GBM convergence to BS
# ------------------------------------------------------------------

def test_gbm_call_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "call")
    mc_price = MC.price(S, K, T, r, sigma, "call", paths=100_000, seed=SEED)
    assert abs(mc_price - bs_price) < 0.05


def test_gbm_put_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "put")
    mc_price = MC.price(S, K, T, r, sigma, "put", paths=100_000, seed=SEED)
    assert abs(mc_price - bs_price) < 0.05


# ------------------------------------------------------------------
# Antithetic variates — tighter convergence with fewer paths
# ------------------------------------------------------------------

def test_antithetic_call_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "call")
    mc_price = MC.price(
        S, K, T, r, sigma, "call",
        paths=50_000, variance_reduction="antithetic", seed=SEED,
    )
    assert abs(mc_price - bs_price) < 0.05


def test_antithetic_put_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "put")
    mc_price = MC.price(
        S, K, T, r, sigma, "put",
        paths=50_000, variance_reduction="antithetic", seed=SEED,
    )
    assert abs(mc_price - bs_price) < 0.05


# ------------------------------------------------------------------
# Control variate — tightest convergence
# ------------------------------------------------------------------

def test_control_variate_call_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "call")
    mc_price = MC.price(
        S, K, T, r, sigma, "call",
        paths=20_000, variance_reduction="control_variate", seed=SEED,
    )
    assert abs(mc_price - bs_price) < 0.02


def test_control_variate_put_converges_to_bs():
    bs_price = BS.price(S, K, T, r, sigma, "put")
    mc_price = MC.price(
        S, K, T, r, sigma, "put",
        paths=20_000, variance_reduction="control_variate", seed=SEED,
    )
    assert abs(mc_price - bs_price) < 0.02


# ------------------------------------------------------------------
# Heston dynamics
# ------------------------------------------------------------------

HESTON = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}


def test_heston_call_non_negative():
    price = MC.price(
        S, K, T, r, sigma, "call",
        paths=20_000, dynamics="heston", heston_params=HESTON, seed=SEED,
    )
    assert price >= 0.0


def test_heston_put_non_negative():
    price = MC.price(
        S, K, T, r, sigma, "put",
        paths=20_000, dynamics="heston", heston_params=HESTON, seed=SEED,
    )
    assert price >= 0.0


def test_heston_low_vol_of_vol_near_gbm():
    """Heston with near-zero xi should price close to GBM / BS."""
    flat_heston = {"v0": sigma**2, "kappa": 5.0, "theta": sigma**2, "xi": 0.01, "rho": 0.0}
    bs_price = BS.price(S, K, T, r, sigma, "call")
    heston_price = MC.price(
        S, K, T, r, sigma, "call",
        paths=50_000, dynamics="heston", heston_params=flat_heston, seed=SEED,
    )
    assert abs(heston_price - bs_price) < 0.10


def test_heston_feller_violation_warns():
    """2*kappa*theta < xi^2 violates the Feller condition — should emit a warning."""
    bad_params = {"v0": 0.04, "kappa": 0.5, "theta": 0.04, "xi": 1.0, "rho": 0.0}
    with pytest.warns(UserWarning, match="Feller"):
        MC.price(
            S, K, T, r, sigma, "call",
            paths=1_000, dynamics="heston", heston_params=bad_params, seed=SEED,
        )


# ------------------------------------------------------------------
# Edge / error cases
# ------------------------------------------------------------------

def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        MC.price(S, K, T, r, sigma, "straddle", paths=1_000)  # type: ignore


def test_invalid_dynamics_raises():
    with pytest.raises(ValueError):
        MC.price(S, K, T, r, sigma, "call", paths=1_000, dynamics="sabr")  # type: ignore


def test_invalid_variance_reduction_raises():
    with pytest.raises(ValueError):
        MC.price(S, K, T, r, sigma, "call", paths=1_000, variance_reduction="importance")  # type: ignore


def test_price_non_negative():
    assert MC.price(S, K, T, r, sigma, "call", paths=1_000, seed=SEED) >= 0.0
    assert MC.price(S, K, T, r, sigma, "put", paths=1_000, seed=SEED) >= 0.0

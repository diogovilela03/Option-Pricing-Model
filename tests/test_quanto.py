"""Tests for quanto option pricing."""
import math
import pytest

from pricing.quanto import QuantoOption
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
_qto = QuantoOption()
_bs  = BlackScholes()


def test_quanto_reduces_to_bs_at_zero_correlation():
    """rho=0 and sigma_FX=0 → no FX drift adjustment → identical to BS."""
    q = _qto.price(S, K, T, r, sigma, "call", r_d=r, r_f=0.0,
                   sigma_S=sigma, sigma_FX=0.0, rho=0.0, Q0=1.0)
    bs = _bs.price(S, K, T, r, sigma, "call")
    assert abs(q - bs) < 1e-6


def test_negative_correlation_increases_call_price():
    """Negative rho: b = r_d - r_f - rho·σ_S·σ_FX > b at rho=0 → higher call price."""
    q_neg = _qto.price(S, K, T, r, sigma, "call", r_d=r, r_f=0.0,
                       sigma_S=sigma, sigma_FX=0.15, rho=-0.5, Q0=1.0)
    q_zero = _qto.price(S, K, T, r, sigma, "call", r_d=r, r_f=0.0,
                        sigma_S=sigma, sigma_FX=0.15, rho=0.0, Q0=1.0)
    assert q_neg > q_zero


def test_q0_scales_price():
    """Doubling Q0 doubles the price."""
    q1 = _qto.price(S, K, T, r, sigma, "call", Q0=1.0)
    q2 = _qto.price(S, K, T, r, sigma, "call", Q0=2.0)
    assert abs(q2 - 2 * q1) < 1e-6


def test_nonnegative():
    for ot in ("call", "put"):
        assert _qto.price(S, K, T, r, sigma, ot) >= 0.0


def test_decompose_components_sum_to_price():
    components = _qto.decompose(S, K, T, r, sigma, "call",
                                sigma_FX=0.10, rho=-0.3, Q0=1.0)
    total = next(c["value"] for c in components if "Quanto" in c["component"])
    # First component is vanilla + adjustment = total
    assert total == pytest.approx(_qto.price(S, K, T, r, sigma, "call",
                                             sigma_FX=0.10, rho=-0.3, Q0=1.0), rel=1e-4)


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        _qto.price(S, K, T, r, sigma, "other")

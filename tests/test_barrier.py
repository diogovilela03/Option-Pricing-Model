"""Tests for barrier option pricing (Reiner-Rubinstein + Monte Carlo)."""
import pytest

from pricing.barrier import BarrierOption, barrier_mc_price
from pricing.black_scholes import BlackScholes

S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.20
H_dn, H_up = 90.0, 110.0
_bar = BarrierOption()
_bs  = BlackScholes()


def _vanilla(ot): return _bs.price(S, K, T, r, sigma, ot)
def _doi(ot): return _bar.price(S, K, T, r, sigma, ot, H_dn, "down-and-in")
def _doo(ot): return _bar.price(S, K, T, r, sigma, ot, H_dn, "down-and-out")
def _uoi(ot): return _bar.price(S, K, T, r, sigma, ot, H_up, "up-and-in")
def _uoo(ot): return _bar.price(S, K, T, r, sigma, ot, H_up, "up-and-out")


def test_down_in_plus_out_equals_vanilla_call():
    """In + Out = Vanilla (exact parity, core correctness check)."""
    assert abs(_doi("call") + _doo("call") - _vanilla("call")) < 1e-8


def test_down_in_plus_out_equals_vanilla_put():
    assert abs(_doi("put") + _doo("put") - _vanilla("put")) < 1e-8


def test_up_in_plus_out_equals_vanilla_call():
    assert abs(_uoi("call") + _uoo("call") - _vanilla("call")) < 1e-8


def test_up_in_plus_out_equals_vanilla_put():
    assert abs(_uoi("put") + _uoo("put") - _vanilla("put")) < 1e-8


def test_down_and_out_remote_barrier_near_vanilla():
    """Barrier far below spot → DOC ≈ vanilla call."""
    doc = _bar.price(S, K, T, r, sigma, "call", 1.0, "down-and-out")
    assert abs(doc - _vanilla("call")) < 0.01


def test_up_and_in_remote_barrier_near_zero():
    """Barrier far above spot → UIC ≈ 0 (never triggers)."""
    uic = _bar.price(S, K, T, r, sigma, "call", S * 10, "up-and-in")
    assert uic < 0.01


def test_non_negative_prices():
    for bt in ("down-and-out", "down-and-in"):
        for ot in ("call", "put"):
            p = _bar.price(S, K, T, r, sigma, ot, H_dn, bt)
            assert p >= 0.0
    for bt in ("up-and-out", "up-and-in"):
        for ot in ("call", "put"):
            p = _bar.price(S, K, T, r, sigma, ot, H_up, bt)
            assert p >= 0.0


def test_mc_close_to_rr_down_and_out_call():
    """MC vs Reiner-Rubinstein within 5% at 50k paths."""
    rr = _doo("call")
    mc = barrier_mc_price(S, K, T, r, sigma, "call", H_dn, "down-and-out",
                          paths=50_000, seed=42)
    assert abs(mc - rr) / (rr + 1e-6) < 0.05


@pytest.mark.parametrize("barrier_type,option_type,H,K_test", [
    ("down-and-out", "call", 80.0, 100.0),
    ("down-and-out", "call", 80.0, 70.0),   # K <= H branch
    ("down-and-out", "put",  80.0, 100.0),
    ("up-and-out",   "call", 120.0, 100.0),
    ("up-and-out",   "put",  120.0, 100.0),
    ("up-and-out",   "put",  120.0, 130.0),  # K > H branch
])
def test_mc_close_to_rr_all_out_types(barrier_type, option_type, H, K_test):
    """Regression guard for the Reiner-Rubinstein branch-table fix: every
    'out' combo (both K>H and K<=H branches) must track MC within 10%."""
    rr = _bar.price(S, K_test, T, r, sigma, option_type, H, barrier_type)
    mc = barrier_mc_price(S, K_test, T, r, sigma, option_type, H, barrier_type,
                          paths=200_000, steps=200, seed=42)
    assert abs(rr - mc) / max(mc, 0.1) < 0.10


def test_up_and_out_call_drops_to_zero_past_barrier():
    """Regression guard: an up-and-out option must be worth ~0 once spot has
    already breached the barrier at inception (previously this extrapolated
    the RR formula past its valid domain and returned nonsense)."""
    p_at_barrier = _bar.price(120.0, 100.0, 1.0, 0.02, 0.20, "call", 120.0, "up-and-out")
    p_past_barrier = _bar.price(130.0, 100.0, 1.0, 0.02, 0.20, "call", 120.0, "up-and-out")
    assert p_at_barrier == 0.0
    assert p_past_barrier == 0.0


def test_down_and_out_put_already_breached_is_zero():
    p = _bar.price(75.0, 100.0, 1.0, 0.02, 0.20, "put", 80.0, "down-and-out")
    assert p == 0.0


def test_up_and_in_already_breached_equals_vanilla():
    """An 'in' option that has already breached at inception behaves as
    the plain vanilla."""
    p = _bar.price(125.0, 100.0, 1.0, 0.02, 0.20, "call", 120.0, "up-and-in")
    v = _bs.price(125.0, 100.0, 1.0, 0.02, 0.20, "call")
    assert abs(p - v) < 1e-9


def test_invalid_option_type_raises():
    with pytest.raises(ValueError):
        _bar.price(S, K, T, r, sigma, "other", H_dn, "down-and-out")


def test_invalid_barrier_type_raises():
    with pytest.raises(ValueError):
        _bar.price(S, K, T, r, sigma, "call", H_dn, "sideways-and-confused")

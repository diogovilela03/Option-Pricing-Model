"""
Arbitrage check tests.

Butterfly: Gatheral-Jacquier g(k) >= 0 within a slice.
Calendar:  Total variance non-decreasing across expiries for each k.
"""
import numpy as np
import pytest
from vol_surface.svi import svi_total_var
from vol_surface.arbitrage_checks import check_butterfly, check_calendar

K_GRID = np.linspace(-0.5, 0.5, 51)

# Well-behaved params — should be butterfly-arbitrage-free
CLEAN_PARAMS = {"a": 0.04, "b": 0.1, "rho": -0.3, "m": 0.0, "sigma": 0.2}

# Params chosen to violate butterfly (extremely steep wings, high b)
BUTTERFLY_VIOLATION = {"a": 0.001, "b": 2.0, "rho": 0.0, "m": 0.0, "sigma": 0.01}


# ------------------------------------------------------------------
# Butterfly checks
# ------------------------------------------------------------------

def test_butterfly_clean_passes():
    result = check_butterfly(K_GRID, CLEAN_PARAMS)
    assert result["arbitrage_free"] is True


def test_butterfly_violation_detected():
    result = check_butterfly(K_GRID, BUTTERFLY_VIOLATION)
    assert result["arbitrage_free"] is False


def test_butterfly_result_has_required_keys():
    result = check_butterfly(K_GRID, CLEAN_PARAMS)
    assert "arbitrage_free" in result
    assert "min_g" in result
    assert "violation_count" in result


def test_butterfly_min_g_non_negative_when_clean():
    result = check_butterfly(K_GRID, CLEAN_PARAMS)
    assert result["min_g"] >= 0


# ------------------------------------------------------------------
# Calendar spread checks
# ------------------------------------------------------------------

def _make_slices(scale_factors, base=CLEAN_PARAMS):
    """Build a list of (T, params) with increasing total variance."""
    slices = []
    for i, scale in enumerate(scale_factors):
        T = (i + 1) * 0.25
        params = {**base, "a": base["a"] * scale}
        slices.append({"T": T, "params": params})
    return slices


def test_calendar_increasing_variance_passes():
    slices = _make_slices([1.0, 1.5, 2.0])
    result = check_calendar(K_GRID, slices)
    assert result["arbitrage_free"] is True


def test_calendar_decreasing_variance_fails():
    slices = _make_slices([2.0, 1.5, 1.0])  # total var decreases with T
    result = check_calendar(K_GRID, slices)
    assert result["arbitrage_free"] is False


def test_calendar_result_has_required_keys():
    slices = _make_slices([1.0, 2.0])
    result = check_calendar(K_GRID, slices)
    assert "arbitrage_free" in result
    assert "violation_count" in result
    assert "pair_results" in result


def test_calendar_single_slice_passes():
    """One slice — no pairs to check, trivially arbitrage-free."""
    slices = _make_slices([1.0])
    result = check_calendar(K_GRID, slices)
    assert result["arbitrage_free"] is True

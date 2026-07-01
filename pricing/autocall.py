"""Autocallable structured product pricers (Monte Carlo).

All products share a common observation-schedule path simulator.

Products:
    AutocallIncremental — knocked out early if S >= autocall level; pays n_elapsed coupons
    Phoenix             — coupon paid conditionally on each date; capital loss if barrier hit
    PhoenixMemory       — same as Phoenix but missed coupons accumulate and are paid later
"""
import math
from typing import Sequence

import numpy as np


def _simulate_obs_paths(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    observation_dates: Sequence[float],
    paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate GBM paths and return prices at each observation date.

    Returns shape (n_obs, paths).
    observation_dates: sorted list of times in years, e.g. [0.25, 0.5, 0.75, 1.0].
    """
    rng = np.random.default_rng(seed)
    obs = sorted(observation_dates)
    n_obs = len(obs)
    result = np.empty((n_obs, paths))

    t_prev = 0.0
    log_S = np.zeros(paths)

    for i, t in enumerate(obs):
        dt = t - t_prev
        if dt > 0:
            drift = (r - 0.5 * sigma ** 2) * dt
            vol = sigma * math.sqrt(dt)
            Z = rng.standard_normal(paths)
            log_S += drift + vol * Z
        result[i] = S0 * np.exp(log_S)
        t_prev = t

    return result


class AutocallIncremental:
    """Autocall Incremental (vanilla autocall).

    On each observation date:
        If S_i >= autocall_level * S0: product is called → pay notional + i×coupon × notional
    At final maturity (if never called):
        If S_T >= protection_barrier * S0: pay notional
        Else: pay (S_T / S0) * notional  (capital loss)
    """

    def price(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        autocall_level: float = 1.0,
        coupon_rate: float = 0.08,
        observation_dates: Sequence[float] | None = None,
        protection_barrier: float = 0.60,
        notional: float = 1000.0,
        paths: int = 50_000,
        seed: int | None = None,
    ) -> dict:
        if observation_dates is None:
            n_obs = max(1, int(T * 4))
            observation_dates = [round(T * i / n_obs, 6) for i in range(1, n_obs + 1)]

        obs_prices = _simulate_obs_paths(S0, T, r, sigma, observation_dates, paths, seed)
        obs_dates = sorted(observation_dates)
        n_obs = len(obs_dates)

        pv_cashflows = np.zeros(paths)
        alive = np.ones(paths, dtype=bool)

        for i in range(n_obs):
            t_i = obs_dates[i]
            S_i = obs_prices[i]
            called = alive & (S_i >= autocall_level * S0)

            if called.any():
                payout = notional * (1 + (i + 1) * coupon_rate)
                pv_cashflows[called] += payout * math.exp(-r * t_i)
                alive[called] = False

        # Maturity payoff for survivors
        if alive.any():
            S_T = obs_prices[-1]
            maturity_payout = np.where(
                S_T[alive] >= protection_barrier * S0,
                notional,
                notional * S_T[alive] / S0,
            )
            pv_cashflows[alive] += maturity_payout * math.exp(-r * T)

        price = float(np.mean(pv_cashflows))
        autocall_prob = float(np.mean(~alive))

        return {
            "price": price,
            "yield": coupon_rate,
            "autocall_probability": autocall_prob,
            "notional": notional,
            "obs_prices": obs_prices,
            "obs_dates": obs_dates,
        }

    def decompose(self, S0, T, r, sigma, autocall_level=1.0, coupon_rate=0.08,
                  observation_dates=None, protection_barrier=0.60, notional=1000.0,
                  paths=50_000, seed=None) -> list[dict]:
        d = self.price(S0, T, r, sigma, autocall_level, coupon_rate,
                       observation_dates, protection_barrier, notional, paths, seed)
        bond_floor = notional * math.exp(-r * T)
        return [
            {"component": "Bond floor (capital protection value)",
             "value": bond_floor, "pct": bond_floor / notional * 100},
            {"component": f"Autocall feature ({coupon_rate:.1%}/period coupon)",
             "value": d["price"] - bond_floor, "pct": (d["price"] - bond_floor) / notional * 100},
            {"component": f"Autocall Price (P(called)={d['autocall_probability']:.1%})",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class Phoenix:
    """Phoenix Autocall.

    On each observation date:
        If S_i >= autocall_level * S0: called → pay notional + all coupons
        Elif S_i >= coupon_barrier * S0: pay coupon, continue
        Else: no coupon, continue
    At maturity:
        If S_T >= protection_barrier * S0: pay notional
        Else: pay (S_T / S0) * notional
    """

    def price(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        autocall_level: float = 1.0,
        coupon_barrier: float = 0.70,
        protection_barrier: float = 0.60,
        coupon_rate: float = 0.08,
        observation_dates: Sequence[float] | None = None,
        notional: float = 1000.0,
        paths: int = 50_000,
        seed: int | None = None,
    ) -> dict:
        if observation_dates is None:
            n_obs = max(1, int(T * 4))
            observation_dates = [round(T * i / n_obs, 6) for i in range(1, n_obs + 1)]

        obs_prices = _simulate_obs_paths(S0, T, r, sigma, observation_dates, paths, seed)
        obs_dates = sorted(observation_dates)

        pv_cashflows = np.zeros(paths)
        alive = np.ones(paths, dtype=bool)

        for i, t_i in enumerate(obs_dates):
            S_i = obs_prices[i]

            called = alive & (S_i >= autocall_level * S0)
            if called.any():
                pv_cashflows[called] += notional * (1 + coupon_rate) * math.exp(-r * t_i)
                alive[called] = False

            coupon_paid = alive & (S_i >= coupon_barrier * S0)
            pv_cashflows[coupon_paid] += notional * coupon_rate * math.exp(-r * t_i)

        if alive.any():
            S_T = obs_prices[-1]
            maturity_payout = np.where(
                S_T[alive] >= protection_barrier * S0,
                notional,
                notional * S_T[alive] / S0,
            )
            pv_cashflows[alive] += maturity_payout * math.exp(-r * T)

        price = float(np.mean(pv_cashflows))
        return {
            "price": price,
            "yield": coupon_rate,
            "autocall_probability": float(np.mean(~alive)),
            "notional": notional,
            "obs_prices": obs_prices,
            "obs_dates": obs_dates,
        }

    def decompose(self, S0, T, r, sigma, autocall_level=1.0, coupon_barrier=0.70,
                  protection_barrier=0.60, coupon_rate=0.08, observation_dates=None,
                  notional=1000.0, paths=50_000, seed=None) -> list[dict]:
        d = self.price(S0, T, r, sigma, autocall_level, coupon_barrier, protection_barrier,
                       coupon_rate, observation_dates, notional, paths, seed)
        bond_floor = notional * math.exp(-r * T)
        return [
            {"component": "Bond floor",
             "value": bond_floor, "pct": bond_floor / notional * 100},
            {"component": f"Conditional coupons (barrier={coupon_barrier:.0%} S0)",
             "value": d["price"] - bond_floor, "pct": (d["price"] - bond_floor) / notional * 100},
            {"component": f"Phoenix Price (P(called)={d['autocall_probability']:.1%})",
             "value": d["price"], "pct": d["price"] / notional * 100},
        ]


class PhoenixMemory:
    """Phoenix Memory Autocall.

    Same as Phoenix but missed coupons accumulate.
    When S crosses coupon_barrier on a later date, all accumulated coupons are paid.
    """

    def price(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        autocall_level: float = 1.0,
        coupon_barrier: float = 0.70,
        protection_barrier: float = 0.60,
        coupon_rate: float = 0.08,
        observation_dates: Sequence[float] | None = None,
        notional: float = 1000.0,
        paths: int = 50_000,
        seed: int | None = None,
    ) -> dict:
        if observation_dates is None:
            n_obs = max(1, int(T * 4))
            observation_dates = [round(T * i / n_obs, 6) for i in range(1, n_obs + 1)]

        obs_prices = _simulate_obs_paths(S0, T, r, sigma, observation_dates, paths, seed)
        obs_dates = sorted(observation_dates)

        pv_cashflows = np.zeros(paths)
        alive = np.ones(paths, dtype=bool)
        accumulated = np.zeros(paths)  # missed coupons accumulating per path

        for i, t_i in enumerate(obs_dates):
            S_i = obs_prices[i]

            called = alive & (S_i >= autocall_level * S0)
            if called.any():
                # Pay all accumulated + current coupon
                total_coupon = (accumulated[called] + coupon_rate) * notional
                pv_cashflows[called] += (notional + total_coupon) * math.exp(-r * t_i)
                alive[called] = False

            coupon_ok = alive & (S_i >= coupon_barrier * S0)
            if coupon_ok.any():
                total_coupon = (accumulated[coupon_ok] + coupon_rate) * notional
                pv_cashflows[coupon_ok] += total_coupon * math.exp(-r * t_i)
                accumulated[coupon_ok] = 0.0

            missed = alive & ~coupon_ok
            accumulated[missed] += coupon_rate   # memory: store for later

        if alive.any():
            S_T = obs_prices[-1]
            maturity_payout = np.where(
                S_T[alive] >= protection_barrier * S0,
                notional,
                notional * S_T[alive] / S0,
            )
            pv_cashflows[alive] += maturity_payout * math.exp(-r * T)

        price = float(np.mean(pv_cashflows))
        return {
            "price": price,
            "yield": coupon_rate,
            "autocall_probability": float(np.mean(~alive)),
            "notional": notional,
            "obs_prices": obs_prices,
            "obs_dates": obs_dates,
        }

    def decompose(self, S0, T, r, sigma, autocall_level=1.0, coupon_barrier=0.70,
                  protection_barrier=0.60, coupon_rate=0.08, observation_dates=None,
                  notional=1000.0, paths=50_000, seed=None) -> list[dict]:
        d_mem = self.price(S0, T, r, sigma, autocall_level, coupon_barrier,
                           protection_barrier, coupon_rate, observation_dates,
                           notional, paths, seed)
        d_phx = Phoenix().price(S0, T, r, sigma, autocall_level, coupon_barrier,
                                protection_barrier, coupon_rate, observation_dates,
                                notional, paths, seed + 1 if seed else None)
        memory_premium = d_mem["price"] - d_phx["price"]
        return [
            {"component": "Phoenix (no memory) base price",
             "value": d_phx["price"], "pct": d_phx["price"] / notional * 100},
            {"component": "Memory premium (value of accumulated coupons)",
             "value": memory_premium, "pct": memory_premium / notional * 100},
            {"component": f"Phoenix Memory Price (P(called)={d_mem['autocall_probability']:.1%})",
             "value": d_mem["price"], "pct": d_mem["price"] / notional * 100},
        ]

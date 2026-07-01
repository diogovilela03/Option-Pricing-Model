"""Tab 3 — Tier C: Autocallable Structured Products."""
import math

import numpy as np
import pandas as pd
import streamlit as st

from pricing.autocall import AutocallIncremental, Phoenix, PhoenixMemory
from dashboard.charts import autocall_schedule_figure, decomposition_waterfall_figure

_AUTO = AutocallIncremental()
_PHX  = Phoenix()
_PMEM = PhoenixMemory()

AUTOCALL_PRODUCTS = ["Autocall Incremental", "Phoenix", "Phoenix Memory"]


def render_autocall_sidebar() -> dict:
    product = st.sidebar.selectbox("Product", AUTOCALL_PRODUCTS)
    st.sidebar.markdown("---")
    S0 = st.sidebar.number_input("Initial Spot S₀", value=100.0, min_value=1.0, step=1.0)
    T  = st.sidebar.number_input("Maturity T (years)", value=2.0, min_value=0.25, step=0.25)
    r  = st.sidebar.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.3f")
    sigma = st.sidebar.number_input("Volatility σ", value=0.20, step=0.01, format="%.2f")
    autocall_lvl = st.sidebar.number_input(
        "Autocall Level (× S₀)", value=1.00, min_value=0.5, step=0.05, format="%.2f")
    coupon = st.sidebar.number_input("Coupon Rate (per obs)", value=0.08, step=0.01, format="%.2f")
    prot_barrier = st.sidebar.number_input(
        "Protection Barrier (× S₀)", value=0.60, min_value=0.1, step=0.05, format="%.2f")

    extra = {}
    if product in ("Phoenix", "Phoenix Memory"):
        extra["coupon_barrier"] = st.sidebar.number_input(
            "Coupon Barrier (× S₀)", value=0.70, step=0.05, format="%.2f")

    with st.sidebar.expander("MC Settings"):
        extra["paths"] = st.number_input("Paths", value=20_000, step=5000)
        extra["seed"]  = st.number_input("Seed", value=42, step=1)

    return dict(product=product, S0=S0, T=T, r=r, sigma=sigma,
                autocall_lvl=autocall_lvl, coupon=coupon,
                prot_barrier=prot_barrier, **extra)


def render_autocall(params: dict):
    product   = params["product"]
    S0        = params["S0"]
    T         = params["T"]
    r         = params["r"]
    sigma     = params["sigma"]
    autocall_lvl   = params["autocall_lvl"]
    coupon    = params["coupon"]
    prot      = params["prot_barrier"]
    coupon_b  = params.get("coupon_barrier", 0.70)
    paths     = int(params.get("paths", 20_000))
    seed      = int(params.get("seed", 42))

    n_obs = max(1, int(T * 4))
    obs_dates = [round(T * i / n_obs, 4) for i in range(1, n_obs + 1)]

    st.subheader(product)

    if product == "Autocall Incremental":
        result = _AUTO.price(S0, T, r, sigma, autocall_lvl, coupon,
                             obs_dates, prot, 1000.0, paths, seed)
        components = _AUTO.decompose(S0, T, r, sigma, autocall_lvl, coupon,
                                     obs_dates, prot, 1000.0, paths, seed)
    elif product == "Phoenix":
        result = _PHX.price(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                            coupon, obs_dates, 1000.0, paths, seed)
        components = _PHX.decompose(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                                    coupon, obs_dates, 1000.0, paths, seed)
    else:
        result = _PMEM.price(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                             coupon, obs_dates, 1000.0, paths, seed)
        components = _PMEM.decompose(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                                     coupon, obs_dates, 1000.0, paths, seed)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price (per 1000 notional)", f"{result['price']:.2f}")
    c2.metric("% of Notional", f"{result['price'] / 1000 * 100:.1f}%")
    c3.metric("P(autocall)", f"{result['autocall_probability']:.1%}")

    # Decomposition waterfall
    st.markdown("**Decomposition**")
    st.dataframe(pd.DataFrame(components)[["component", "value", "pct"]].rename(
        columns={"component": "Component", "value": "Value ($)", "pct": "% Notional"}
    ).assign(**{"% Notional": lambda df: df["% Notional"].apply(lambda x: f"{x:.1f}%")}),
        use_container_width=True)
    st.plotly_chart(decomposition_waterfall_figure(components), use_container_width=True)

    # Observation schedule fan chart
    obs_prices = result["obs_prices"]  # (n_obs, paths)
    n_show = min(obs_prices.shape[1], 80)
    sample_paths = obs_prices[:, :n_show]

    cb = coupon_b if product != "Autocall Incremental" else None
    st.plotly_chart(
        autocall_schedule_figure(obs_dates, sample_paths, S0,
                                 autocall_lvl, cb, prot),
        use_container_width=True)

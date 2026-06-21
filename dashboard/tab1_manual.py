"""Tab 1 — Manual Parameters: pricing table, Greeks, heatmap."""
import numpy as np
import pandas as pd
import streamlit as st

from pricing.black_scholes import BlackScholes
from pricing.binomial import BinomialTree
from pricing.monte_carlo import MonteCarlo
from dashboard.charts import heatmap_figure

_BS = BlackScholes()
_BT = BinomialTree()
_MC = MonteCarlo()

_HESTON_DEFAULTS = {"v0": 0.04, "kappa": 2.0, "theta": 0.04, "xi": 0.3, "rho": -0.7}


@st.cache_data(show_spinner=False)
def _mc_price(S, K, T, r, sigma, option_type, dynamics, seed=42):
    heston_params = _HESTON_DEFAULTS if dynamics == "Heston" else None
    return _MC.price(
        S, K, T, r, sigma, option_type,
        paths=10_000, dynamics=dynamics.lower(),
        variance_reduction="antithetic",
        heston_params=heston_params,
        seed=seed,
    )


@st.cache_data(show_spinner=False)
def _heatmap_prices(K, T, r, s_min, s_max, sig_min, sig_max):
    s_grid = np.linspace(s_min, s_max, 10)
    sig_grid = np.linspace(sig_min, sig_max, 10)
    call_prices = np.zeros((10, 10))
    put_prices = np.zeros((10, 10))
    for i, sig in enumerate(sig_grid):
        for j, s in enumerate(s_grid):
            call_prices[i, j] = _BS.price(s, K, T, r, sig, "call")
            put_prices[i, j] = _BS.price(s, K, T, r, sig, "put")
    return s_grid, sig_grid, call_prices, put_prices


def render_tab1():
    # ----------------------------------------------------------------
    # Inputs
    # ----------------------------------------------------------------
    st.subheader("Parameters")
    c1, c2, c3, c4, c5 = st.columns(5)
    S = c1.number_input("Spot (S)", min_value=1.0, value=100.0, step=1.0)
    K = c2.number_input("Strike (K)", min_value=1.0, value=100.0, step=1.0)
    T = c3.number_input("Maturity (T, years)", min_value=0.01, value=0.5, step=0.01)
    sigma = c4.number_input("Volatility (σ)", min_value=0.01, max_value=5.0, value=0.20, step=0.01)
    r = c5.number_input("Risk-free rate (r)", min_value=0.0, max_value=1.0, value=0.05, step=0.005)

    dynamics = st.radio(
        "Monte Carlo dynamics", ["GBM", "Heston"], horizontal=True,
        help="GBM = geometric Brownian motion. Heston uses fixed default stochastic vol params.",
    )

    st.divider()

    # ----------------------------------------------------------------
    # Pricing comparison table
    # ----------------------------------------------------------------
    st.subheader("Price Comparison")

    bs_call = _BS.price(S, K, T, r, sigma, "call")
    bs_put = _BS.price(S, K, T, r, sigma, "put")
    bt_call = _BT.price(S, K, T, r, sigma, "call")
    bt_put = _BT.price(S, K, T, r, sigma, "put")

    with st.spinner("Running Monte Carlo..."):
        mc_call = _mc_price(S, K, T, r, sigma, "call", dynamics)
        mc_put = _mc_price(S, K, T, r, sigma, "put", dynamics)

    ee_call = bt_call - bs_call
    ee_put = bt_put - bs_put

    table = pd.DataFrame({
        "Engine": ["Black-Scholes (European)", f"Monte Carlo — {dynamics} (European)", "Binomial CRR (American)"],
        "Call": [f"{bs_call:.4f}", f"{mc_call:.4f}", f"{bt_call:.4f}"],
        "Put": [f"{bs_put:.4f}", f"{mc_put:.4f}", f"{bt_put:.4f}"],
        "Early Exercise Premium (call)": ["—", "—", f"{ee_call:.4f}"],
        "Early Exercise Premium (put)": ["—", "—", f"{ee_put:.4f}"],
    })
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------------------------------
    # Greeks
    # ----------------------------------------------------------------
    st.subheader("Black-Scholes Greeks")

    call_greeks = _BS.greeks(S, K, T, r, sigma, "call")
    put_greeks = _BS.greeks(S, K, T, r, sigma, "put")

    greeks_df = pd.DataFrame({
        "Greek": ["Delta", "Gamma", "Vega (per 1% σ)", "Theta (per day)", "Rho (per 1% r)"],
        "Call": [f"{call_greeks[g]:.6f}" for g in ["delta", "gamma", "vega", "theta", "rho"]],
        "Put":  [f"{put_greeks[g]:.6f}"  for g in ["delta", "gamma", "vega", "theta", "rho"]],
    })
    st.dataframe(greeks_df, use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------------------------------
    # Heatmap
    # ----------------------------------------------------------------
    st.subheader("Price Heatmap (BS only — S vs σ)")
    st.caption("K, T, r are held fixed at the values above. ★ marks the current (S, σ) point.")

    hc1, hc2, hc3, hc4 = st.columns(4)
    s_min = hc1.number_input("Min Spot", value=max(1.0, S * 0.7), step=1.0)
    s_max = hc2.number_input("Max Spot", value=S * 1.3, step=1.0)
    sig_min = hc3.number_input("Min σ", value=0.05, step=0.01)
    sig_max = hc4.number_input("Max σ", value=0.60, step=0.01)

    if s_min >= s_max or sig_min >= sig_max:
        st.warning("Min must be less than Max for both axes.")
        return

    s_grid, sig_grid, call_prices, put_prices = _heatmap_prices(
        K, T, r, s_min, s_max, sig_min, sig_max
    )
    fig = heatmap_figure(s_grid, sig_grid, call_prices, put_prices, S, sigma)
    st.plotly_chart(fig, use_container_width=True)

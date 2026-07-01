"""Tab 3 — Tier D: Multi-Asset Options (basket, worst-of, rainbow)."""
import numpy as np
import pandas as pd
import streamlit as st

from pricing.multi_asset import (
    BasketOption, WorstOfOption, RainbowOption, correlation_sensitivity,
)
from dashboard.charts import multi_asset_corr_sensitivity_figure

MULTI_PRODUCTS = ["Basket Option", "Worst-of Option", "Rainbow Option"]


def render_multi_sidebar() -> dict:
    product = st.sidebar.selectbox("Product", MULTI_PRODUCTS)
    st.sidebar.markdown("---")
    n_assets = st.sidebar.slider("Number of Assets", 2, 5, 2)
    spots = [st.sidebar.number_input(f"S₀ Asset {i+1}", value=100.0, step=1.0, key=f"s{i}")
             for i in range(n_assets)]
    vols  = [st.sidebar.number_input(f"σ Asset {i+1}", value=0.20 + i * 0.02,
                                     step=0.01, format="%.2f", key=f"v{i}")
             for i in range(n_assets)]

    if n_assets == 2:
        rho = st.sidebar.slider("Correlation ρ₁₂", -0.99, 0.99, 0.3, 0.01)
        corr = np.array([[1.0, rho], [rho, 1.0]])
    else:
        rho = st.sidebar.slider("Uniform Correlation ρ", -0.99, 0.99, 0.3, 0.01)
        corr = np.full((n_assets, n_assets), rho)
        np.fill_diagonal(corr, 1.0)

    K = st.sidebar.number_input("Strike K", value=100.0, step=1.0)
    T = st.sidebar.number_input("Maturity T (years)", value=1.0, step=0.25)
    r = st.sidebar.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.3f")
    option_type = st.sidebar.radio("Option Type", ["call", "put"])

    weights = None
    if product == "Basket Option":
        w_raw = [st.sidebar.number_input(f"Weight Asset {i+1}", value=1.0 / n_assets,
                                         step=0.05, format="%.2f", key=f"w{i}")
                 for i in range(n_assets)]
        s = sum(w_raw) or 1.0
        weights = [w / s for w in w_raw]

    with st.sidebar.expander("MC Settings"):
        paths = st.number_input("Paths", value=20_000, step=5000)
        seed  = st.number_input("Seed", value=42, step=1)

    return dict(product=product, spots=spots, vols=vols, corr=corr,
                K=K, T=T, r=r, option_type=option_type,
                weights=weights, rho=rho if n_assets == 2 else rho,
                paths=int(paths), seed=int(seed))


def render_multi(params: dict):
    product     = params["product"]
    spots       = params["spots"]
    vols        = params["vols"]
    corr        = params["corr"]
    K           = params["K"]
    T           = params["T"]
    r           = params["r"]
    ot          = params["option_type"]
    weights     = params["weights"]
    paths       = params["paths"]
    seed        = params["seed"]

    st.subheader(product)

    if product == "Basket Option":
        price = BasketOption().price(spots, weights, K, T, r, vols, corr, ot, paths, seed=seed)
    elif product == "Worst-of Option":
        price = WorstOfOption().price(spots, 1.0, T, r, vols, corr, ot, paths, seed=seed)
    else:
        price = RainbowOption().price(spots, 1.0, T, r, vols, corr, ot, paths, seed=seed)

    st.metric("Option Price", f"{price:.4f}")

    # Asset parameter summary
    asset_rows = [{"Asset": i + 1, "S₀": spots[i], "σ": f"{vols[i]:.1%}"}
                  for i in range(len(spots))]
    if weights:
        for i, row in enumerate(asset_rows):
            row["Weight"] = f"{weights[i]:.1%}"
    st.dataframe(pd.DataFrame(asset_rows), use_container_width=True)

    # Correlation sensitivity chart
    st.markdown("**Correlation Sensitivity**")
    n = len(spots)
    product_key = (
        "basket" if product == "Basket Option" else
        "worst-of" if product == "Worst-of Option" else
        "rainbow"
    )
    with st.spinner("Computing correlation sensitivity..."):
        rho_grid, prices = correlation_sensitivity(
            spots, K, T, r, vols, ot,
            product=product_key,
            weights=weights,
            paths=max(5_000, paths // 4),
            seed=seed,
        )
    st.plotly_chart(
        multi_asset_corr_sensitivity_figure(rho_grid, prices, product),
        use_container_width=True)

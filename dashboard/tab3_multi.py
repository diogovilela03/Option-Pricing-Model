"""Tab 3 — Tier D: Multi-Asset Options (basket, worst-of, rainbow)."""
import numpy as np
import pandas as pd
import streamlit as st

from pricing.multi_asset import (
    BasketOption, WorstOfOption, RainbowOption, correlation_sensitivity,
)
from dashboard.charts import multi_asset_corr_sensitivity_figure

MULTI_PRODUCTS = ["Basket Option", "Worst-of Option", "Rainbow Option"]


def _render_corr_matrix_editor(n_assets: int) -> np.ndarray:
    """Editable pairwise correlation matrix for n_assets >= 3.

    Only the upper triangle is meaningful — it's mirrored onto the lower
    triangle and the diagonal is forced to 1.0, so the user only has to
    fill in each pair once and can't create an asymmetric input.
    """
    labels = [f"Asset {i+1}" for i in range(n_assets)]
    default = pd.DataFrame(np.eye(n_assets), index=labels, columns=labels)

    st.sidebar.caption("Edit pairwise correlations (upper triangle; symmetrized automatically).")
    edited = st.sidebar.data_editor(
        default, key="corr_matrix_editor",
        column_config={c: st.column_config.NumberColumn(c, min_value=-0.99, max_value=0.99, step=0.05)
                       for c in labels},
    )

    raw = edited.to_numpy(dtype=float)
    corr = np.triu(raw, k=1)
    corr = corr + corr.T
    np.fill_diagonal(corr, 1.0)

    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        st.sidebar.error(
            "This correlation matrix is not positive semi-definite (mathematically "
            "invalid — e.g. asset A and B highly correlated, B and C highly "
            "correlated, but A and C anti-correlated is impossible). "
            "Falling back to zero correlation until you fix the entries above."
        )
        corr = np.eye(n_assets)

    return corr


def render_multi_sidebar() -> dict:
    product = st.sidebar.selectbox("Product", MULTI_PRODUCTS)
    st.sidebar.markdown("---")
    n_assets = st.sidebar.slider("Number of Assets", 2, 5, 2)
    spots = [st.sidebar.number_input(f"S₀ Asset {i+1}", value=100.0, step=1.0, key=f"s{i}")
             for i in range(n_assets)]
    vols  = [st.sidebar.number_input(f"σ Asset {i+1}", value=0.20 + i * 0.02,
                                     step=0.01, format="%.2f", key=f"v{i}")
             for i in range(n_assets)]

    rho = None
    if n_assets == 2:
        rho = st.sidebar.slider("Correlation ρ₁₂", -0.99, 0.99, 0.3, 0.01)
        corr = np.array([[1.0, rho], [rho, 1.0]])
    else:
        corr_mode = st.sidebar.radio(
            "Correlation Input", ["Uniform ρ", "Custom Matrix"],
            help="Uniform applies one correlation to every pair. Custom "
                 "lets each pair have its own correlation (n>=3).")
        if corr_mode == "Uniform ρ":
            rho = st.sidebar.slider("Uniform Correlation ρ", -0.99, 0.99, 0.3, 0.01)
            corr = np.full((n_assets, n_assets), rho)
            np.fill_diagonal(corr, 1.0)
        else:
            corr = _render_corr_matrix_editor(n_assets)

    if product == "Basket Option":
        K = st.sidebar.number_input("Strike K", value=100.0, step=1.0)
        K_relative = None
    else:
        K_pct = st.sidebar.number_input(
            "Strike (% of Initial)", value=100.0, step=1.0, format="%.1f",
            help="Worst-of/Rainbow payoffs are on relative performance "
                 "(terminal/initial), so the strike is a percentage, not "
                 "an absolute price.")
        K = K_pct  # kept for display/summary purposes
        K_relative = K_pct / 100.0
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
                K=K, K_relative=K_relative, T=T, r=r, option_type=option_type,
                weights=weights, rho=rho,
                paths=int(paths), seed=int(seed))


def render_multi(params: dict):
    product     = params["product"]
    spots       = params["spots"]
    vols        = params["vols"]
    corr        = params["corr"]
    K           = params["K"]
    K_relative  = params["K_relative"]
    T           = params["T"]
    r           = params["r"]
    ot          = params["option_type"]
    weights     = params["weights"]
    paths       = params["paths"]
    seed        = params["seed"]

    st.subheader(product)

    # Basket prices on absolute strike K; Worst-of/Rainbow price on relative
    # performance, so they need K_relative (e.g. K_pct=100 -> 1.0), not K.
    sensitivity_K = K if product == "Basket Option" else K_relative

    if product == "Basket Option":
        price = BasketOption().price(spots, weights, K, T, r, vols, corr, ot, paths, seed=seed)
    elif product == "Worst-of Option":
        price = WorstOfOption().price(spots, K_relative, T, r, vols, corr, ot, paths, seed=seed)
    else:
        price = RainbowOption().price(spots, K_relative, T, r, vols, corr, ot, paths, seed=seed)

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
            spots, sensitivity_K, T, r, vols, ot,
            product=product_key,
            weights=weights,
            paths=max(5_000, paths // 4),
            seed=seed,
        )
    st.plotly_chart(
        multi_asset_corr_sensitivity_figure(rho_grid, prices, product),
        use_container_width=True)

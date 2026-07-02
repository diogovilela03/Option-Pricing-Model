"""Tab 1 — Manual Parameters: pricing table, Greeks, heatmap."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pricing.black_scholes import BlackScholes
from pricing.binomial import BinomialTree
from pricing.monte_carlo import MonteCarlo
from dashboard.charts import heatmap_figure, greek_sensitivity_figure

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


_SENSITIVITY_CONFIGS = {
    "Spot (S)": {
        "key": "S",
        "xlabel": "Spot Price",
        "range_fn": lambda S, K: (max(1.0, K * 0.4), K * 2.0),
    },
    "Volatility (σ)": {
        "key": "sigma",
        "xlabel": "Volatility (σ)",
        "range": (0.01, 1.0),
    },
    "Time to Maturity (T)": {
        "key": "T",
        "xlabel": "Time to Maturity (years)",
        "range": (0.01, 2.0),
    },
}

_STRIKE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # lower / ATM / upper

_ALL_GREEKS = ["delta", "gamma", "vega", "theta", "rho", "volga", "vanna", "charm"]


def _compute_sensitivity(S, K, T, r, sigma, vary_label):
    cfg = _SENSITIVITY_CONFIGS[vary_label]
    N = 80
    lo, hi = cfg["range_fn"](S, K) if "range_fn" in cfg else cfg["range"]
    xs = np.linspace(lo, hi, N)
    key = cfg["key"]

    # Three strikes: −20%S, ATM, +20%S
    offset = 0.20 * S
    strikes = [
        (max(1.0, K - offset), f"K = {K - offset:.0f} (−20%)"),
        (K,                     f"K = {K:.0f} (ATM)"),
        (K + offset,            f"K = {K + offset:.0f} (+20%)"),
    ]

    series = []
    for (strike, label), color in zip(strikes, _STRIKE_COLORS):
        call_g = {g: [] for g in _ALL_GREEKS}
        put_g  = {g: [] for g in _ALL_GREEKS}
        for x in xs:
            kw = {"S": S, "K": strike, "T": T, "r": r, "sigma": sigma}
            kw[key] = x
            kw["T"] = max(kw["T"], 1e-6)  # guard against T→0
            cg = _BS.greeks(**kw, option_type="call")
            pg = _BS.greeks(**kw, option_type="put")
            for g in _ALL_GREEKS:
                call_g[g].append(cg[g])
                put_g[g].append(pg[g])
        series.append({"label": label, "color": color,
                        "call_greeks": call_g, "put_greeks": put_g})

    current_x = {"S": S, "K": K, "T": T, "sigma": sigma}[key]
    return xs, series, current_x, cfg["xlabel"]


def render_tab1_sidebar() -> dict:
    """Render Tab 1 sidebar inputs and return their values."""
    st.sidebar.subheader("Option Parameters")
    S     = st.sidebar.number_input("Spot (S)",            min_value=1.0,  value=100.0, step=1.0,  key="t1_S")
    K     = st.sidebar.number_input("Strike (K)",          min_value=1.0,  value=100.0, step=1.0,  key="t1_K")
    T     = st.sidebar.number_input("Maturity (T, years)", min_value=0.01, value=0.5,   step=0.01, key="t1_T")
    sigma = st.sidebar.number_input("Volatility (σ)",      min_value=0.01, max_value=5.0, value=0.20, step=0.01, key="t1_sigma")
    r     = st.sidebar.number_input("Risk-free rate (r)",  min_value=0.0,  max_value=1.0, value=0.05, step=0.005, key="t1_r")

    dynamics = st.sidebar.radio(
        "MC dynamics", ["GBM", "Heston"],
        help="GBM = geometric Brownian motion. Heston uses fixed default stochastic vol params.",
        key="t1_dynamics",
    )

    st.sidebar.subheader("Heatmap Ranges")
    s_min   = st.sidebar.number_input("Min Spot",  value=max(1.0, round(S * 0.7, 1)), step=1.0,  key="t1_smin")
    s_max   = st.sidebar.number_input("Max Spot",  value=round(S * 1.3, 1),           step=1.0,  key="t1_smax")
    sig_min = st.sidebar.number_input("Min σ",     value=0.05,  step=0.01, key="t1_sigmin")
    sig_max = st.sidebar.number_input("Max σ",     value=0.60,  step=0.01, key="t1_sigmax")

    return dict(S=S, K=K, T=T, sigma=sigma, r=r, dynamics=dynamics,
                s_min=s_min, s_max=s_max, sig_min=sig_min, sig_max=sig_max)


def render_tab1(params: dict):
    S, K, T, sigma, r = params["S"], params["K"], params["T"], params["sigma"], params["r"]
    dynamics = params["dynamics"]

    # ----------------------------------------------------------------
    # Pricing comparison table
    # ----------------------------------------------------------------
    st.subheader("Price Comparison")

    bs_call = _BS.price(S, K, T, r, sigma, "call")
    bs_put  = _BS.price(S, K, T, r, sigma, "put")
    bt_call = _BT.price(S, K, T, r, sigma, "call")
    bt_put  = _BT.price(S, K, T, r, sigma, "put")

    with st.spinner("Running Monte Carlo..."):
        mc_call = _mc_price(S, K, T, r, sigma, "call", dynamics)
        mc_put  = _mc_price(S, K, T, r, sigma, "put",  dynamics)

    table = pd.DataFrame({
        "Engine": ["Black-Scholes (European)", f"Monte Carlo — {dynamics} (European)", "Binomial CRR (American)"],
        "Call":   [f"{bs_call:.4f}", f"{mc_call:.4f}", f"{bt_call:.4f}"],
        "Put":    [f"{bs_put:.4f}",  f"{mc_put:.4f}",  f"{bt_put:.4f}"],
        "Early Exercise Premium (call)": ["—", "—", f"{bt_call - bs_call:.4f}"],
        "Early Exercise Premium (put)":  ["—", "—", f"{bt_put  - bs_put:.4f}"],
    })
    st.dataframe(table, width='stretch', hide_index=True)

    st.divider()

    # ----------------------------------------------------------------
    # Greeks
    # ----------------------------------------------------------------
    st.subheader("Black-Scholes Greeks")

    call_greeks = _BS.greeks(S, K, T, r, sigma, "call")
    put_greeks  = _BS.greeks(S, K, T, r, sigma, "put")

    greek_rows = [
        ("delta", "Delta"),
        ("gamma", "Gamma"),
        ("vega",  "Vega (per 1% σ)"),
        ("theta", "Theta (per day)"),
        ("rho",   "Rho (per 1% r)"),
        ("volga", "Volga ∂²V/∂σ² (per 1% σ)"),
        ("vanna", "Vanna ∂²V/∂S∂σ (per 1% σ)"),
        ("charm", "Charm ∂Δ/∂T (per day)"),
    ]
    greeks_df = pd.DataFrame({
        "Greek": [label for _, label in greek_rows],
        "Call":  [f"{call_greeks[g]:.6f}" for g, _ in greek_rows],
        "Put":   [f"{put_greeks[g]:.6f}"  for g, _ in greek_rows],
    })
    st.dataframe(greeks_df, width='stretch', hide_index=True)

    st.divider()

    # ----------------------------------------------------------------
    # Premium & P&L vs Spot
    # ----------------------------------------------------------------
    st.subheader("Premium & P&L vs Spot (BS only)")
    st.caption("K, T, r, σ held fixed at sidebar values. P&L assumes the position "
              "was opened today at the sidebar spot, for the premium shown above.")

    s_grid = np.linspace(max(S * 0.5, 0.01), S * 1.5, 200)
    call_premium = np.array([_BS.price(s, K, T, r, sigma, "call") for s in s_grid])
    put_premium  = np.array([_BS.price(s, K, T, r, sigma, "put")  for s in s_grid])
    call_pnl = call_premium - bs_call
    put_pnl  = put_premium  - bs_put

    col_prem, col_pnl = st.columns(2)
    with col_prem:
        fig_prem = go.Figure()
        fig_prem.add_trace(go.Scatter(x=s_grid, y=call_premium, mode="lines",
                                      line=dict(color="#1f77b4", width=2), name="Call"))
        fig_prem.add_trace(go.Scatter(x=s_grid, y=put_premium, mode="lines",
                                      line=dict(color="#d62728", width=2), name="Put"))
        fig_prem.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text=f"S={S:.0f}")
        fig_prem.update_layout(title="Premium vs Spot", xaxis_title="Spot",
                               yaxis_title="Premium", height=380,
                               legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_prem, width='stretch')
    with col_pnl:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(x=s_grid, y=call_pnl, mode="lines",
                                     line=dict(color="#1f77b4", width=2), name="Call"))
        fig_pnl.add_trace(go.Scatter(x=s_grid, y=put_pnl, mode="lines",
                                     line=dict(color="#d62728", width=2), name="Put"))
        fig_pnl.add_hline(y=0, line_dash="dot", line_color="gray")
        fig_pnl.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text=f"S={S:.0f}")
        fig_pnl.update_layout(title="P&L vs Spot (long position)", xaxis_title="Spot",
                              yaxis_title="P&L", height=380,
                              legend=dict(orientation="h", y=-0.22))
        st.plotly_chart(fig_pnl, width='stretch')

    st.divider()

    # ----------------------------------------------------------------
    # Heatmap
    # ----------------------------------------------------------------
    st.subheader("Price Heatmap (BS only — S vs σ)")
    st.caption("K, T, r held fixed at sidebar values. ★ marks the current (S, σ) point.")

    s_min, s_max   = params["s_min"], params["s_max"]
    sig_min, sig_max = params["sig_min"], params["sig_max"]

    if s_min >= s_max or sig_min >= sig_max:
        st.warning("Heatmap: Min must be less than Max for both axes.")
        return

    s_grid, sig_grid, call_prices, put_prices = _heatmap_prices(
        K, T, r, s_min, s_max, sig_min, sig_max
    )
    fig = heatmap_figure(s_grid, sig_grid, call_prices, put_prices, S, sigma)
    st.plotly_chart(fig, width='stretch')

    st.divider()

    # ----------------------------------------------------------------
    # Greek Sensitivities
    # ----------------------------------------------------------------
    st.subheader("Greek Sensitivities")
    st.caption(
        "Each subplot shows how a Greek varies as one parameter is swept, "
        "with all others held fixed at the sidebar values. "
        "The dotted vertical line marks the current sidebar value."
    )

    _GREEK_LABELS = {
        "delta": "Delta",
        "gamma": "Gamma",
        "vega":  "Vega (per 1% σ)",
        "theta": "Theta (per day)",
        "rho":   "Rho (per 1% r)",
        "volga": "Volga ∂²V/∂σ²",
        "vanna": "Vanna ∂²V/∂S∂σ",
        "charm": "Charm ∂Δ/∂T (per day)",
    }

    sel_col, vary_col = st.columns(2)
    with sel_col:
        greek_key = st.selectbox(
            "Greek", list(_GREEK_LABELS.keys()),
            format_func=lambda k: _GREEK_LABELS[k],
            key="t1_greek",
        )
    with vary_col:
        vary_label = st.selectbox(
            "Vary parameter",
            list(_SENSITIVITY_CONFIGS.keys()),
            key="t1_vary",
        )

    show_all = st.checkbox("Compare ITM / ATM / OTM strikes", value=True, key="t1_show_all")

    xs, series, current_x, xlabel = _compute_sensitivity(S, K, T, r, sigma, vary_label)

    if show_all:
        active_series = series
        st.caption(
            f"3 strikes — K−20%S: **{K - 0.2*S:.1f}**,  ATM: **{K:.1f}**,  K+20%S: **{K + 0.2*S:.1f}**. "
            "Dotted line = current sidebar value."
        )
    else:
        active_series = [series[1]]  # ATM only (middle series)
        st.caption(
            f"Single strike — ATM: **{K:.1f}**. "
            "Dotted line = current sidebar value."
        )

    fig_g = greek_sensitivity_figure(
        xs, active_series, current_x, xlabel,
        greek=greek_key, greek_label=_GREEK_LABELS[greek_key],
    )
    st.plotly_chart(fig_g, width='stretch')

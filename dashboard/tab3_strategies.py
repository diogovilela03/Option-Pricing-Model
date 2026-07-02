"""Tab 3 — Tier A: Option Strategies using opstrat for P&L diagrams."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from pricing.black_scholes import BlackScholes

_BS = BlackScholes()


_STRATEGIES = [
    "Long Call", "Long Put",
    "Bull Call Spread", "Bear Put Spread",
    "Straddle", "Strangle",
    "Risk Reversal", "Butterfly (Call)",
    "Condor", "Iron Condor",
]

_DESCRIPTIONS = {
    "Long Call":         "Buy a call — unlimited upside, premium at risk.",
    "Long Put":          "Buy a put — profits as asset falls, limited risk.",
    "Bull Call Spread":  "Buy lower call, sell higher call — capped upside, lower cost.",
    "Bear Put Spread":   "Buy higher put, sell lower put — profits on decline.",
    "Straddle":          "Buy call + put at same strike — profits from big moves either way.",
    "Strangle":          "Buy OTM call + OTM put — cheaper than straddle, wider breakevens.",
    "Risk Reversal":     "Buy OTM call, sell OTM put — directional with skew exposure.",
    "Butterfly (Call)":  "Sell two ATM calls, buy wings — profits in quiet market.",
    "Condor":            "Wider body than butterfly — profits in wider range.",
    "Iron Condor":       "Sell OTM strangle, buy wider strangle — income with capped risk.",
}


def _legs_for(strategy: str, S: float, K: float, T: float, r: float, sigma: float):
    """Return list of opstrat leg dicts and descriptions for the chosen strategy."""
    def pr(k, ot): return _BS.price(S, k, T, r, sigma, ot)

    dK = S * 0.05  # 5% OTM wings

    if strategy == "Long Call":
        return [{"op_type": "c", "strike": K, "tr_type": "b", "op_pr": pr(K, "call")}]

    if strategy == "Long Put":
        return [{"op_type": "p", "strike": K, "tr_type": "b", "op_pr": pr(K, "put")}]

    K1, K2 = K - dK, K + dK
    if strategy == "Bull Call Spread":
        return [
            {"op_type": "c", "strike": K1, "tr_type": "b", "op_pr": pr(K1, "call")},
            {"op_type": "c", "strike": K2, "tr_type": "s", "op_pr": pr(K2, "call")},
        ]

    if strategy == "Bear Put Spread":
        return [
            {"op_type": "p", "strike": K2, "tr_type": "b", "op_pr": pr(K2, "put")},
            {"op_type": "p", "strike": K1, "tr_type": "s", "op_pr": pr(K1, "put")},
        ]

    if strategy == "Straddle":
        return [
            {"op_type": "c", "strike": K, "tr_type": "b", "op_pr": pr(K, "call")},
            {"op_type": "p", "strike": K, "tr_type": "b", "op_pr": pr(K, "put")},
        ]

    if strategy == "Strangle":
        return [
            {"op_type": "c", "strike": K2, "tr_type": "b", "op_pr": pr(K2, "call")},
            {"op_type": "p", "strike": K1, "tr_type": "b", "op_pr": pr(K1, "put")},
        ]

    if strategy == "Risk Reversal":
        return [
            {"op_type": "c", "strike": K2, "tr_type": "b", "op_pr": pr(K2, "call")},
            {"op_type": "p", "strike": K1, "tr_type": "s", "op_pr": pr(K1, "put")},
        ]

    K0, K3 = K - 2 * dK, K + 2 * dK
    if strategy == "Butterfly (Call)":
        return [
            {"op_type": "c", "strike": K0, "tr_type": "b", "op_pr": pr(K0, "call")},
            {"op_type": "c", "strike": K,  "tr_type": "s", "op_pr": pr(K, "call")},
            {"op_type": "c", "strike": K,  "tr_type": "s", "op_pr": pr(K, "call")},
            {"op_type": "c", "strike": K3, "tr_type": "b", "op_pr": pr(K3, "call")},
        ]

    if strategy == "Condor":
        return [
            {"op_type": "c", "strike": K0, "tr_type": "b", "op_pr": pr(K0, "call")},
            {"op_type": "c", "strike": K1, "tr_type": "s", "op_pr": pr(K1, "call")},
            {"op_type": "c", "strike": K2, "tr_type": "s", "op_pr": pr(K2, "call")},
            {"op_type": "c", "strike": K3, "tr_type": "b", "op_pr": pr(K3, "call")},
        ]

    # Iron Condor
    return [
        {"op_type": "p", "strike": K0, "tr_type": "b", "op_pr": pr(K0, "put")},
        {"op_type": "p", "strike": K1, "tr_type": "s", "op_pr": pr(K1, "put")},
        {"op_type": "c", "strike": K2, "tr_type": "s", "op_pr": pr(K2, "call")},
        {"op_type": "c", "strike": K3, "tr_type": "b", "op_pr": pr(K3, "call")},
    ]


def render_strategies_sidebar() -> dict:
    strategy = st.sidebar.selectbox("Strategy", _STRATEGIES)
    st.sidebar.markdown("---")
    S = st.sidebar.number_input("Spot (S)", value=100.0, min_value=1.0, step=1.0)
    K = st.sidebar.number_input("Strike (K)", value=100.0, min_value=1.0, step=1.0)
    T = st.sidebar.number_input("Maturity T (years)", value=0.5, min_value=0.01, step=0.05)
    r = st.sidebar.number_input("Risk-free rate r", value=0.05, min_value=0.0, step=0.005, format="%.3f")
    sigma = st.sidebar.number_input("Volatility σ", value=0.20, min_value=0.01, step=0.01, format="%.2f")
    return dict(strategy=strategy, S=S, K=K, T=T, r=r, sigma=sigma)


def render_strategies(params: dict):
    strategy = params["strategy"]
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]

    st.subheader(strategy)
    st.caption(_DESCRIPTIONS[strategy])

    legs = _legs_for(strategy, S, K, T, r, sigma)

    # Leg table — premium shown as signed cash flow: negative = paid (buy),
    # positive = received (sell). opstrat itself needs the unsigned op_pr
    # (it applies the buy/sell sign internally), so legs stay untouched.
    def _signed(leg):
        return -leg["op_pr"] if leg["tr_type"] == "b" else leg["op_pr"]

    rows = []
    for leg in legs:
        ot = "Call" if leg["op_type"] == "c" else "Put"
        side = "Buy" if leg["tr_type"] == "b" else "Sell"
        rows.append({"Option": ot, "Strike": f"{leg['strike']:.2f}",
                     "Side": side, "Premium": f"{_signed(leg):.4f}"})

    net_cost = sum(_signed(leg) for leg in legs)
    rows.append({"Option": "", "Strike": "", "Side": "Net cost", "Premium": f"{net_cost:.4f}"})

    import pandas as pd
    st.table(pd.DataFrame(rows))
    st.caption("Premium and net cost are signed cash flows: negative = paid (buy), positive = received (sell).")

    # P&L chart via opstrat
    try:
        import opstrat as op
        import matplotlib.pyplot as plt

        # opstrat's spot_range is a +/- percentage around spot (int/float),
        # not a [low, high] list — 40 gives roughly 0.6x-1.4x of spot.
        op.multi_plotter(
            op_list=legs,
            spot=S,
            spot_range=40,
            save=False,
        )
        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.warning(f"opstrat chart unavailable: {e}")
        # Fallback: manual payoff chart
        s_vals = np.linspace(S * 0.6, S * 1.4, 300)
        pnl = np.zeros(len(s_vals))
        for leg in legs:
            k = leg["strike"]
            pr = leg["op_pr"]
            sign = 1 if leg["tr_type"] == "b" else -1
            if leg["op_type"] == "c":
                pnl += sign * (np.maximum(s_vals - k, 0) - pr)
            else:
                pnl += sign * (np.maximum(k - s_vals, 0) - pr)
        gofig = go.Figure()
        gofig.add_trace(go.Scatter(x=s_vals, y=pnl, mode="lines",
                                   line=dict(color="#1f77b4", width=2)))
        gofig.add_hline(y=0, line_dash="dash", line_color="gray")
        gofig.add_vline(x=S, line_dash="dot", line_color="gray")
        gofig.update_layout(xaxis_title="Spot at Expiry", yaxis_title="P&L", height=380)
        st.plotly_chart(gofig, use_container_width=True)

    st.divider()
    st.markdown("**Greek vs Spot**")
    greek_choice = st.selectbox(
        "Greek", ["delta", "gamma", "vega", "theta", "rho"], key="strategy_greek")

    s_vals = np.linspace(S * 0.6, S * 1.4, 120)
    greek_vals = np.zeros(len(s_vals))
    for leg in legs:
        ot_leg = "call" if leg["op_type"] == "c" else "put"
        sign = 1 if leg["tr_type"] == "b" else -1
        for i, s in enumerate(s_vals):
            greek_vals[i] += sign * _BS.greeks(s, leg["strike"], T, r, sigma, ot_leg)[greek_choice]

    gfig = go.Figure()
    gfig.add_trace(go.Scatter(x=s_vals, y=greek_vals, mode="lines",
                              line=dict(color="#9467bd", width=2.5)))
    gfig.add_hline(y=0, line_dash="dash", line_color="gray")
    gfig.add_vline(x=S, line_dash="dot", line_color="gray", annotation_text=f"S={S:.0f}")
    gfig.update_layout(xaxis_title="Spot (today)", yaxis_title=greek_choice.capitalize(), height=380)
    st.plotly_chart(gfig, use_container_width=True)
    st.caption("Analytic per-leg Black-Scholes Greeks, signed by side and summed across legs.")

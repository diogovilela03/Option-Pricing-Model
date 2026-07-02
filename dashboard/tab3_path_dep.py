"""Tab 3 — Tier B: Path-Dependent & Structured Products."""
import math

import numpy as np
import pandas as pd
import streamlit as st

from pricing.black_scholes import BlackScholes
from pricing.digital import DigitalOption, call_spread_approximation
from pricing.asian import GeometricAsian, asian_mc_price, asian_running_avg_paths
from pricing.barrier import BarrierOption, barrier_mc_price, barrier_delta_profile, get_mc_paths_for_display
from pricing.double_barrier import DoubleBarrierOption, double_barrier_mc_price
from pricing.quanto import QuantoOption
from pricing.structured import (
    ReverseConvertible, BarrierReverseConvertible,
    DiscountCertificate, BonusCertificate,
    AirbagCertificate, TwinWinCertificate,
)
from pricing.greeks_fd import fd_greeks_profile
from dashboard.charts import (
    exotic_payoff_figure, barrier_delta_figure, barrier_mc_paths_figure,
    asian_running_avg_figure, asian_vol_comparison_figure,
    digital_call_spread_figure, digital_delta_vs_T_figure,
    decomposition_waterfall_figure, premium_vs_spot_figure, greeks_grid_figure,
)

_BS = BlackScholes()
_dig = DigitalOption()
_geo = GeometricAsian()
_bar = BarrierOption()
_dbl = DoubleBarrierOption()
_qto = QuantoOption()


PATH_DEP_PRODUCTS = [
    "Digital — Cash-or-Nothing",
    "Digital — Asset-or-Nothing",
    "Asian — Geometric (CF)",
    "Asian — Arithmetic (MC)",
    "Knock-In / Knock-Out",
    "Double Barrier",
    "Quanto Call/Put",
]

STRUCTURED_PRODUCTS = [
    "Reverse Convertible",
    "Barrier Reverse Convertible",
    "Discount Certificate",
    "Bonus Certificate",
    "Airbag Certificate",
    "Twin-Win Certificate",
]


def render_path_dep_sidebar(sub_category: str) -> dict:
    products = PATH_DEP_PRODUCTS if sub_category == "Path-Dependent" else STRUCTURED_PRODUCTS
    product = st.sidebar.selectbox("Product", products)
    st.sidebar.markdown("---")
    S = st.sidebar.number_input("Spot (S)", value=100.0, min_value=1.0, step=1.0)
    K = st.sidebar.number_input("Strike / Level (K)", value=100.0, min_value=1.0, step=1.0)
    T = st.sidebar.number_input("Maturity T (years)", value=0.5, min_value=0.01, step=0.05)
    r = st.sidebar.number_input("Risk-free rate r", value=0.05, min_value=0.0, step=0.005, format="%.3f")
    sigma = st.sidebar.number_input("Volatility σ", value=0.20, min_value=0.01, step=0.01, format="%.2f")
    option_type = st.sidebar.radio("Option Type", ["call", "put"])

    extra = {}
    if "Knock-In" in product or "Knock-Out" in product:
        extra["barrier"] = st.sidebar.number_input("Barrier H", value=90.0, step=1.0)
        extra["barrier_type"] = st.sidebar.selectbox(
            "Barrier Type", ["down-and-out", "down-and-in", "up-and-out", "up-and-in"])
    if product == "Double Barrier":
        extra["H_lower"] = st.sidebar.number_input("Lower Barrier H_L", value=85.0, step=1.0)
        extra["H_upper"] = st.sidebar.number_input("Upper Barrier H_U", value=120.0, step=1.0)
    if product == "Quanto Call/Put":
        extra["r_f"] = st.sidebar.number_input("Foreign rate r_f", value=0.03, step=0.005, format="%.3f")
        extra["sigma_FX"] = st.sidebar.number_input("FX Vol σ_FX", value=0.15, step=0.01, format="%.2f")
        extra["rho"] = st.sidebar.slider("Correlation ρ", -1.0, 1.0, -0.3, 0.05)
        extra["Q0"] = st.sidebar.number_input("FX Rate Q₀", value=1.0, step=0.01)
    if product == "Reverse Convertible":
        extra["coupon_rate"] = st.sidebar.number_input("Coupon Rate", value=0.10, step=0.01, format="%.2f")
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)
    if product == "Barrier Reverse Convertible":
        extra["coupon_rate"] = st.sidebar.number_input("Coupon Rate", value=0.10, step=0.01, format="%.2f")
        extra["barrier_brc"] = st.sidebar.number_input("DI Barrier H", value=75.0, step=1.0)
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)
    if product == "Discount Certificate":
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)
    if product == "Bonus Certificate":
        extra["K_bonus"] = st.sidebar.number_input("Bonus Level", value=110.0, step=1.0)
        extra["barrier_bonus"] = st.sidebar.number_input("Barrier H", value=80.0, step=1.0)
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)
    if product == "Airbag Certificate":
        extra["participation"] = st.sidebar.number_input("Participation %", value=1.0, step=0.05, format="%.2f")
        extra["K_floor"] = st.sidebar.number_input("Floor Level K_floor", value=90.0, step=1.0)
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)
    if product == "Twin-Win Certificate":
        extra["barrier_tw"] = st.sidebar.number_input("Barrier H", value=70.0, step=1.0)
        extra["notional"] = st.sidebar.number_input("Notional", value=1000.0, step=100.0)

    with st.sidebar.expander("MC Settings"):
        extra["paths"] = st.number_input("Paths", value=20_000, step=5000)
        extra["steps"] = st.number_input("Steps", value=126, step=10)
        extra["seed"] = st.number_input("Seed", value=42, step=1)

    return dict(product=product, S=S, K=K, T=T, r=r, sigma=sigma,
                option_type=option_type, **extra)


# ─────────────────────── shared Premium + Greeks panel ────────────────────────

def _render_premium_and_greeks(price_fn, S: float, K: float, T: float, r: float,
                               sigma: float, ot: str, label: str, n_grid: int = 40,
                               show_vanilla: bool = True):
    """Premium-vs-Spot + Greeks-vs-Spot (delta/gamma/vega/theta/rho), all via
    finite differences on price_fn(S, K, T, r, sigma) -> float.

    show_vanilla=False skips the BS reference line — structured products are
    priced on notional (~$1000s), not a comparable per-share option premium.
    """
    s_grid = np.linspace(S * 0.6, S * 1.4, n_grid)
    premium = np.array([price_fn(s, K, T, r, sigma) for s in s_grid])
    vanilla = np.array([_BS.price(s, K, T, r, sigma, ot) for s in s_grid]) if show_vanilla else None

    st.markdown("**Premium vs Spot**")
    st.plotly_chart(premium_vs_spot_figure(s_grid, premium, vanilla, label, S),
                    use_container_width=True)

    st.markdown("**Greeks vs Spot** (finite-difference)")
    greeks = fd_greeks_profile(price_fn, s_grid, K, T, r, sigma)
    st.plotly_chart(greeks_grid_figure(s_grid, greeks, S), use_container_width=True)


# ─────────────────────── individual product renderers ────────────────────────

def _render_digital(params: dict, digital_type: str):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    ot = params["option_type"]

    price = _dig.price(S, K, T, r, sigma, ot, digital_type)
    vanilla = _BS.price(S, K, T, r, sigma, ot)
    st.metric("Digital Price", f"{price:.4f}")
    st.metric("Vanilla Price (ref)", f"{vanilla:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        # Payoff chart
        s_grid = np.linspace(S * 0.5, S * 1.5, 300)
        if digital_type == "cash-or-nothing":
            exotic_payoffs = np.where(s_grid > K, math.exp(-r * T), 0.0) if ot == "call" \
                else np.where(s_grid < K, math.exp(-r * T), 0.0)
        else:
            exotic_payoffs = np.where(s_grid > K, s_grid * math.exp(-r * T), 0.0) if ot == "call" \
                else np.where(s_grid < K, s_grid * math.exp(-r * T), 0.0)
        vanilla_payoffs = np.maximum((1 if ot == "call" else -1) * (s_grid - K), 0) * math.exp(-r * T)
        st.plotly_chart(
            exotic_payoff_figure(s_grid, exotic_payoffs, vanilla_payoffs,
                                 f"{digital_type} {ot}", S),
            use_container_width=True)

    with col2:
        # Call-spread convergence (cash-or-nothing only for calls)
        dK_grid = np.logspace(-3, 0, 30)
        cs_prices = np.array([call_spread_approximation(S, K, T, r, sigma, dK) for dK in dK_grid])
        dig_price = _dig.price(S, K, T, r, sigma, "call", "cash-or-nothing")
        st.plotly_chart(digital_call_spread_figure(dK_grid, cs_prices, dig_price),
                        use_container_width=True)

    # Delta vs T chart
    T_grid = np.linspace(0.02, 2.0, 80)
    h = 1e-4
    deltas = np.array([
        (_dig.price(S * (1 + h), K, t, r, sigma, "call", "cash-or-nothing") -
         _dig.price(S * (1 - h), K, t, r, sigma, "call", "cash-or-nothing")) / (2 * S * h)
        for t in T_grid
    ])
    st.plotly_chart(digital_delta_vs_T_figure(T_grid, np.clip(deltas, -5, 5), T),
                    use_container_width=True)

    st.divider()
    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: _dig.price(s, k, t, r_, sig, ot, digital_type),
        S, K, T, r, sigma, ot, f"Digital ({digital_type})")


def _render_asian(params: dict, averaging: str):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    ot = params["option_type"]
    paths = int(params.get("paths", 20_000))
    seed = int(params.get("seed", 42))

    geo_price = _geo.price(S, K, T, r, sigma, ot)
    mc_price = asian_mc_price(S, K, T, r, sigma, ot, averaging=averaging,
                               paths=paths, seed=seed)
    vanilla = _BS.price(S, K, T, r, sigma, ot)

    c1, c2, c3 = st.columns(3)
    c1.metric("Geometric CF", f"{geo_price:.4f}")
    c2.metric(f"Arithmetic MC ({paths:,})", f"{mc_price:.4f}")
    c3.metric("Vanilla (ref)", f"{vanilla:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        tg, pp, ap = asian_running_avg_paths(S, T, r, sigma, n_display_paths=20, seed=seed)
        st.plotly_chart(asian_running_avg_figure(tg, pp, ap, K), use_container_width=True)
    with col2:
        sig_grid = np.linspace(0.05, 0.50, 25)
        v_prices = np.array([_BS.price(S, K, T, r, s, ot) for s in sig_grid])
        g_prices = np.array([_geo.price(S, K, T, r, s, ot) for s in sig_grid])
        a_prices = np.array([asian_mc_price(S, K, T, r, s, ot, paths=5_000, seed=seed) for s in sig_grid])
        st.plotly_chart(asian_vol_comparison_figure(sig_grid, v_prices, g_prices, a_prices, sigma),
                        use_container_width=True)

    st.divider()
    st.caption("Premium/Greeks below use the fast geometric-CF price as a proxy "
              "for both averaging conventions — arithmetic MC is too noisy for "
              "stable finite-difference Greeks at interactive path counts.")
    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: _geo.price(s, k, t, r_, sig, ot),
        S, K, T, r, sigma, ot, f"Asian ({averaging})")


def _render_barrier(params: dict):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    ot = params["option_type"]
    H = params.get("barrier", 90.0)
    bt = params.get("barrier_type", "down-and-out")
    paths = int(params.get("paths", 20_000))
    steps = int(params.get("steps", 126))
    seed = int(params.get("seed", 42))

    analytic = _bar.price(S, K, T, r, sigma, ot, H, bt)
    mc = barrier_mc_price(S, K, T, r, sigma, ot, H, bt, paths=paths, steps=steps, seed=seed)
    vanilla = _BS.price(S, K, T, r, sigma, ot)

    c1, c2, c3 = st.columns(3)
    c1.metric("RR Analytical", f"{analytic:.4f}")
    c2.metric(f"MC ({paths:,})", f"{mc:.4f}")
    c3.metric("Vanilla (ref)", f"{vanilla:.4f}")

    st.caption(
        "Payoff assumes the terminal-spot vs barrier convention: e.g. an "
        "up-and-out option is treated as knocked out whenever S_T is past "
        "H. This is a simplification for a static diagram — a real barrier "
        "option's knock status depends on the full path, not S_T alone."
    )
    col1, col2 = st.columns(2)
    with col1:
        s_grid = np.linspace(S * 0.5, S * 1.5, 200)
        vanilla_payoff = np.maximum((1 if ot == "call" else -1) * (s_grid - K), 0.0)
        if bt == "down-and-out":
            alive = s_grid > H
        elif bt == "down-and-in":
            alive = s_grid <= H
        elif bt == "up-and-out":
            alive = s_grid < H
        else:  # up-and-in
            alive = s_grid >= H
        exotic_payoff = np.where(alive, vanilla_payoff, 0.0)
        st.plotly_chart(
            exotic_payoff_figure(s_grid, exotic_payoff, vanilla_payoff, f"{bt} {ot}", S),
            use_container_width=True)
    with col2:
        exotic_d = barrier_delta_profile(s_grid, K, T, r, sigma, H, bt, ot)
        vanilla_d = np.array([_BS.greeks(s, K, T, r, sigma, ot)["delta"] for s in s_grid])
        st.plotly_chart(barrier_delta_figure(s_grid, exotic_d, vanilla_d, H, bt, S),
                        use_container_width=True)

    st.divider()
    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: _bar.price(s, k, t, r_, sig, ot, H, bt),
        S, K, T, r, sigma, ot, f"{bt} {ot}")

    tg, mc_paths, knocked = get_mc_paths_for_display(S, T, r, sigma, H, bt, n_display=60,
                                                      steps=steps, seed=seed)
    st.plotly_chart(barrier_mc_paths_figure(tg, mc_paths, H, bt, knocked),
                    use_container_width=True)


def _render_double_barrier(params: dict):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    ot = params["option_type"]
    HL = params.get("H_lower", 85.0)
    HU = params.get("H_upper", 120.0)
    paths = int(params.get("paths", 20_000))
    steps = int(params.get("steps", 126))
    seed = int(params.get("seed", 42))

    dko = _dbl.price(S, K, T, r, sigma, ot, HL, HU, "double-knock-out")
    dki = _dbl.price(S, K, T, r, sigma, ot, HL, HU, "double-knock-in")
    mc = double_barrier_mc_price(S, K, T, r, sigma, ot, HL, HU, paths=paths, steps=steps, seed=seed)
    vanilla = _BS.price(S, K, T, r, sigma, ot)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("DKO (tree)", f"{dko:.4f}")
    c2.metric("DKI (tree)", f"{dki:.4f}")
    c3.metric(f"MC DKO", f"{mc:.4f}")
    c4.metric("Vanilla", f"{vanilla:.4f}")

    st.caption(
        "Payoff assumes the terminal-spot vs corridor convention: knocked "
        "out whenever S_T falls outside [H_lower, H_upper]. A real double-"
        "barrier option's knock status depends on the full path, not S_T alone."
    )
    s_grid = np.linspace(HL * 0.8, HU * 1.2, 300)
    inside = (s_grid > HL) & (s_grid < HU)
    vanilla_payoff = np.maximum((1 if ot == "call" else -1) * (s_grid - K), 0.0)
    dko_payoff = np.where(inside, vanilla_payoff, 0.0)
    st.plotly_chart(exotic_payoff_figure(s_grid, dko_payoff, vanilla_payoff, "Double Knock-Out", S),
                    use_container_width=True)

    st.divider()
    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: _dbl.price(s, k, t, r_, sig, ot, HL, HU, "double-knock-out"),
        S, K, T, r, sigma, ot, "Double Knock-Out")


def _render_quanto(params: dict):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    ot = params["option_type"]
    r_f = params.get("r_f", 0.03)
    sig_FX = params.get("sigma_FX", 0.15)
    rho = params.get("rho", -0.3)
    Q0 = params.get("Q0", 1.0)

    price = _qto.price(S, K, T, r, sigma, ot, r_d=r, r_f=r_f,
                       sigma_S=sigma, sigma_FX=sig_FX, rho=rho, Q0=Q0)
    vanilla = _BS.price(S, K, T, r, sigma, ot)
    components = _qto.decompose(S, K, T, r, sigma, ot,
                                sigma_FX=sig_FX, rho=rho, Q0=Q0)

    c1, c2 = st.columns(2)
    c1.metric("Quanto Price", f"{price:.4f}")
    c2.metric("Vanilla (ref)", f"{vanilla:.4f}")

    st.markdown("**Decomposition**")
    st.dataframe(pd.DataFrame(components)[["component", "value"]].rename(
        columns={"component": "Component", "value": "Value ($)"}
    ), use_container_width=True)

    # Rho sensitivity
    rho_grid = np.linspace(-0.99, 0.99, 50)
    rho_prices = np.array([
        _qto.price(S, K, T, r, sigma, ot, r_d=r, r_f=r_f,
                   sigma_S=sigma, sigma_FX=sig_FX, rho=rh, Q0=Q0)
        for rh in rho_grid
    ])
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rho_grid, y=rho_prices, mode="lines",
                             line=dict(color="#1f77b4", width=2), name="Quanto price"))
    fig.add_hline(y=vanilla, line_dash="dot", line_color="gray",
                  annotation_text="Vanilla")
    fig.add_vline(x=rho, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title="ρ (stock–FX corr)", yaxis_title="Price",
                      height=360, margin=dict(t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: _qto.price(s, k, t, r_, sig, ot, r_d=r_,
                                            r_f=r_f, sigma_S=sig, sigma_FX=sig_FX, rho=rho, Q0=Q0),
        S, K, T, r, sigma, ot, "Quanto")


def _render_structured(params: dict):
    S, K, T, r, sigma = params["S"], params["K"], params["T"], params["r"], params["sigma"]
    product = params["product"]

    pricer_map = {
        "Reverse Convertible": lambda: ReverseConvertible().price(
            S, K, T, r, sigma,
            params.get("coupon_rate", 0.10), params.get("notional", 1000.0)),
        "Barrier Reverse Convertible": lambda: BarrierReverseConvertible().price(
            S, K, T, r, sigma,
            params.get("coupon_rate", 0.10), params.get("barrier_brc", 75.0),
            params.get("notional", 1000.0)),
        "Discount Certificate": lambda: DiscountCertificate().price(
            S, K, T, r, sigma, params.get("notional", 1000.0)),
        "Bonus Certificate": lambda: BonusCertificate().price(
            S, params.get("K_bonus", 110.0), T, r, sigma,
            params.get("barrier_bonus", 80.0), params.get("notional", 1000.0)),
        "Airbag Certificate": lambda: AirbagCertificate().price(
            S, K, params.get("K_floor", 90.0), T, r, sigma,
            params.get("participation", 1.0), params.get("notional", 1000.0)),
        "Twin-Win Certificate": lambda: TwinWinCertificate().price(
            S, K, T, r, sigma,
            params.get("barrier_tw", 70.0), params.get("notional", 1000.0)),
    }

    def pricer_map_at(s: float, t: float, r_: float, sig: float) -> dict:
        """Same pricer_map, but with (S,T,r,sigma) swapped for arbitrary
        values — used by the Premium/Greeks finite-difference sweep."""
        if product == "Reverse Convertible":
            return ReverseConvertible().price(
                s, K, t, r_, sig, params.get("coupon_rate", 0.10), params.get("notional", 1000.0))
        if product == "Barrier Reverse Convertible":
            return BarrierReverseConvertible().price(
                s, K, t, r_, sig, params.get("coupon_rate", 0.10),
                params.get("barrier_brc", 75.0), params.get("notional", 1000.0))
        if product == "Discount Certificate":
            return DiscountCertificate().price(s, K, t, r_, sig, params.get("notional", 1000.0))
        if product == "Bonus Certificate":
            return BonusCertificate().price(
                s, params.get("K_bonus", 110.0), t, r_, sig,
                params.get("barrier_bonus", 80.0), params.get("notional", 1000.0))
        if product == "Airbag Certificate":
            return AirbagCertificate().price(
                s, K, params.get("K_floor", 90.0), t, r_, sig,
                params.get("participation", 1.0), params.get("notional", 1000.0))
        return TwinWinCertificate().price(
            s, K, t, r_, sig, params.get("barrier_tw", 70.0), params.get("notional", 1000.0))

    decompose_map = {
        "Reverse Convertible": lambda: ReverseConvertible().decompose(
            S, K, T, r, sigma, params.get("coupon_rate", 0.10), params.get("notional", 1000.0)),
        "Barrier Reverse Convertible": lambda: BarrierReverseConvertible().decompose(
            S, K, T, r, sigma, params.get("coupon_rate", 0.10),
            params.get("barrier_brc", 75.0), params.get("notional", 1000.0)),
        "Discount Certificate": lambda: DiscountCertificate().decompose(
            S, K, T, r, sigma, params.get("notional", 1000.0)),
        "Bonus Certificate": lambda: BonusCertificate().decompose(
            S, params.get("K_bonus", 110.0), T, r, sigma,
            params.get("barrier_bonus", 80.0), params.get("notional", 1000.0)),
        "Airbag Certificate": lambda: AirbagCertificate().decompose(
            S, K, params.get("K_floor", 90.0), T, r, sigma,
            params.get("participation", 1.0), params.get("notional", 1000.0)),
        "Twin-Win Certificate": lambda: TwinWinCertificate().decompose(
            S, K, T, r, sigma, params.get("barrier_tw", 70.0), params.get("notional", 1000.0)),
    }

    result = pricer_map[product]()
    notional = params.get("notional", 1000.0)
    price_val = result.get("price", result.get("value", 0.0))

    c1, c2 = st.columns(2)
    c1.metric("Product Price", f"{price_val:.2f}")
    c2.metric("% of Notional", f"{price_val / notional * 100:.1f}%")
    if "coupon_rate" in result:
        st.metric("Coupon / Yield", f"{result['coupon_rate']:.2%}")

    components = decompose_map[product]()
    st.markdown("**Product Decomposition**")
    df = pd.DataFrame(components)
    if "pct" in df.columns:
        df["pct"] = df["pct"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(df.rename(columns={"component": "Component", "value": "Value ($)", "pct": "% Notional"}),
                 use_container_width=True)
    st.plotly_chart(decomposition_waterfall_figure(components), use_container_width=True)

    st.divider()
    st.markdown("**Payoff vs Spot at Maturity**")
    st.caption(
        "Barrier-dependent products (Barrier RC, Bonus, Twin-Win) use the "
        "terminal-spot vs barrier convention, same simplification as the "
        "Path-Dependent barrier products."
    )
    s_grid = np.linspace(S * 0.5, S * 1.5, 200)
    K_bonus = params.get("K_bonus", 110.0)
    barrier_bonus = params.get("barrier_bonus", 80.0)
    barrier_brc = params.get("barrier_brc", 75.0)
    barrier_tw = params.get("barrier_tw", 70.0)
    coupon_rate = params.get("coupon_rate", 0.10)
    K_floor = params.get("K_floor", 90.0)
    participation = params.get("participation", 1.0)

    def _payoff(s_t):
        n = notional / S
        if product == "Reverse Convertible":
            return np.where(s_t >= K, notional * (1 + coupon_rate), s_t / K * notional)
        if product == "Barrier Reverse Convertible":
            touched = s_t <= barrier_brc
            below = touched & (s_t < K)
            return np.where(below, s_t / K * notional, notional * (1 + coupon_rate))
        if product == "Discount Certificate":
            return n * np.minimum(s_t, K)
        if product == "Bonus Certificate":
            touched = s_t <= barrier_bonus
            return np.where(touched, n * s_t, n * np.maximum(s_t, K_bonus))
        if product == "Airbag Certificate":
            return n * (s_t + participation * np.maximum(K_floor - s_t, 0.0)
                        - np.maximum(s_t - K, 0.0))
        # Twin-Win Certificate
        touched = s_t <= barrier_tw
        return np.where(touched, n * s_t, n * (s_t + np.abs(s_t - K)))

    payoff = _payoff(s_grid)
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s_grid, y=payoff, mode="lines",
                             line=dict(color="#2ca02c", width=2.5), name=product))
    fig.add_hline(y=notional, line_dash="dot", line_color="gray", annotation_text="Notional")
    fig.add_vline(x=S, line_dash="dash", line_color="gray", annotation_text=f"S={S:.0f}")
    fig.update_layout(xaxis_title="Spot at Maturity", yaxis_title="Payoff ($)",
                      height=380, margin=dict(t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

    _render_premium_and_greeks(
        lambda s, k, t, r_, sig: pricer_map_at(s, t, r_, sig)["price"],
        S, K, T, r, sigma, "call", product, show_vanilla=False)


def render_path_dep(params: dict):
    product = params["product"]
    st.subheader(product)

    if product == "Digital — Cash-or-Nothing":
        _render_digital(params, "cash-or-nothing")
    elif product == "Digital — Asset-or-Nothing":
        _render_digital(params, "asset-or-nothing")
    elif product == "Asian — Geometric (CF)":
        _render_asian(params, "geometric")
    elif product == "Asian — Arithmetic (MC)":
        _render_asian(params, "arithmetic")
    elif product == "Knock-In / Knock-Out":
        _render_barrier(params)
    elif product == "Double Barrier":
        _render_double_barrier(params)
    elif product == "Quanto Call/Put":
        _render_quanto(params)
    elif product in STRUCTURED_PRODUCTS:
        _render_structured(params)

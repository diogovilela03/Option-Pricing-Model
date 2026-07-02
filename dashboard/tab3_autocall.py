"""Tab 3 — Tier C: Autocallable Structured Products."""
import pandas as pd
import streamlit as st

from pricing.autocall import AutocallIncremental, Phoenix, PhoenixMemory, default_observation_dates
from dashboard.charts import autocall_schedule_figure, decomposition_waterfall_figure

_AUTO = AutocallIncremental()
_PHX  = Phoenix()
_PMEM = PhoenixMemory()

AUTOCALL_PRODUCTS = ["Autocall Incremental", "Phoenix", "Phoenix Memory"]
OBSERVATION_FREQUENCIES = ["annual", "semi-annual", "quarterly", "monthly"]
_PERIODS_PER_YEAR = {"annual": 1, "semi-annual": 2, "quarterly": 4, "monthly": 12}


def render_autocall_sidebar() -> dict:
    product = st.sidebar.selectbox("Product", AUTOCALL_PRODUCTS)
    st.sidebar.markdown("---")
    S0 = st.sidebar.number_input("Spot S₀", value=100.0, min_value=1.0, step=1.0)
    T  = st.sidebar.number_input("Maturity T (years)", value=2.0, min_value=0.25, step=0.25)
    r  = st.sidebar.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.3f")
    sigma = st.sidebar.number_input("Volatility σ", value=0.20, step=0.01, format="%.2f")

    frequency = st.sidebar.selectbox("Observation Frequency", OBSERVATION_FREQUENCIES, index=2)

    dip_style = st.sidebar.radio(
        "DIP Style", ["European", "American"],
        help="European: protection barrier checked only at maturity. "
             "American: protection is lost the first time spot ever touches "
             "the barrier at any observation date.").lower()

    autocall_lvl = st.sidebar.number_input(
        "Autocallable Barrier (× S₀)", value=1.00, min_value=0.5, step=0.05, format="%.2f")
    prot_barrier = st.sidebar.number_input(
        "DIP / Protection Barrier (× S₀)", value=0.60, min_value=0.1, step=0.05, format="%.2f")

    extra = {}
    if product in ("Phoenix", "Phoenix Memory"):
        extra["coupon_barrier"] = st.sidebar.number_input(
            "Coupon Barrier (× S₀)", value=0.70, step=0.05, format="%.2f")

    coupon_annual = st.sidebar.number_input(
        "Annualized Coupon Rate", value=0.08, step=0.01, format="%.2f",
        help="Converted internally to a per-observation-period coupon based "
             "on the observation frequency.")
    extra["coupon_annual"] = coupon_annual
    extra["coupon"] = coupon_annual / _PERIODS_PER_YEAR[frequency]

    with st.sidebar.expander("MC Settings"):
        extra["paths"] = st.number_input("Paths", value=20_000, step=5000)
        extra["seed"]  = st.number_input("Seed", value=42, step=1)

    return dict(product=product, S0=S0, T=T, r=r, sigma=sigma, frequency=frequency,
                dip_style=dip_style, autocall_lvl=autocall_lvl,
                prot_barrier=prot_barrier, **extra)


def render_autocall(params: dict):
    product   = params["product"]
    S0        = params["S0"]
    T         = params["T"]
    r         = params["r"]
    sigma     = params["sigma"]
    frequency = params["frequency"]
    dip_style = params["dip_style"]
    autocall_lvl   = params["autocall_lvl"]
    coupon    = params["coupon"]
    coupon_annual = params["coupon_annual"]
    prot      = params["prot_barrier"]
    coupon_b  = params.get("coupon_barrier", 0.70)
    paths     = int(params.get("paths", 20_000))
    seed      = int(params.get("seed", 42))

    obs_dates = default_observation_dates(T, frequency)

    st.subheader(product)
    st.caption(f"DIP style: **{dip_style}** — {'protection checked only at maturity' if dip_style == 'european' else 'protection lost permanently the first time spot touches the barrier'}.")

    if product == "Autocall Incremental":
        result = _AUTO.price(S0, T, r, sigma, autocall_lvl, coupon,
                             obs_dates, prot, dip_style, 1000.0, paths, seed)
        components = _AUTO.decompose(S0, T, r, sigma, autocall_lvl, coupon,
                                     obs_dates, prot, dip_style, 1000.0, paths, seed)
    elif product == "Phoenix":
        result = _PHX.price(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                            coupon, obs_dates, dip_style, 1000.0, paths, seed)
        components = _PHX.decompose(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                                    coupon, obs_dates, dip_style, 1000.0, paths, seed)
    else:
        result = _PMEM.price(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                             coupon, obs_dates, dip_style, 1000.0, paths, seed)
        components = _PMEM.decompose(S0, T, r, sigma, autocall_lvl, coupon_b, prot,
                                     coupon, obs_dates, dip_style, 1000.0, paths, seed)

    c1, c2, c3 = st.columns(3)
    c1.metric("Price (per 1000 notional)", f"{result['price']:.2f}")
    c2.metric("% of Notional", f"{result['price'] / 1000 * 100:.1f}%")
    c3.metric("P(autocall)", f"{result['autocall_probability']:.1%}")
    st.caption(f"Annualized coupon {coupon_annual:.1%} → {coupon:.2%} per {frequency} observation "
              f"({len(obs_dates)} observations).")

    # Decomposition waterfall
    st.markdown("**Decomposition**")
    st.dataframe(pd.DataFrame(components)[["component", "value", "pct"]].rename(
        columns={"component": "Component", "value": "Value ($)", "pct": "% Notional"}
    ).assign(**{"% Notional": lambda df: df["% Notional"].apply(lambda x: f"{x:.1f}%")}),
        use_container_width=True)
    st.plotly_chart(decomposition_waterfall_figure(components), use_container_width=True)

    # Additional analytics
    st.markdown("**Additional Analytics**")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Equivalent Expected Maturity", f"{result['expected_exit_time']:.2f} yrs")
    a2.metric("Equivalent Zero-Coupon Bond", f"{result['equivalent_zcb']:.2f}")
    a3.metric("Forward at Maturity", f"{result['forward_at_maturity']:.2f}")
    a4.metric("P(Capital Loss)", f"{result['capital_loss_probability']:.2%}")
    st.caption("P(Capital Loss) = P(product survives to maturity, protection is not "
              "intact, and S_T < S₀) — excludes any coupons received along the way.")

    # Observation-date probability table
    st.markdown("**Observation Schedule Probabilities**")
    obs_df = pd.DataFrame(result["obs_table"]).rename(columns={
        "observation": "Observation", "maturity_probability": "Maturity Probability",
        "coupon_probability": "Coupon Probability",
    })
    obs_df["Maturity Probability"] = obs_df["Maturity Probability"].apply(lambda x: f"{x:.2%}")
    obs_df["Coupon Probability"] = obs_df["Coupon Probability"].apply(lambda x: f"{x:.2%}")
    st.dataframe(obs_df, use_container_width=True, hide_index=True)

    # Observation schedule fan chart
    obs_prices = result["obs_prices"]  # (n_obs, paths)
    n_show = min(obs_prices.shape[1], 80)
    sample_paths = obs_prices[:, :n_show]

    cb = coupon_b if product != "Autocall Incremental" else None
    st.plotly_chart(
        autocall_schedule_figure(obs_dates, sample_paths, S0,
                                 autocall_lvl, cb, prot),
        use_container_width=True)

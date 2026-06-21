"""Tab 2 — Market Data Snapshot: SPY chain, IV inversion, SVI calibration, arbitrage checks."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from storage.db import load_chain
from vol_surface.iv_inversion import implied_vol
from vol_surface.svi import fit_slice, svi_total_var
from vol_surface.arbitrage_checks import check_butterfly, check_calendar
from dashboard.charts import smile_figure

DB_PATH = Path("data/spy_chain.db")
K_DENSE = np.linspace(-0.8, 0.8, 200)


@st.cache_data(show_spinner=False)
def _load() -> pd.DataFrame:
    return load_chain(DB_PATH)


@st.cache_data(show_spinner=False)
def _compute_ivs(expiry: str, spot: float, r: float, df_json: str) -> pd.DataFrame:
    df = pd.read_json(df_json, orient="records")
    slice_df = df[df["expiration"] == expiry].copy()
    today = pd.Timestamp.utcnow().normalize()
    T = max((pd.Timestamp(expiry) - today).days / 365.0, 1 / 365)

    ivs = []
    for _, row in slice_df.iterrows():
        try:
            iv = implied_vol(spot, row["strike"], T, r, row["close_price"], row["option_type"])
            ivs.append(iv)
        except ValueError:
            ivs.append(float("nan"))

    slice_df["iv"] = ivs
    slice_df["T"] = T
    slice_df["log_moneyness"] = np.log(slice_df["strike"] / (spot * math.exp(r * T)))
    slice_df["total_var"] = slice_df["iv"] ** 2 * T
    return slice_df.dropna(subset=["iv"])


@st.cache_data(show_spinner=False)
def _fit_svi(df_json: str) -> dict:
    """Returns dict of {expiry: SVI params}."""
    df = pd.read_json(df_json, orient="records")
    results = {}
    for expiry, group in df.groupby("expiration"):
        clean = group.dropna(subset=["total_var"])
        if len(clean) < 5:
            continue
        try:
            params = fit_slice(clean["log_moneyness"].values, clean["total_var"].values)
            results[expiry] = params
        except Exception:
            pass
    return results


def _arbitrage_summary(svi_fits: dict) -> pd.DataFrame:
    rows = []
    for expiry, params in svi_fits.items():
        bf = check_butterfly(K_DENSE, params)
        rows.append({
            "Expiry": expiry,
            "Butterfly": "✅ Pass" if bf["arbitrage_free"] else f"❌ Fail",
            "Min g(k)": f"{bf['min_g']:.6f}",
            "BF Violations": bf["violation_count"],
        })
    results_df = pd.DataFrame(rows)

    slices = [{"T": i * 0.1, "params": p} for i, (_, p) in enumerate(svi_fits.items())]
    cal = check_calendar(K_DENSE, slices)
    cal_rows = []
    for pr in cal["pair_results"]:
        cal_rows.append({
            "T1 index": f"slice {int(pr['T1']/0.1)}",
            "T2 index": f"slice {int(pr['T2']/0.1)}",
            "Calendar": "✅ Pass" if pr["arbitrage_free"] else "❌ Fail",
            "Min ΔW": f"{pr['min_diff']:.6f}",
            "Violations": pr["violation_count"],
        })

    return results_df, pd.DataFrame(cal_rows)


def render_tab2():
    st.caption(
        "⚠️ Prices are single-day end-of-day **last trade prices** (not bid-ask mid). "
        "This is a disclosed methodology choice — yfinance free tier does not expose live quotes."
    )

    if not DB_PATH.exists():
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Run `python scripts/fetch_snapshot.py` first to build the snapshot."
        )
        return

    with st.spinner("Loading chain from SQLite..."):
        df = _load()

    fetched_at = df["fetched_at"].iloc[0][:10]
    st.info(f"Snapshot date: **{fetched_at}** · {len(df):,} contracts · {df['expiration'].nunique()} expiries")

    # ----------------------------------------------------------------
    # Config
    # ----------------------------------------------------------------
    cc1, cc2 = st.columns(2)
    spot = cc1.number_input("Spot price (S)", value=746.74, step=0.5,
                             help="SPY spot at time of snapshot. Used for log-moneyness and IV inversion.")
    r = cc2.number_input("Risk-free rate (r)", value=0.05, step=0.005,
                          help="Used to compute the forward and discount factor.")

    st.divider()

    # ----------------------------------------------------------------
    # Raw chain table
    # ----------------------------------------------------------------
    with st.expander("Raw Option Chain", expanded=False):
        st.dataframe(df.drop(columns=["fetched_at"]), use_container_width=True, hide_index=True)

    st.divider()

    # ----------------------------------------------------------------
    # Expiry selector + IV smile
    # ----------------------------------------------------------------
    st.subheader("Implied Volatility Smile")

    expiries = sorted(df["expiration"].unique())
    selected_expiry = st.selectbox("Select expiry", expiries)

    with st.spinner("Inverting implied vols..."):
        slice_df = _compute_ivs(selected_expiry, spot, r, df.to_json(orient="records"))

    if slice_df.empty:
        st.warning("No contracts with valid IVs for this expiry.")
        return

    calls = slice_df[slice_df["option_type"] == "call"]
    puts = slice_df[slice_df["option_type"] == "put"]

    with st.spinner("Fitting SVI..."):
        try:
            svi_params = fit_slice(slice_df["log_moneyness"].values, slice_df["total_var"].values)
            T_expiry = float(slice_df["T"].iloc[0])
            w_dense = svi_total_var(K_DENSE, svi_params)
            svi_iv = np.sqrt(np.maximum(w_dense, 0) / T_expiry)
            F = spot * math.exp(r * T_expiry)
            svi_strikes = F * np.exp(K_DENSE)
            svi_fit_ok = True
        except Exception as e:
            st.warning(f"SVI fit failed for {selected_expiry}: {e}")
            svi_fit_ok = False

    fig = smile_figure(
        strikes=slice_df["strike"].values,
        iv_calls=calls["iv"].values if not calls.empty else None,
        iv_puts=puts["iv"].values if not puts.empty else None,
        call_strikes=calls["strike"].values if not calls.empty else None,
        put_strikes=puts["strike"].values if not puts.empty else None,
        svi_strikes=svi_strikes if svi_fit_ok else np.array([]),
        svi_iv=svi_iv if svi_fit_ok else np.array([]),
        expiry=selected_expiry,
        spot=spot,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # ----------------------------------------------------------------
    # Arbitrage checks across all expiries
    # ----------------------------------------------------------------
    st.subheader("Arbitrage Checks")

    with st.spinner("Fitting SVI across all expiries and running checks..."):
        all_iv_data = {}
        for expiry in expiries:
            exp_df = _compute_ivs(expiry, spot, r, df.to_json(orient="records"))
            if len(exp_df) >= 5:
                all_iv_data[expiry] = exp_df

        svi_fits = _fit_svi(
            pd.concat(all_iv_data.values()).to_json(orient="records")
            if all_iv_data else "{}"
        )

    if not svi_fits:
        st.warning("Could not fit SVI for any expiry.")
        return

    bf_df, cal_df = _arbitrage_summary(svi_fits)

    st.markdown("**Butterfly arbitrage (within each slice)** — requires g(k) ≥ 0 everywhere")
    st.dataframe(bf_df, use_container_width=True, hide_index=True)

    st.markdown("**Calendar spread arbitrage (across consecutive slices)** — requires ΔW ≥ 0 for all k")
    if cal_df.empty:
        st.info("Only one slice fitted — no calendar pairs to check.")
    else:
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

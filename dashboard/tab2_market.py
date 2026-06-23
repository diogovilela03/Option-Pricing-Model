"""Tab 2 — Market Data Snapshot: IV inversion, SVI / SSVI / Heston comparison, arbitrage checks."""
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from storage.db import load_chain, load_calibration_cache
from vol_surface.iv_inversion import implied_vol
from vol_surface.svi import fit_slice, svi_total_var
from vol_surface.ssvi import fit_ssvi, ssvi_total_var
from vol_surface.heston_calibration import calibrate_heston
from vol_surface.arbitrage_checks import (
    check_butterfly, check_calendar,
    check_butterfly_w, check_calendar_w,
)
from pricing.black_scholes import BlackScholes
from pricing.heston_cf import heston_price
from dashboard.charts import (
    smile_figure, vol_surface_3d,
    greek_market_profile_figure,
)

_BS         = BlackScholes()
DB_PATH     = Path("data/spy_chain.db")
CACHE_PATH  = Path("data/calibration_cache.json")
K_DENSE     = np.linspace(-0.8, 0.8, 200)

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


# ──────────────────────────────────────────────
# Cached data / computation helpers
# ──────────────────────────────────────────────

@st.cache_resource
def _load() -> pd.DataFrame:
    return load_chain(DB_PATH)


@st.cache_resource
def _load_calibration_cache() -> dict | None:
    return load_calibration_cache(CACHE_PATH)


def _cache_hit(spot: float, r: float) -> bool:
    cache = _load_calibration_cache()
    return (
        cache is not None
        and abs(cache["spot"] - spot) < 0.01
        and abs(cache["r"] - r) < 1e-4
    )


@st.cache_resource
def _compute_ivs(expiry: str, spot: float, r: float) -> pd.DataFrame:
    """IV inversion for one expiry slice, with outlier filtering."""
    if _cache_hit(spot, r):
        cache = _load_calibration_cache()
        ivs_data = cache.get("ivs", {}).get(expiry)  # type: ignore[union-attr]
        if ivs_data:
            return pd.DataFrame(ivs_data)

    df = _load()
    slice_df = df[df["expiration"] == expiry].copy()
    T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)

    ivs = []
    for _, row in slice_df.iterrows():
        try:
            iv = implied_vol(spot, row["strike"], T, r, row["close_price"], row["option_type"])
            ivs.append(iv)
        except ValueError:
            ivs.append(float("nan"))

    slice_df["iv"] = ivs
    slice_df["T"]  = T
    slice_df["log_moneyness"] = np.log(slice_df["strike"] / (spot * math.exp(r * T)))
    slice_df["total_var"]     = slice_df["iv"] ** 2 * T

    # Filter stale / illiquid prices: drop extreme IVs
    slice_df = slice_df.dropna(subset=["iv"])
    slice_df = slice_df[slice_df["iv"].between(0.01, 1.5)]
    return slice_df


@st.cache_resource
def _compute_market_greeks(expiry: str, spot: float, r: float) -> pd.DataFrame:
    slice_df = _compute_ivs(expiry, spot, r)
    rows = []
    for _, row in slice_df.iterrows():
        try:
            g = _BS.greeks(spot, row["strike"], row["T"], r, row["iv"], row["option_type"])
            rows.append({**row.to_dict(), **g})
        except Exception:
            pass
    return pd.DataFrame(rows)


@st.cache_resource
def _fit_svi_all(spot: float, r: float) -> dict:
    """Fit raw SVI per expiry slice. Returns {expiry: params}."""
    if _cache_hit(spot, r):
        cache = _load_calibration_cache()
        svi_fits = cache.get("svi_fits")  # type: ignore[union-attr]
        if svi_fits:
            return svi_fits

    df      = _load()
    results = {}
    for expiry in sorted(df["expiration"].unique()):
        exp_df = _compute_ivs(expiry, spot, r)
        if len(exp_df) < 5:
            continue
        try:
            params = fit_slice(exp_df["log_moneyness"].values, exp_df["total_var"].values)
            results[expiry] = params
        except Exception:
            pass
    return results


@st.cache_resource
def _fit_ssvi_surface(spot: float, r: float) -> dict | None:
    """Calibrate SSVI globally across all expiries.

    Returns dict with keys: rho, eta, gamma, rmse, thetas {expiry: theta}.
    """
    if _cache_hit(spot, r):
        cache = _load_calibration_cache()
        ssvi = cache.get("ssvi")  # type: ignore[union-attr]
        if ssvi:
            return ssvi

    svi_fits = _fit_svi_all(spot, r)
    if len(svi_fits) < 2:
        return None

    slices = []
    thetas = {}
    for expiry, params in svi_fits.items():
        exp_df = _compute_ivs(expiry, spot, r)
        if len(exp_df) < 5:
            continue
        theta = float(svi_total_var(np.array([0.0]), params)[0])  # ATM total var
        thetas[expiry] = theta
        slices.append({
            "log_moneyness": exp_df["log_moneyness"].values,
            "total_var":     exp_df["total_var"].values,
            "theta":         theta,
        })

    if len(slices) < 2:
        return None

    try:
        result = fit_ssvi(slices)
        result["thetas"] = thetas
        return result
    except Exception:
        return None


@st.cache_resource
def _calibrate_heston_cached(spot: float, r: float) -> dict | None:
    """Heston calibration across the full chain (cached per session)."""
    if _cache_hit(spot, r):
        cache = _load_calibration_cache()
        heston = cache.get("heston")  # type: ignore[union-attr]
        if heston:
            return heston

    df = _load()
    all_rows = []
    for expiry in sorted(df["expiration"].unique()):
        exp_df = _compute_ivs(expiry, spot, r)
        if not exp_df.empty:
            all_rows.append(exp_df)
    if not all_rows:
        return None
    full_df = pd.concat(all_rows, ignore_index=True)
    try:
        return calibrate_heston(full_df, S=spot, r=r)
    except Exception:
        return None


def _ssvi_smile_curve(expiry: str, spot: float, r: float, ssvi: dict,
                      k_lo: float, k_hi: float):
    """Evaluate SSVI on a dense strike grid bounded by market strikes."""
    theta = ssvi["thetas"].get(expiry)
    if theta is None:
        return np.array([]), np.array([])
    T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)
    F = spot * math.exp(r * T)
    k_plot = np.linspace(k_lo, k_hi, 200)
    w = ssvi_total_var(k_plot, theta, ssvi["rho"], ssvi["eta"], ssvi["gamma"])
    iv = np.sqrt(np.maximum(w, 0) / T)
    return F * np.exp(k_plot), iv


def _heston_smile_curve(expiry: str, spot: float, r: float, heston: dict,
                        k_lo: float, k_hi: float):
    """Evaluate Heston IV on a coarse strike grid bounded by market strikes."""
    T = max((date.fromisoformat(expiry) - date.today()).days / 365.0, 1 / 365)
    F = spot * math.exp(r * T)
    k_coarse = np.linspace(k_lo, k_hi, 30)
    strikes  = F * np.exp(k_coarse)
    ivs = []
    for K in strikes:
        try:
            price = heston_price(
                spot, K, T, r,
                heston["v0"], heston["kappa"], heston["theta"],
                heston["xi"], heston["rho"],
                option_type="call",
            )
            iv = implied_vol(spot, K, T, r, price, "call")
            ivs.append(iv)
        except Exception:
            ivs.append(float("nan"))
    ivs = np.array(ivs)
    mask = ~np.isnan(ivs)
    return strikes[mask], ivs[mask]


# ──────────────────────────────────────────────
# Arbitrage summary helpers
# ──────────────────────────────────────────────

def _arbitrage_summary(svi_fits: dict, ssvi_result: dict | None, spot: float, r: float):
    """Butterfly and calendar checks for SVI, SSVI, and Heston.

    SVI  — checked analytically per slice (may fail; expected).
    SSVI — checked numerically; butterfly guaranteed free by calibration constraint.
    Heston — guaranteed free by model construction; noted without recomputation.
    """
    from datetime import date as _date

    # ── Butterfly ──────────────────────────────────────────────────────
    bf_rows = []
    for expiry, params in svi_fits.items():
        T = max((_date.fromisoformat(expiry) - _date.today()).days / 365.0, 1 / 365)
        F = spot * math.exp(r * T)

        # SVI (analytical derivatives)
        bf_svi = check_butterfly(K_DENSE, params)

        # SSVI (numerical derivatives on dense grid)
        if ssvi_result and expiry in ssvi_result.get("thetas", {}):
            theta = ssvi_result["thetas"][expiry]
            k_dense = np.linspace(K_DENSE[0], K_DENSE[-1], 400)
            w_ssvi = ssvi_total_var(
                k_dense, theta,
                ssvi_result["rho"], ssvi_result["eta"], ssvi_result["gamma"],
            )
            bf_ssvi = check_butterfly_w(k_dense, w_ssvi)
            ssvi_bf = "✅ Pass" if bf_ssvi["arbitrage_free"] else "❌ Fail"
            ssvi_g  = f"{bf_ssvi['min_g']:.6f}"
        else:
            ssvi_bf, ssvi_g = "—", "—"

        bf_rows.append({
            "Expiry":          expiry,
            "SVI BF":          "✅ Pass" if bf_svi["arbitrage_free"] else "❌ Fail",
            "SVI min g(k)":    f"{bf_svi['min_g']:.6f}",
            "SSVI BF":         ssvi_bf,
            "SSVI min g(k)":   ssvi_g,
            "Heston BF":       "✅ Pass*",
        })

    # ── Calendar ───────────────────────────────────────────────────────
    expiries_sorted = sorted(svi_fits.keys())
    T_map = {
        e: max((_date.fromisoformat(e) - _date.today()).days / 365.0, 1 / 365)
        for e in expiries_sorted
    }

    # SVI calendar
    svi_slices = [{"T": T_map[e], "params": svi_fits[e]} for e in expiries_sorted]
    cal_svi = check_calendar(K_DENSE, svi_slices)

    # SSVI calendar
    ssvi_slices_w = []
    if ssvi_result:
        for e in expiries_sorted:
            theta = ssvi_result.get("thetas", {}).get(e)
            if theta is not None:
                w = ssvi_total_var(
                    K_DENSE, theta,
                    ssvi_result["rho"], ssvi_result["eta"], ssvi_result["gamma"],
                )
                ssvi_slices_w.append({"T": T_map[e], "w": w})
    cal_ssvi = check_calendar_w(K_DENSE, ssvi_slices_w) if len(ssvi_slices_w) >= 2 else None

    cal_rows = []
    for pr_svi in cal_svi["pair_results"]:
        t1, t2 = pr_svi["T1"], pr_svi["T2"]

        # Match the corresponding SSVI pair
        ssvi_pr = None
        if cal_ssvi:
            for pr in cal_ssvi["pair_results"]:
                if abs(pr["T1"] - t1) < 0.01 and abs(pr["T2"] - t2) < 0.01:
                    ssvi_pr = pr
                    break

        e1 = min(expiries_sorted, key=lambda e: abs(T_map[e] - t1))
        e2 = min(expiries_sorted, key=lambda e: abs(T_map[e] - t2))
        cal_rows.append({
            "T1 expiry":    e1,
            "T2 expiry":    e2,
            "SVI Cal":      "✅ Pass" if pr_svi["arbitrage_free"] else "❌ Fail",
            "SVI min ΔW":   f"{pr_svi['min_diff']:.6f}",
            "SSVI Cal":     ("✅ Pass" if ssvi_pr["arbitrage_free"] else "❌ Fail") if ssvi_pr else "—",
            "SSVI min ΔW":  f"{ssvi_pr['min_diff']:.6f}" if ssvi_pr else "—",
            "Heston Cal":   "✅ Pass*",
        })

    return pd.DataFrame(bf_rows), pd.DataFrame(cal_rows)


# ──────────────────────────────────────────────
# Sidebar + main render
# ──────────────────────────────────────────────

def render_tab2_sidebar(expiries: list[str]) -> dict:
    cache = _load_calibration_cache()
    default_spot = float(cache["spot"]) if cache else 746.74
    default_r    = float(cache.get("r", 0.05)) if cache else 0.05

    st.sidebar.subheader("Market Snapshot")
    spot = st.sidebar.number_input(
        "Spot price (S)", value=default_spot, step=0.5, key="t2_spot",
        help="SPY spot at time of snapshot.",
    )
    r = st.sidebar.number_input(
        "Risk-free rate (r)", value=default_r, step=0.005, key="t2_r",
        help="Used to compute the forward and discount factor.",
    )
    selected_expiry = st.sidebar.selectbox("Expiry", expiries, key="t2_expiry")
    return dict(spot=spot, r=r, selected_expiry=selected_expiry)


def render_tab2(params: dict):
    st.caption(
        "⚠️ Prices are single-day end-of-day **last trade prices** (not bid-ask mid). "
        "IV outliers (σ < 1% or σ > 150%) are filtered as stale / illiquid."
    )

    if not DB_PATH.exists():
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Run `python scripts/fetch_snapshot.py` first."
        )
        return

    df = _load()
    fetched_at = df["fetched_at"].iloc[0][:10]
    st.info(
        f"Snapshot date: **{fetched_at}** · {len(df):,} contracts · "
        f"{df['expiration'].nunique()} expiries"
    )

    spot, r = params["spot"], params["r"]
    selected_expiry = params["selected_expiry"]

    with st.expander("Raw Option Chain", expanded=False):
        st.dataframe(df.drop(columns=["fetched_at"]), width='stretch', hide_index=True)

    st.divider()

    # ── IV smile ──────────────────────────────────────────────────────
    st.subheader(f"IV Smile — {selected_expiry}")
    st.caption("SVI (black · solid), SSVI (blue · dashed), Heston (orange · dot).")

    with st.spinner("Inverting IVs..."):
        slice_df = _compute_ivs(selected_expiry, spot, r)

    if slice_df.empty:
        st.warning("No contracts with valid IVs for this expiry.")
        return

    calls = slice_df[slice_df["option_type"] == "call"]
    puts  = slice_df[slice_df["option_type"] == "put"]
    T_exp = float(slice_df["T"].iloc[0])

    # Strike bounds from filtered market data — models only plotted within this range
    k_lo = float(slice_df["log_moneyness"].min())
    k_hi = float(slice_df["log_moneyness"].max())
    k_plot = np.linspace(k_lo, k_hi, 200)
    F = spot * math.exp(r * T_exp)

    # SVI curve
    svi_params = None
    svi_strikes, svi_iv = np.array([]), np.array([])
    with st.spinner("Fitting SVI..."):
        try:
            svi_params  = fit_slice(slice_df["log_moneyness"].values, slice_df["total_var"].values)
            w_dense     = svi_total_var(k_plot, svi_params)
            svi_iv      = np.sqrt(np.maximum(w_dense, 0) / T_exp)
            svi_strikes = F * np.exp(k_plot)
        except Exception as e:
            st.warning(f"SVI failed: {e}")

    # SSVI curve
    ssvi_strikes, ssvi_iv = np.array([]), np.array([])
    with st.spinner("Fitting SSVI surface..."):
        ssvi_result = _fit_ssvi_surface(spot, r)
        if ssvi_result:
            ssvi_strikes, ssvi_iv = _ssvi_smile_curve(
                selected_expiry, spot, r, ssvi_result, k_lo, k_hi
            )

    # Heston curve
    heston_strikes, heston_iv_curve = np.array([]), np.array([])
    with st.spinner("Loading Heston parameters..."):
        heston_result = _calibrate_heston_cached(spot, r)
        if heston_result:
            heston_strikes, heston_iv_curve = _heston_smile_curve(
                selected_expiry, spot, r, heston_result, k_lo, k_hi
            )

    model_curves = [
        {"name": "SVI",    "strikes": svi_strikes,    "iv": svi_iv,
         "color": "black",       "dash": "solid"},
        {"name": "SSVI",   "strikes": ssvi_strikes,   "iv": ssvi_iv,
         "color": "royalblue",   "dash": "dash"},
        {"name": "Heston", "strikes": heston_strikes, "iv": heston_iv_curve,
         "color": "darkorange",  "dash": "dot"},
    ]

    fig = smile_figure(
        call_strikes=calls["strike"].values if not calls.empty else None,
        iv_calls=calls["iv"].values         if not calls.empty else None,
        put_strikes=puts["strike"].values   if not puts.empty  else None,
        iv_puts=puts["iv"].values           if not puts.empty  else None,
        model_curves=model_curves,
        expiry=selected_expiry,
        spot=spot,
    )
    st.plotly_chart(fig, width='stretch')

    # ── Model parameter tables ─────────────────────────────────────────
    col_svi, col_ssvi, col_heston = st.columns(3)

    with col_svi:
        st.markdown("**SVI params** (per slice)")
        if svi_params:
            svi_df = pd.DataFrame([{
                "a": f"{svi_params['a']:.4f}",
                "b": f"{svi_params['b']:.4f}",
                "ρ": f"{svi_params['rho']:.4f}",
                "m": f"{svi_params['m']:.4f}",
                "σ": f"{svi_params['sigma']:.4f}",
            }]).T.rename(columns={0: "value"})
            st.dataframe(svi_df)

    with col_ssvi:
        st.markdown("**SSVI params** (global surface)")
        if ssvi_result:
            ssvi_df = pd.DataFrame([{
                "ρ":    f"{ssvi_result['rho']:.4f}",
                "η":    f"{ssvi_result['eta']:.4f}",
                "γ":    f"{ssvi_result['gamma']:.4f}",
                "RMSE": f"{ssvi_result['rmse']:.6f}",
            }]).T.rename(columns={0: "value"})
            st.dataframe(ssvi_df)
        else:
            st.info("SSVI fit unavailable.")

    with col_heston:
        st.markdown("**Heston params** (global calibration)")
        if heston_result:
            feller_ok = heston_result["feller_satisfied"]
            heston_df = pd.DataFrame([{
                "v₀":   f"{heston_result['v0']:.4f}",
                "κ":    f"{heston_result['kappa']:.4f}",
                "θ":    f"{heston_result['theta']:.4f}",
                "ξ":    f"{heston_result['xi']:.4f}",
                "ρ":    f"{heston_result['rho']:.4f}",
                "RMSE": f"{heston_result['rmse']:.6f}",
                "Feller 2κθ−ξ²": f"{'✅' if feller_ok else '❌'} {heston_result['feller_lhs']:.4f}",
            }]).T.rename(columns={0: "value"})
            st.dataframe(heston_df)
        else:
            st.info("Heston calibration unavailable.")

    st.divider()

    # ── 3D Vol Surface ─────────────────────────────────────────────────
    st.subheader("3D Volatility Surface (SVI)")
    with st.spinner("Building 3D surface..."):
        svi_fits_3d   = _fit_svi_all(spot, r)
        all_iv_frames = []
        for exp in svi_fits_3d:
            try:
                all_iv_frames.append(_compute_ivs(exp, spot, r))
            except Exception:
                pass

    if len(svi_fits_3d) >= 2 and all_iv_frames:
        expiry_T_map = {
            exp: max((date.fromisoformat(exp) - date.today()).days / 365.0, 1 / 365)
            for exp in svi_fits_3d
        }
        market_iv_df = pd.concat(all_iv_frames, ignore_index=True)[
            ["strike", "expiration", "option_type", "iv", "T"]
        ].dropna(subset=["iv"])
        fig_3d = vol_surface_3d(svi_fits_3d, expiry_T_map, market_iv_df, spot, r)
        st.plotly_chart(fig_3d, width='stretch')
    else:
        st.info("Need at least 2 fitted expiry slices to render the 3D surface.")

    st.divider()

    # ── Market Greeks ──────────────────────────────────────────────────
    st.subheader(f"Market Greeks — {selected_expiry}")
    st.caption("Greeks at each contract's own market-implied vol. Dashed line = spot.")

    with st.spinner("Computing market Greeks..."):
        greeks_slice = _compute_market_greeks(selected_expiry, spot, r)

    if not greeks_slice.empty:
        profile_greek = st.selectbox(
            "Greek", list(_GREEK_LABELS.keys()),
            format_func=lambda k: _GREEK_LABELS[k],
            key="t2_profile_greek",
        )
        calls_g = greeks_slice[greeks_slice["option_type"] == "call"]
        puts_g  = greeks_slice[greeks_slice["option_type"] == "put"]
        fig_profile = greek_market_profile_figure(
            calls_g, puts_g,
            greek=profile_greek,
            greek_label=_GREEK_LABELS[profile_greek],
            spot=spot,
        )
        st.plotly_chart(fig_profile, width='stretch')
    else:
        st.warning("No Greeks available for this expiry.")

    st.divider()

    # ── Arbitrage Checks ───────────────────────────────────────────────
    st.subheader("Arbitrage Checks — SVI vs SSVI vs Heston")
    st.caption(
        "**Butterfly** (within each slice): g(k) ≥ 0 everywhere — Gatheral-Jacquier density condition. "
        "**Calendar** (across slices): ΔW(k) ≥ 0 — Roper no-calendar-arb condition. "
        "\\* Heston passes both by model construction (single consistent SDE); shown as validation."
    )

    with st.spinner("Running arbitrage checks across all models..."):
        svi_fits    = _fit_svi_all(spot, r)
        ssvi_checks = _fit_ssvi_surface(spot, r)

    if not svi_fits:
        st.warning("No SVI fits available.")
        return

    bf_df, cal_df = _arbitrage_summary(svi_fits, ssvi_checks, spot, r)

    st.markdown("**Butterfly** — within each expiry slice")
    st.dataframe(bf_df, width='stretch', hide_index=True)

    st.markdown("**Calendar spread** — across consecutive expiries")
    if cal_df.empty:
        st.info("Only one slice fitted — no calendar pairs to check.")
    else:
        st.dataframe(cal_df, width='stretch', hide_index=True)

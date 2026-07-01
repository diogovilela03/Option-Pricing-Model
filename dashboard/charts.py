"""Shared Plotly chart helpers."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def heatmap_figure(
    s_grid: np.ndarray,
    sigma_grid: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    s_current: float,
    sigma_current: float,
) -> go.Figure:
    """Two side-by-side heatmaps (call, put) with current (S, σ) highlighted."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Call Price (BS)", "Put Price (BS)"],
        horizontal_spacing=0.12,
    )

    common = dict(
        x=np.round(s_grid, 2),
        y=np.round(sigma_grid * 100, 1),
        colorscale="viridis",
        showscale=True,
        texttemplate="%{z:.2f}",
        textfont=dict(size=11, color="white"),
    )

    fig.add_trace(
        go.Heatmap(**common, z=call_prices, name="Call", colorbar=dict(x=0.46, len=0.9)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Heatmap(**common, z=put_prices, name="Put", colorbar=dict(x=1.01, len=0.9)),
        row=1, col=2,
    )

    marker = dict(
        x=[s_current], y=[round(sigma_current * 100, 1)],
        mode="markers",
        marker=dict(symbol="star", size=14, color="white", line=dict(color="black", width=1)),
        showlegend=False,
        name="Current (S, σ)",
    )
    fig.add_trace(go.Scatter(**marker), row=1, col=1)
    fig.add_trace(go.Scatter(**marker), row=1, col=2)

    fig.update_xaxes(title_text="Spot (S)")
    fig.update_yaxes(title_text="Volatility σ (%)")
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    return fig


def greek_sensitivity_figure(
    xs: np.ndarray,
    series: list[dict],
    current_x: float,
    xlabel: str,
    greek: str,
    greek_label: str,
) -> go.Figure:
    """Two side-by-side line charts (Call | Put), each with one line per strike series.

    series: list of dicts with keys 'label', 'color',
            'call_greeks' {greek_name: [float]}, 'put_greeks' {greek_name: [float]}
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Call — {greek_label}", f"Put — {greek_label}"],
        horizontal_spacing=0.12,
    )

    for i, s in enumerate(series):
        show_legend = True
        fig.add_trace(go.Scatter(
            x=xs, y=s["call_greeks"][greek],
            mode="lines", line=dict(color=s["color"], width=2),
            name=s["label"], legendgroup=s["label"], showlegend=show_legend,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=xs, y=s["put_greeks"][greek],
            mode="lines", line=dict(color=s["color"], width=2),
            name=s["label"], legendgroup=s["label"], showlegend=False,
        ), row=1, col=2)

    for col in (1, 2):
        fig.add_vline(
            x=current_x,
            line_dash="dot", line_color="gray", line_width=1,
            row=1, col=col,
        )

    fig.update_xaxes(title_text=xlabel)
    fig.update_yaxes(title_text=greek_label, col=1)
    fig.update_layout(
        height=420,
        margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center"),
    )
    return fig


def vol_surface_3d(
    svi_fits: dict,
    expiry_T: dict,
    market_df,
    spot: float,
    r: float,
) -> go.Figure:
    """3D vol surface: smooth SVI mesh + raw market IV scatter dots."""
    import math
    import pandas as pd

    sorted_expiries = sorted(svi_fits.keys(), key=lambda e: expiry_T[e])
    T_vals = [expiry_T[e] for e in sorted_expiries]

    # global strike range across all expiries
    k_min = market_df["strike"].min()
    k_max = market_df["strike"].max()
    k_grid = np.linspace(k_min, k_max, 60)

    z_surf = np.full((len(sorted_expiries), len(k_grid)), np.nan)
    for i, expiry in enumerate(sorted_expiries):
        T = expiry_T[expiry]
        F = spot * math.exp(r * T)
        log_m = np.log(k_grid / F)
        from vol_surface.svi import svi_total_var
        w = svi_total_var(log_m, svi_fits[expiry])
        iv = np.sqrt(np.maximum(w, 0) / T)
        z_surf[i, :] = iv * 100

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=k_grid,
        y=T_vals,
        z=z_surf,
        colorscale="viridis",
        opacity=0.85,
        colorbar=dict(title="IV (%)", len=0.6, thickness=15),
        name="SVI surface",
        showlegend=False,
    ))

    # raw market IV dots
    call_df = market_df[market_df["option_type"] == "call"]
    put_df  = market_df[market_df["option_type"] == "put"]

    for df, color, symbol, label in [
        (call_df, "steelblue", "triangle-up", "Market IV (call)"),
        (put_df,  "tomato",    "circle",      "Market IV (put)"),
    ]:
        if df.empty:
            continue
        fig.add_trace(go.Scatter3d(
            x=df["strike"].values,
            y=df["T"].values,
            z=df["iv"].values * 100,
            mode="markers",
            marker=dict(size=3, color=color, symbol="circle"),
            name=label,
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Strike",
            yaxis_title="Time to Maturity (years)",
            zaxis_title="Implied Vol (%)",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
        ),
        height=550,
        margin=dict(t=30, b=10),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
    )
    return fig


def greek_market_profile_figure(
    calls_df,
    puts_df,
    greek: str,
    greek_label: str,
    spot: float,
) -> go.Figure:
    """Call | Put Greek profiles vs strike using market-implied vols."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Call — {greek_label}", f"Put — {greek_label}"],
        horizontal_spacing=0.12,
    )

    if not calls_df.empty:
        c = calls_df.sort_values("strike")
        fig.add_trace(go.Scatter(
            x=c["strike"], y=c[greek],
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=5),
            name="Call", showlegend=False,
        ), row=1, col=1)

    if not puts_df.empty:
        p = puts_df.sort_values("strike")
        fig.add_trace(go.Scatter(
            x=p["strike"], y=p[greek],
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=5),
            name="Put", showlegend=False,
        ), row=1, col=2)

    for col in (1, 2):
        fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                      line_width=1, row=1, col=col)

    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text=greek_label, col=1)
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    return fig


def greek_market_heatmap_figure(
    all_greeks_df,
    greek: str,
    greek_label: str,
) -> go.Figure:
    """Side-by-side heatmaps: Call Greek | Put Greek across expiries × strikes."""
    import pandas as pd

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"Call — {greek_label}", f"Put — {greek_label}"],
        horizontal_spacing=0.14,
    )

    for col, otype in [(1, "call"), (2, "put")]:
        sub = all_greeks_df[all_greeks_df["option_type"] == otype]
        if sub.empty:
            continue
        pivot = (
            sub.pivot_table(index="expiration", columns="strike", values=greek, aggfunc="mean")
            .sort_index()
        )
        fig.add_trace(go.Heatmap(
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            z=pivot.values,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(x=0.46 if col == 1 else 1.01, len=0.9, thickness=12),
            showscale=True,
        ), row=1, col=col)

    fig.update_xaxes(title_text="Strike")
    fig.update_yaxes(title_text="Expiry", col=1)
    fig.update_layout(height=420, margin=dict(t=40, b=20))
    return fig


def exotic_payoff_figure(
    S_grid: np.ndarray,
    exotic_payoffs: np.ndarray,
    vanilla_payoffs: np.ndarray,
    exotic_label: str,
    S_current: float,
) -> go.Figure:
    """Payoff-at-expiry: exotic vs vanilla benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid, y=vanilla_payoffs, mode="lines",
        line=dict(color="gray", width=1.5, dash="dot"), name="Vanilla",
    ))
    fig.add_trace(go.Scatter(
        x=S_grid, y=exotic_payoffs, mode="lines",
        line=dict(color="#1f77b4", width=2.5), name=exotic_label,
    ))
    fig.add_vline(x=S_current, line_dash="dash", line_color="gray", line_width=1,
                  annotation_text=f"S={S_current:.0f}")
    fig.update_layout(
        xaxis_title="Spot at Expiry", yaxis_title="Payoff",
        height=380, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def barrier_delta_figure(
    S_grid: np.ndarray,
    exotic_delta: np.ndarray,
    vanilla_delta: np.ndarray,
    barrier: float,
    barrier_type: str,
    S_current: float,
) -> go.Figure:
    """Delta profile near barrier (shows discontinuity / explosion)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=S_grid, y=vanilla_delta, mode="lines",
        line=dict(color="gray", width=1.5, dash="dot"), name="Vanilla delta",
    ))
    fig.add_trace(go.Scatter(
        x=S_grid, y=exotic_delta, mode="lines",
        line=dict(color="#d62728", width=2.5), name="Barrier delta",
    ))
    fig.add_vline(x=barrier, line_dash="dash", line_color="orange", line_width=1.5,
                  annotation_text=f"Barrier {barrier:.0f}")
    fig.add_vline(x=S_current, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        title=f"Delta Profile — {barrier_type}",
        xaxis_title="Spot", yaxis_title="Delta",
        height=380, margin=dict(t=40, b=20),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def barrier_mc_paths_figure(
    time_grid: np.ndarray,
    paths: np.ndarray,
    barrier: float,
    barrier_type: str,
    knocked_mask: np.ndarray,
) -> go.Figure:
    """Fan chart of MC paths — red=knocked-out, teal=survived."""
    fig = go.Figure()
    colors = np.where(knocked_mask, "rgba(214,39,40,0.35)", "rgba(31,119,180,0.35)")
    for i in range(paths.shape[1]):
        fig.add_trace(go.Scatter(
            x=time_grid, y=paths[:, i],
            mode="lines", line=dict(color=colors[i], width=0.8),
            showlegend=False,
        ))
    fig.add_hline(y=barrier, line_color="orange", line_dash="dash",
                  annotation_text=f"{barrier_type} barrier = {barrier:.0f}")
    # Legend proxies
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="rgba(31,119,180,0.8)", width=2),
                             name="Survived"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="rgba(214,39,40,0.8)", width=2),
                             name="Knocked out"))
    fig.update_layout(
        xaxis_title="Time (years)", yaxis_title="Asset Price",
        height=380, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def asian_running_avg_figure(
    time_grid: np.ndarray,
    price_paths: np.ndarray,
    avg_paths: np.ndarray,
    K: float,
) -> go.Figure:
    """Running arithmetic-average paths alongside asset paths, with strike line."""
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Asset Paths", "Running Average"],
                        horizontal_spacing=0.1)
    n = price_paths.shape[1]
    for i in range(n):
        fig.add_trace(go.Scatter(
            x=time_grid, y=price_paths[:, i], mode="lines",
            line=dict(color="rgba(31,119,180,0.35)", width=0.8), showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=time_grid, y=avg_paths[:, i], mode="lines",
            line=dict(color="rgba(44,160,44,0.35)", width=0.8), showlegend=False,
        ), row=1, col=2)
    for col in (1, 2):
        fig.add_hline(y=K, line_dash="dash", line_color="red",
                      annotation_text=f"K={K:.0f}", row=1, col=col)
    fig.update_xaxes(title_text="Time (years)")
    fig.update_yaxes(title_text="Price", col=1)
    fig.update_layout(height=380, margin=dict(t=40, b=20))
    return fig


def asian_vol_comparison_figure(
    sig_grid: np.ndarray,
    vanilla_prices: np.ndarray,
    geo_prices: np.ndarray,
    arith_prices: np.ndarray,
    current_sig: float,
) -> go.Figure:
    """Call price vs sigma: vanilla vs geometric vs arithmetic Asian."""
    fig = go.Figure()
    for y, name, color, dash in [
        (vanilla_prices, "Vanilla BS", "gray", "dot"),
        (geo_prices, "Geometric Asian", "#ff7f0e", "solid"),
        (arith_prices, "Arithmetic Asian (MC)", "#1f77b4", "solid"),
    ]:
        fig.add_trace(go.Scatter(
            x=sig_grid * 100, y=y, mode="lines",
            line=dict(color=color, width=2, dash=dash), name=name,
        ))
    fig.add_vline(x=current_sig * 100, line_dash="dot", line_color="gray", line_width=1)
    fig.update_layout(
        xaxis_title="Volatility σ (%)", yaxis_title="Call Price",
        height=380, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def digital_call_spread_figure(
    dK_grid: np.ndarray,
    cs_prices: np.ndarray,
    digital_price: float,
) -> go.Figure:
    """Call spread price converging to digital (cash-or-nothing) as dK → 0."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dK_grid, y=cs_prices, mode="lines+markers",
        line=dict(color="#1f77b4", width=2), marker=dict(size=5),
        name="Call spread",
    ))
    fig.add_hline(y=digital_price, line_dash="dash", line_color="red",
                  annotation_text=f"Digital = {digital_price:.4f}")
    fig.update_layout(
        xaxis_title="dK (spread width)", yaxis_title="Price",
        xaxis_type="log",
        height=380, margin=dict(t=30, b=20),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def digital_delta_vs_T_figure(
    T_grid: np.ndarray,
    deltas: np.ndarray,
    current_T: float,
) -> go.Figure:
    """Digital call delta spike as T → 0 (ATM delta → ∞)."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=T_grid, y=deltas, mode="lines",
        line=dict(color="#d62728", width=2), name="Delta",
    ))
    fig.add_vline(x=current_T, line_dash="dot", line_color="gray", line_width=1,
                  annotation_text=f"T={current_T:.2f}")
    fig.update_layout(
        xaxis_title="Time to Expiry T (years)", yaxis_title="Delta",
        height=380, margin=dict(t=30, b=20),
    )
    return fig


def autocall_schedule_figure(
    obs_dates: list,
    S_paths: np.ndarray,
    S0: float,
    autocall_lvl: float,
    coupon_barrier: float | None,
    prot_barrier: float,
) -> go.Figure:
    """Swimlane chart: MC paths with autocall/coupon/protection barrier lines."""
    fig = go.Figure()
    n_show = min(S_paths.shape[1], 80)
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            x=obs_dates, y=S_paths[:, i] / S0, mode="lines+markers",
            line=dict(color="rgba(31,119,180,0.25)", width=0.8),
            marker=dict(size=3), showlegend=False,
        ))
    fig.add_hline(y=autocall_lvl, line_color="green", line_width=2,
                  annotation_text=f"Autocall {autocall_lvl:.0%}")
    if coupon_barrier is not None:
        fig.add_hline(y=coupon_barrier, line_color="orange", line_width=1.5,
                      line_dash="dash", annotation_text=f"Coupon {coupon_barrier:.0%}")
    fig.add_hline(y=prot_barrier, line_color="red", line_width=1.5,
                  line_dash="dot", annotation_text=f"Protection {prot_barrier:.0%}")
    fig.update_layout(
        xaxis_title="Observation Date (years)", yaxis_title="S / S₀ (performance)",
        height=420, margin=dict(t=30, b=20),
    )
    return fig


def multi_asset_corr_sensitivity_figure(
    rho_grid: np.ndarray,
    prices: np.ndarray,
    product_label: str,
) -> go.Figure:
    """Option price vs uniform pairwise correlation ρ."""
    fig = go.Figure()
    valid = ~np.isnan(prices)
    fig.add_trace(go.Scatter(
        x=rho_grid[valid], y=prices[valid], mode="lines+markers",
        line=dict(color="#1f77b4", width=2), marker=dict(size=5),
        name=product_label,
    ))
    fig.update_layout(
        xaxis_title="Pairwise Correlation ρ", yaxis_title="Option Price",
        xaxis=dict(range=[-1, 1], tickformat=".1f"),
        height=380, margin=dict(t=30, b=20),
    )
    return fig


def decomposition_waterfall_figure(components: list[dict]) -> go.Figure:
    """Waterfall chart of structured product building blocks.

    components: list of dicts with keys 'component' (str) and 'value' (float).
    """
    labels = [c["component"] for c in components]
    values = [c["value"] for c in components]
    n = len(values)

    measure = ["relative"] * (n - 1) + ["total"]
    text = [f"{v:+.2f}" if i < n - 1 else f"{v:.2f}" for i, v in enumerate(values)]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        text=text,
        textposition="outside",
        connector=dict(line=dict(color="rgb(63,63,63)")),
        increasing=dict(marker=dict(color="#2ca02c")),
        decreasing=dict(marker=dict(color="#d62728")),
        totals=dict(marker=dict(color="#1f77b4")),
    ))
    fig.update_layout(
        yaxis_title="Value", height=380, margin=dict(t=30, b=20),
        showlegend=False,
    )
    return fig


def smile_figure(
    call_strikes: np.ndarray | None,
    iv_calls: np.ndarray | None,
    put_strikes: np.ndarray | None,
    iv_puts: np.ndarray | None,
    model_curves: list[dict],
    expiry: str,
    spot: float,
) -> go.Figure:
    """IV smile: market dots (calls/puts) + one line per fitted model.

    model_curves: list of dicts with keys:
        name   : str
        strikes: np.ndarray
        iv     : np.ndarray  (decimal, not %)
        color  : str
        dash   : str  ('solid', 'dash', 'dot', ...)
    """
    fig = go.Figure()

    if call_strikes is not None and iv_calls is not None:
        fig.add_trace(go.Scatter(
            x=call_strikes, y=iv_calls * 100,
            mode="markers",
            marker=dict(symbol="triangle-up", size=7, color="steelblue"),
            name="Market IV (call)",
        ))

    if put_strikes is not None and iv_puts is not None:
        fig.add_trace(go.Scatter(
            x=put_strikes, y=iv_puts * 100,
            mode="markers",
            marker=dict(symbol="circle", size=7, color="tomato"),
            name="Market IV (put)",
        ))

    for mc in model_curves:
        if mc["strikes"] is not None and len(mc["strikes"]) > 0:
            fig.add_trace(go.Scatter(
                x=mc["strikes"], y=mc["iv"] * 100,
                mode="lines",
                line=dict(color=mc["color"], width=2, dash=mc["dash"]),
                name=mc["name"],
            ))

    fig.add_vline(x=spot, line_dash="dash", line_color="gray",
                  annotation_text=f"Spot ${spot:.0f}")

    fig.update_layout(
        title=f"IV Smile — {expiry}",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        height=440,
        legend=dict(orientation="h", y=-0.22),
    )
    return fig

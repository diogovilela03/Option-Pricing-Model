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
        colorscale="RdYlGn",
        showscale=True,
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


def smile_figure(
    strikes: np.ndarray,
    iv_calls: np.ndarray | None,
    iv_puts: np.ndarray | None,
    call_strikes: np.ndarray | None,
    put_strikes: np.ndarray | None,
    svi_strikes: np.ndarray,
    svi_iv: np.ndarray,
    expiry: str,
    spot: float,
) -> go.Figure:
    """IV smile: market dots (calls/puts) + SVI fitted curve."""
    fig = go.Figure()

    if call_strikes is not None and iv_calls is not None:
        fig.add_trace(go.Scatter(
            x=call_strikes, y=iv_calls * 100,
            mode="markers",
            marker=dict(symbol="triangle-up", size=8, color="steelblue"),
            name="Market IV (call)",
        ))

    if put_strikes is not None and iv_puts is not None:
        fig.add_trace(go.Scatter(
            x=put_strikes, y=iv_puts * 100,
            mode="markers",
            marker=dict(symbol="circle", size=8, color="tomato"),
            name="Market IV (put)",
        ))

    fig.add_trace(go.Scatter(
        x=svi_strikes, y=svi_iv * 100,
        mode="lines",
        line=dict(color="black", width=2),
        name="SVI fit",
    ))

    fig.add_vline(x=spot, line_dash="dash", line_color="gray", annotation_text=f"Spot ${spot:.0f}")

    fig.update_layout(
        title=f"IV Smile — {expiry}",
        xaxis_title="Strike",
        yaxis_title="Implied Volatility (%)",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig

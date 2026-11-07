"""
Charts for the Overview screen (Layer 2).

Rules followed: one y-axis per chart (no dual axes), thin marks, recessive
grid, a legend whenever there are two or more series, hover on every mark, and
colours validated for contrast and colour-blind separation.
"""
import warnings

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

PRIMARY = "#00869C"
SECONDARY = "#D97A2B"
INK = "#15202B"
SLATE = "#5A6472"
GRID = "#EEF0F3"


def _base_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=SLATE),
        hoverlabel=dict(bgcolor="#FFFFFF", bordercolor="#E2E5EA", font=dict(color=INK, family="Inter")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=INK)),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E5EA", ticks="", tickfont=dict(color=SLATE))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor="rgba(0,0,0,0)", tickfont=dict(color=SLATE))
    return fig


def _empty(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text, showarrow=False, font=dict(color=SLATE))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _base_layout(fig)


def render_sales_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Daily sales (bars) with the 7-day average (line) — same rupee axis."""
    if df.empty:
        return _empty("No sales in this period")
    df = df.copy()
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["avg7"] = df["daily_revenue"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_bar(
        x=df["sale_date"], y=df["daily_revenue"], name="Daily sales", marker_color=PRIMARY, opacity=0.55,
        marker_line_width=0,
        customdata=df["transaction_count"],
        hovertemplate="%{x|%a %d %b}<br>Sales ₹%{y:,.0f}<br>%{customdata} bills<extra></extra>",
    )
    fig.add_scatter(
        x=df["sale_date"], y=df["avg7"], name="7-day average", mode="lines",
        line=dict(color=INK, width=2), hovertemplate="7-day average ₹%{y:,.0f}<extra></extra>",
    )
    fig.update_yaxes(tickprefix="₹", tickformat="~s")
    fig.update_xaxes(tickformat="%d %b")
    return _base_layout(fig)


def render_hourly_traffic_chart(df: pd.DataFrame) -> go.Figure:
    """Visitors vs bills per hour — both are counts, so one axis."""
    if df.empty:
        return _empty("No visitor data for this day")
    labels = [f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}" for h in df["hour"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=df["visitors"], name="Visitors", marker_color=PRIMARY, marker_line_width=0,
                hovertemplate="%{x}<br>%{y} visitors<extra></extra>")
    fig.add_bar(x=labels, y=df["bills"], name="Bills", marker_color=SECONDARY, marker_line_width=0,
                customdata=df["conversion_pct"],
                hovertemplate="%{x}<br>%{y} bills (%{customdata:.0f}% of visitors)<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return _base_layout(fig)


def render_category_donut_chart(df: pd.DataFrame) -> go.Figure:
    """Kept for compatibility: stock value by category as a simple bar (easier to compare than a donut)."""
    if df.empty:
        return _empty("No stock data")
    cat = df.groupby("category")["stock_value"].sum().sort_values()
    fig = go.Figure(go.Bar(x=cat.values, y=cat.index, orientation="h", marker_color=PRIMARY,
                           hovertemplate="%{y}<br>₹%{x:,.0f}<extra></extra>"))
    fig.update_xaxes(tickprefix="₹", tickformat="~s")
    return _base_layout(fig, height=260)

"""Charts for the Sales Forecast screen (same style rules as dashboard_views: one axis, thin marks, legend, hover)."""
from datetime import timedelta
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go

from src.dashboard_views import INK, PRIMARY, SECONDARY, SLATE, _base_layout, _empty

BAND = "rgba(0,134,156,0.14)"
FESTIVAL = "rgba(217,122,43,0.10)"


def _festival_spans(fig: go.Figure, events: List[Dict], x0, x1) -> None:
    """Shade each festival once (several product groups share the same dates)."""
    seen = set()
    for e in events:
        key = (e["name"], e["start"], e["end"])
        if key in seen:
            continue
        seen.add(key)
        s, t = max(e["start"], x0), min(e["end"], x1)
        if s > t:
            continue
        fig.add_vrect(x0=s, x1=t + timedelta(days=1), fillcolor=FESTIVAL, line_width=0, layer="below",
                      annotation_text=e["name"].replace(" Pre-Festival Surge", "").replace(" Season", ""),
                      annotation_position="top left", annotation_font=dict(size=11, color=SLATE))


def outlook_chart(history: pd.DataFrame, forecast: pd.DataFrame, low: float, high: float, events: List[Dict],
                  value: str = "revenue") -> go.Figure:
    """Last 60 days actual (bars) and the next 30 days forecast (line) with its likely range."""
    if history.empty and forecast.empty:
        return _empty("No sales history yet")
    fig = go.Figure()
    money = value == "revenue"
    fmt = "₹%{y:,.0f}" if money else "%{y:.0f} cans"
    fig.add_bar(x=history["day"], y=history[value], name="Actual", marker_color=PRIMARY, opacity=0.45,
                marker_line_width=0, hovertemplate="%{x|%a %d %b}<br>" + fmt + "<extra>Actual</extra>")
    f = forecast
    fig.add_scatter(x=list(f["day"]) + list(f["day"])[::-1],
                    y=list(f[value] * high) + list(f[value] * low)[::-1],
                    fill="toself", fillcolor=BAND, line=dict(width=0), name="Likely range", hoverinfo="skip")
    fig.add_scatter(x=f["day"], y=f[value], mode="lines", name="AI forecast", line=dict(color=INK, width=2),
                    hovertemplate="%{x|%a %d %b}<br>" + fmt + "<extra>Forecast</extra>")
    x0 = history["day"].min() if not history.empty else f["day"].min()
    x1 = f["day"].max()
    _festival_spans(fig, events, x0, x1)
    if not f.empty:
        fig.add_vline(x=f["day"].min(), line=dict(color=SLATE, width=1, dash="dot"))
    if money:
        fig.update_yaxes(tickprefix="₹", tickformat="~s")
    fig.update_xaxes(tickformat="%d %b")
    return _base_layout(fig, height=340)


def backtest_weeks_chart(bt: Dict) -> go.Figure:
    """The last 4 weeks, forecast without seeing them: actual vs AI vs simple average, week by week."""
    dates = bt["dates"]
    weeks, labels = [], []
    for i in range(0, len(dates) - len(dates) % 7, 7):
        weeks.append(slice(i, i + 7))
        labels.append(f"{dates[i]:%d %b}–{dates[i + 6]:%d %b}")
    tot = lambda arr: [float(arr[w].sum()) for w in weeks]
    fig = go.Figure()
    fig.add_bar(x=labels, y=tot(bt["actual"]), name="Actually sold", marker_color=INK, marker_line_width=0,
                hovertemplate="%{x}<br>₹%{y:,.0f}<extra>Actual</extra>")
    fig.add_bar(x=labels, y=tot(bt["predicted"]), name="AI forecast", marker_color=PRIMARY, marker_line_width=0,
                hovertemplate="%{x}<br>₹%{y:,.0f}<extra>AI forecast</extra>")
    fig.add_bar(x=labels, y=tot(bt["simple"]), name="Simple 30-day average", marker_color=SECONDARY,
                marker_line_width=0, hovertemplate="%{x}<br>₹%{y:,.0f}<extra>Simple average</extra>")
    fig.update_layout(barmode="group", bargroupgap=0.06)
    fig.update_yaxes(tickprefix="₹", tickformat="~s")
    return _base_layout(fig, height=300)


def rolling_errors_chart(periods: pd.DataFrame) -> go.Figure:
    """How far off each 2-week forecast was per product group, AI vs simple average."""
    if periods.empty:
        return _empty("Not enough history")
    labels = [f"{r['from']:%d %b}" for _, r in periods.iterrows()]
    fig = go.Figure()
    fig.add_bar(x=labels, y=periods["ai_group_err_pct"], name="AI forecast", marker_color=PRIMARY, marker_line_width=0,
                hovertemplate="2 weeks from %{x}<br>off by %{y:.1f}%<extra>AI</extra>")
    fig.add_bar(x=labels, y=periods["simple_group_err_pct"], name="Simple 30-day average", marker_color=SECONDARY,
                marker_line_width=0, hovertemplate="2 weeks from %{x}<br>off by %{y:.1f}%<extra>Simple</extra>")
    fig.update_layout(barmode="group", bargroupgap=0.06)
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return _base_layout(fig, height=260)


def weekday_chart(factors: List[float]) -> go.Figure:
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    pct = [(f - 1) * 100 for f in factors]
    colours = [PRIMARY if p >= 0 else SECONDARY for p in pct]
    fig = go.Figure(go.Bar(x=days, y=pct, marker_color=colours, marker_line_width=0,
                           hovertemplate="%{x}: %{y:+.0f}% vs an average day<extra></extra>"))
    fig.update_yaxes(ticksuffix="%", zeroline=True, zerolinecolor="#C9CED6")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=220)


def product_chart(history: pd.DataFrame, forecast: pd.DataFrame, low: float, high: float, events: List[Dict]) -> go.Figure:
    """One product: cans per day, last 60 days and the next 30."""
    return outlook_chart(history, forecast, low, high, events, value="units")

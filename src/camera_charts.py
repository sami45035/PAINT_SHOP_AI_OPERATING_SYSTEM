"""Charts for the Store Camera screens (same style rules as dashboard_views)."""
import pandas as pd
import plotly.graph_objects as go

from src.dashboard_views import INK, PRIMARY, SECONDARY, SLATE, _base_layout, _empty

TERTIARY = "#6B5BD2"


def _h(h: int) -> str:
    return f"{(int(h) % 12) or 12} {'AM' if int(h) < 12 else 'PM'}"


def funnel_chart(b: dict) -> go.Figure:
    """Came in → stood at a shelf → reached the counter → bills."""
    steps = [("Came in", b["visitors"]), ("Stood at a shelf", b["at_shelves"]), ("Reached the counter", b["at_counter"])]
    if b.get("bills") is not None:
        steps.append(("Bills at the till", b["bills"]))
    if not b["visitors"]:
        return _empty("No visitors counted yet today")
    labels = [s[0] for s in steps][::-1]
    vals = [s[1] for s in steps][::-1]
    base = max(1, b["visitors"])
    colours = [SECONDARY if l.startswith("Bills") else PRIMARY for l in labels]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker_color=colours, marker_line_width=0,
        text=[f"{v}  ·  {v / base * 100:.0f}%" for v in vals], textposition="outside", cliponaxis=False,
        textfont=dict(color=INK), hovertemplate="%{y}: %{x}<extra></extra>"))
    fig.update_xaxes(visible=False, range=[0, max(vals) * 1.25])
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=220)


def hourly_buyers_chart(hourly: pd.DataFrame, real_till: bool = True) -> go.Figure:
    if hourly is None or hourly.empty:
        return _empty("No visitors counted yet today")
    labels = [_h(h) for h in hourly["hour"]]
    fig = go.Figure()
    fig.add_bar(x=labels, y=hourly["visitors"], name="Came in", marker_color=PRIMARY, marker_line_width=0,
                hovertemplate="%{x}<br>%{y} came in<extra></extra>")
    if real_till:
        fig.add_bar(x=labels, y=hourly["bills"], name="Bills", marker_color=SECONDARY, marker_line_width=0,
                    customdata=hourly["bought_pct"].fillna(0),
                    hovertemplate="%{x}<br>%{y} bills (%{customdata:.0f}% bought)<extra></extra>")
    else:
        fig.add_bar(x=labels, y=hourly["at_counter"], name="Reached the counter", marker_color=SECONDARY,
                    marker_line_width=0, hovertemplate="%{x}<br>%{y} at the counter<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.08)
    return _base_layout(fig, height=280)


def daily_bought_chart(daily: pd.DataFrame) -> go.Figure:
    """Share of visitors who bought, per day (one axis: percent)."""
    if daily is None or daily.empty:
        return _empty("Not enough days yet")
    d = daily.copy()
    d["day"] = pd.to_datetime(d["day"])
    avg = d["bills"].sum() / max(1, d["visitors"].sum()) * 100
    fig = go.Figure()
    fig.add_scatter(x=d["day"], y=d["bought_pct"], mode="lines+markers", name="Visitors who bought",
                    line=dict(color=PRIMARY, width=2), marker=dict(size=6),
                    customdata=d[["visitors", "bills"]].values,
                    hovertemplate="%{x|%a %d %b}<br>%{y:.0f}% bought<br>%{customdata[1]} bills / %{customdata[0]} visitors<extra></extra>")
    fig.add_hline(y=avg, line=dict(color=SLATE, width=1, dash="dot"),
                  annotation_text=f"14-day average {avg:.0f}%", annotation_position="top left",
                  annotation_font=dict(color=SLATE, size=11))
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    fig.update_xaxes(tickformat="%d %b")
    fig.update_layout(showlegend=False)
    return _base_layout(fig, height=240)


def picks_vs_sold_chart(watched: pd.DataFrame) -> go.Figure:
    """Per watched product: cans picked up, put back, and sold this week."""
    if watched is None or watched.empty:
        return _empty("No watched shelves yet")
    w = watched.sort_values("picked")
    fig = go.Figure()
    fig.add_bar(y=w["label"], x=w["picked"], name="Picked up", orientation="h", marker_color=PRIMARY,
                marker_line_width=0, hovertemplate="%{y}<br>picked up %{x}×<extra></extra>")
    fig.add_bar(y=w["label"], x=w["put_back"], name="Put back", orientation="h", marker_color=TERTIARY,
                marker_line_width=0, hovertemplate="%{y}<br>put back %{x}×<extra></extra>")
    fig.add_bar(y=w["label"], x=w["sold"], name="Sold at the till", orientation="h", marker_color=SECONDARY,
                marker_line_width=0, hovertemplate="%{y}<br>sold %{x}<extra></extra>")
    fig.update_layout(barmode="group", bargroupgap=0.05)
    return _base_layout(fig, height=60 + 52 * len(w))

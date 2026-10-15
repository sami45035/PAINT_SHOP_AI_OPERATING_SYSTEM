"""
Dashboard Views & Plotly Visualization Components (Layer 2)
Provides interactive, high-aesthetic executive charts and KPI builders for Streamlit.
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_
if not hasattr(np, "string_"):
    np.string_ = np.bytes_

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any, Tuple

def render_sales_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders 30-day daily sales revenue and volume trend with rolling average.
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No sales data available", showarrow=False)
        return fig

    # Calculate 7-day rolling average
    df = df.copy()
    df["rolling_7d_rev"] = df["daily_revenue"].rolling(window=7, min_periods=1).mean()

    fig = go.Figure()

    # Daily revenue bars
    fig.add_trace(go.Bar(
        x=df["sale_date"],
        y=df["daily_revenue"],
        name="Daily Revenue (₹)",
        marker_color="rgba(59, 130, 246, 0.45)",
        hovertemplate="Date: %{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>"
    ))

    # Rolling average trend line
    fig.add_trace(go.Scatter(
        x=df["sale_date"],
        y=df["rolling_7d_rev"],
        name="7-Day Moving Avg",
        line=dict(color="#2563EB", width=3, shape="spline"),
        hovertemplate="7d Avg: ₹%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title="<b>Daily Sales Revenue & 7-Day Trend (₹)</b>",
        xaxis_title="Date",
        yaxis_title="Revenue (₹)",
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_footfall_conversion_chart(df: pd.DataFrame) -> go.Figure:
    """
    Dual-axis chart showing Hourly Visitors (Footfall) and Sales Transactions,
    with an overlay of the Hourly Conversion Rate (%).
    """
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No footfall data available", showarrow=False)
        return fig

    # Take most recent 24 hourly periods
    sub_df = df.tail(24).copy()
    sub_df["time_label"] = pd.to_datetime(sub_df["hour_slot"]).dt.strftime("%d %b %H:%M")

    fig = go.Figure()

    # Visitors Bar
    fig.add_trace(go.Bar(
        x=sub_df["time_label"],
        y=sub_df["visitors"],
        name="Visitors (Footfall)",
        marker_color="rgba(16, 185, 129, 0.6)",
        yaxis="y1"
    ))

    # Transactions Bar
    fig.add_trace(go.Bar(
        x=sub_df["time_label"],
        y=sub_df["transactions"],
        name="POS Transactions",
        marker_color="rgba(245, 158, 11, 0.8)",
        yaxis="y1"
    ))

    # Conversion Rate Line
    fig.add_trace(go.Scatter(
        x=sub_df["time_label"],
        y=sub_df["conversion_rate_pct"],
        name="Conversion Rate (%)",
        line=dict(color="#EF4444", width=3, dash="dot"),
        yaxis="y2"
    ))

    fig.update_layout(
        title="<b>Store Traffic vs. Sales Conversion (Hourly)</b>",
        xaxis_title="Hour Slot",
        yaxis=dict(title="Count (People / Transactions)"),
        yaxis2=dict(
            title="Conversion Rate (%)",
            overlaying="y",
            side="right",
            range=[0, 100],
            showgrid=False
        ),
        barmode="group",
        template="plotly_white",
        margin=dict(l=40, r=40, t=50, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def render_category_donut_chart(df: pd.DataFrame) -> go.Figure:
    """
    Renders inventory distribution and stock value by category.
    """
    if df.empty:
        return go.Figure()

    cat_summary = df.groupby("category").agg({
        "stock_value": "sum",
        "current_stock": "sum"
    }).reset_index()

    fig = px.pie(
        cat_summary,
        values="stock_value",
        names="category",
        hole=0.45,
        title="<b>Inventory Valuation by Category (₹)</b>",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Tied Capital: ₹%{value:,.0f}<br>Share: %{percent}<extra></extra>"
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig

def render_fast_slow_moving_chart(top_df: pd.DataFrame, slow_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """
    Returns two bar charts:
    1. Top 7 Fast-Moving Best Sellers
    2. Top 7 Dead / Slow-Moving Capital Traps
    """
    # 1. Fast Moving
    top_7 = top_df.head(7).copy()
    fig_top = go.Figure(go.Bar(
        x=top_7["total_units_sold"],
        y=top_7["shade_name"] + " (" + top_7["brand"] + ")",
        orientation="h",
        marker=dict(
            color=top_7["total_units_sold"],
            colorscale="Viridis",
            showscale=False
        ),
        hovertemplate="Units Sold: %{x}<br>Revenue: ₹%{customdata:,.0f}<extra></extra>",
        customdata=top_7["total_revenue"]
    ))
    fig_top.update_layout(
        title="<b>🚀 Fast-Moving Top Shades (90 Days)</b>",
        xaxis_title="Units Sold",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=140, r=20, t=40, b=40)
    )

    # 2. Slow Moving
    slow_7 = slow_df.head(7).copy()
    fig_slow = go.Figure(go.Bar(
        x=slow_7["tied_up_capital"],
        y=slow_7["shade_name"] + " (" + slow_7["brand"] + ")",
        orientation="h",
        marker=dict(
            color="rgba(239, 68, 68, 0.7)"
        ),
        hovertemplate="Tied Capital: ₹%{x:,.0f}<br>Stock: %{customdata} units<extra></extra>",
        customdata=slow_7["current_stock"]
    ))
    fig_slow.update_layout(
        title="<b>⚠️ Stagnant Stock Capital Traps</b>",
        xaxis_title="Capital Tied Up (₹)",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        margin=dict(l=140, r=20, t=40, b=40)
    )

    return fig_top, fig_slow

def render_expiry_timeline_chart(inv_df: pd.DataFrame) -> go.Figure:
    """
    Visualizes batches approaching expiration in 3 buckets:
    < 30 days (Critical), 30-60 days (Warning), > 60 days (Safe)
    """
    sub_df = inv_df[inv_df["current_stock"] > 0].copy()
    
    def bucket_expiry(days):
        if days <= 30:
            return "Critical (< 30d)"
        elif days <= 60:
            return "Warning (30-60d)"
        else:
            return "Safe (> 60d)"

    sub_df["expiry_bucket"] = sub_df["days_to_expiry"].apply(bucket_expiry)
    bucket_counts = sub_df.groupby("expiry_bucket")["stock_value"].sum().reset_index()

    color_map = {
        "Critical (< 30d)": "#EF4444",
        "Warning (30-60d)": "#F59E0B",
        "Safe (> 60d)": "#10B981"
    }

    fig = px.bar(
        bucket_counts,
        x="expiry_bucket",
        y="stock_value",
        color="expiry_bucket",
        color_discrete_map=color_map,
        title="<b>Stock Value at Risk by Expiry Timeline</b>"
    )
    fig.update_layout(
        xaxis_title="Shelf-Life Status",
        yaxis_title="Tied Capital (₹)",
        showlegend=False,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig

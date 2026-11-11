"""
Cached data access for the screens. Short time-to-live so camera counts,
orders and alert actions show up quickly; call clear_all() after a write.
"""
import streamlit as st

from src import database as db
from src.predictor import compute_sales_forecast
from src.smart_alerts import alert_id, flatten_alerts, generate_smart_alerts

TTL = 30


@st.cache_data(ttl=TTL, show_spinner=False)
def sales_summary(day_str: str):
    return db.get_sales_summary(30)


@st.cache_data(ttl=TTL, show_spinner=False)
def daily_sales(day_str: str, days: int = 45):
    return db.get_daily_sales_timeseries(days)


@st.cache_data(ttl=TTL, show_spinner=False)
def hourly_traffic(day_str: str):
    from datetime import datetime

    return db.get_hourly_traffic(datetime.strptime(day_str, "%Y-%m-%d").date())


@st.cache_data(ttl=TTL, show_spinner=False)
def top_and_slow(day_str: str):
    return db.get_top_and_slow_shades(7)


@st.cache_data(ttl=TTL, show_spinner=False)
def season(day_str: str):
    return db.get_season_status()


@st.cache_data(ttl=TTL, show_spinner=False)
def inventory(day_str: str):
    return db.get_inventory_status()


@st.cache_data(ttl=TTL, show_spinner=False)
def forecast(day_str: str):
    return compute_sales_forecast(30)


@st.cache_data(ttl=TTL, show_spinner=False)
def _all_alerts(day_str: str):
    return flatten_alerts(generate_smart_alerts())


def open_alerts(day_str: str):
    """Alerts not marked done / hidden."""
    hidden = db.get_hidden_alert_ids()
    items = []
    for a in _all_alerts(day_str):
        a = dict(a)
        a["id"] = alert_id(a)
        if a["id"] not in hidden:
            items.append(a)
    return items


def clear_all():
    st.cache_data.clear()


# --- Shop camera insights (short TTL: a live camera keeps adding rows) -------------
CAM_TTL = 10


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_footfall(source: str, day_str: str):
    from src import shop_insights as si

    return si.footfall(source)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_buyers(source: str, day_str: str):
    from src import shop_insights as si

    return si.buyers(source)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_daily(source: str, day_str: str):
    from src import shop_insights as si

    return si.daily_footfall(source, 14)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_shelves(source: str, day_str: str):
    from src import shop_insights as si

    return si.shelves(source)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_fast_movers(source: str, day_str: str):
    from src import shop_insights as si

    return si.fast_movers(source, 7)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_preorder(source: str, day_str: str):
    from src import shop_insights as si

    return si.preorder(source)


@st.cache_data(ttl=CAM_TTL, show_spinner=False)
def cam_theft(source: str, day_str: str):
    from src import shop_insights as si

    return si.theft(source)


# --- AI sales forecast (refit at most every 10 minutes) ------------------------------
FC_TTL = 600


@st.cache_data(ttl=FC_TTL, show_spinner="Fitting the sales forecast…")
def forecast_outlook(day_str: str):
    from src import sales_forecast as sf

    return sf.outlook(30)


@st.cache_data(ttl=FC_TTL, show_spinner=False)
def forecast_backtest(day_str: str):
    from src import sales_forecast as sf

    return sf.backtest(28)


@st.cache_data(ttl=FC_TTL, show_spinner=False)
def forecast_rolling(day_str: str, horizon: int):
    from src import sales_forecast as sf

    return sf.rolling_backtest(horizon)


@st.cache_data(ttl=FC_TTL, show_spinner=False)
def forecast_product_history(day_str: str, product_id: int):
    from src import sales_forecast as sf

    return sf.product_history(product_id)


@st.cache_data(ttl=FC_TTL, show_spinner=False)
def forecast_product(day_str: str, product_id: int):
    f = forecast_outlook(day_str)["model"].forecast(30)
    return f[f["product_id"] == product_id].sort_values("day").reset_index(drop=True)

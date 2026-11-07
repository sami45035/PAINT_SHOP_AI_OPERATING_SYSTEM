"""
Data Access Layer (Layer 1) for Paint Shop AI Operating System.

All dates come from src.clock (store time zone). Nothing in this module is
pinned to a fixed calendar date.
"""
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src import clock
from src.config import DB_PATH, DEMO_MODE, EXPIRY_WARNING_DAYS


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _day(d: Optional[date]) -> date:
    return d or clock.today()


# --- Demo data upkeep ---------------------------------------------------------

def ensure_demo_dates_current() -> int:
    """
    In demo mode, move every seeded date forward so the newest sale is today.

    Expiry dates, purchase dates, sales, visitor logs and security events all
    shift by the same number of days, so "expires in 23 days" stays true.
    The seasonal calendar is NOT shifted — Diwali is on a real date.
    Returns the number of days shifted (0 if already current or demo mode off).
    """
    if not DEMO_MODE:
        return 0
    with get_connection() as conn:
        cur = conn.cursor()
        latest = cur.execute("SELECT MAX(DATE(sale_timestamp)) FROM sales;").fetchone()[0]
        if not latest:
            return 0
        delta = (clock.today() - datetime.strptime(latest, "%Y-%m-%d").date()).days
        if delta <= 0:
            return 0
        mod = f"+{delta} days"
        cur.execute("UPDATE sales SET sale_timestamp = DATETIME(sale_timestamp, ?);", (mod,))
        cur.execute("UPDATE footfall_logs SET timestamp = DATETIME(timestamp, ?);", (mod,))
        cur.execute("UPDATE security_events SET timestamp = DATETIME(timestamp, ?);", (mod,))
        cur.execute(
            "UPDATE inventory SET purchase_date = DATE(purchase_date, ?), expiry_date = DATE(expiry_date, ?);",
            (mod, mod),
        )
        conn.commit()
        return delta


# --- Products & Catalog Queries ---------------------------------------------------

def get_all_products() -> pd.DataFrame:
    query = """
    SELECT p.*, s.name as supplier_name, s.lead_time_days, s.moq_units
    FROM products p
    LEFT JOIN suppliers s ON p.supplier_id = s.id
    ORDER BY p.brand, p.shade_name, p.pack_size_litres;
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)


def get_product_by_id(product_id: int) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM products WHERE id = ?", (product_id,)).fetchone()
        return dict(row) if row else None


def get_product_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
        return dict(row) if row else None


# --- Inventory & Stock Queries ------------------------------------------------------

def get_inventory_status(on_day: Optional[date] = None) -> pd.DataFrame:
    query = """
    SELECT
        i.id as inventory_id, p.id as product_id, p.sku, p.brand, p.sub_brand,
        p.shade_name, p.shade_code, p.hex_code, p.category, p.finish,
        p.pack_size_litres, p.cost_price, p.selling_price,
        i.batch_no, i.current_stock, i.on_order, i.min_threshold, i.max_threshold,
        i.purchase_date, i.expiry_date, i.shelf_location,
        s.name as supplier_name, s.lead_time_days, s.moq_units
    FROM inventory i
    JOIN products p ON i.product_id = p.id
    LEFT JOIN suppliers s ON p.supplier_id = s.id;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    df["stock_value"] = df["current_stock"] * df["cost_price"]
    df["retail_value"] = df["current_stock"] * df["selling_price"]

    ref = _day(on_day)
    df["expiry_date_dt"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["days_to_expiry"] = df["expiry_date_dt"].apply(lambda d: (d - ref).days)

    def classify_status(row):
        if row["current_stock"] <= 0:
            return "OUT_OF_STOCK"
        if row["days_to_expiry"] < 0:
            return "EXPIRED"
        if row["current_stock"] <= row["min_threshold"]:
            return "LOW_STOCK"
        if row["days_to_expiry"] <= EXPIRY_WARNING_DAYS:
            return "EXPIRING_SOON"
        if row["current_stock"] > row["max_threshold"]:
            return "OVERSTOCK"
        return "HEALTHY"

    df["health_status"] = df.apply(classify_status, axis=1)
    return df


# --- Sales, visitors & daily KPIs -------------------------------------------------

def _day_totals(cur, d: date) -> Dict[str, float]:
    ds = d.strftime("%Y-%m-%d")
    sales = cur.execute(
        """
        SELECT COALESCE(SUM(total_amount), 0) AS revenue,
               COALESCE(SUM(quantity), 0) AS units,
               COUNT(DISTINCT invoice_no) AS bills
        FROM sales WHERE DATE(sale_timestamp) = ?;
        """,
        (ds,),
    ).fetchone()
    visitors = cur.execute(
        "SELECT COALESCE(SUM(count_in), 0) FROM footfall_logs WHERE DATE(timestamp) = ?;", (ds,)
    ).fetchone()[0]
    return {
        "revenue": float(sales[0]),
        "units": int(sales[1]),
        "bills": int(sales[2]),
        "visitors": int(visitors),
    }


def _pct_change(new: float, old: float) -> Optional[float]:
    if not old:
        return None
    return round((new - old) / old * 100.0, 1)


def get_sales_summary(days: int = 30, on_day: Optional[date] = None) -> Dict[str, Any]:
    """
    KPIs for one day plus a comparison with the day before.

    Conversion is total bills ÷ total visitors for the day — not an average of
    hourly percentages (that is what produced the misleading 13.8%).
    """
    ref = _day(on_day)
    with get_connection() as conn:
        cur = conn.cursor()
        today_t = _day_totals(cur, ref)
        yday_t = _day_totals(cur, ref - timedelta(days=1))

        start = (ref - timedelta(days=days)).strftime("%Y-%m-%d")
        period = cur.execute(
            """
            SELECT COALESCE(SUM(total_amount), 0), COALESCE(SUM(quantity), 0), COUNT(DISTINCT invoice_no)
            FROM sales WHERE DATE(sale_timestamp) > ? AND DATE(sale_timestamp) <= ?;
            """,
            (start, ref.strftime("%Y-%m-%d")),
        ).fetchone()

        inv = cur.execute(
            """
            SELECT COALESCE(SUM(i.current_stock * p.cost_price), 0),
                   COALESCE(SUM(i.current_stock * p.selling_price), 0),
                   COALESCE(SUM(i.current_stock), 0),
                   COUNT(DISTINCT i.product_id)
            FROM inventory i JOIN products p ON i.product_id = p.id;
            """
        ).fetchone()

        low_stock_count = cur.execute(
            "SELECT COUNT(*) FROM inventory WHERE current_stock <= min_threshold;"
        ).fetchone()[0]

    def conv(t):
        return round(t["bills"] / t["visitors"] * 100.0, 1) if t["visitors"] else None

    return {
        "date": ref.strftime("%Y-%m-%d"),
        "max_date": ref.strftime("%Y-%m-%d"),
        "today_revenue": round(today_t["revenue"], 2),
        "today_units": today_t["units"],
        "today_invoices": today_t["bills"],
        "today_visitors": today_t["visitors"],
        "today_conversion_pct": conv(today_t),
        "yesterday_revenue": round(yday_t["revenue"], 2),
        "yesterday_conversion_pct": conv(yday_t),
        "revenue_change_pct": _pct_change(today_t["revenue"], yday_t["revenue"]),
        "period_revenue": round(period[0], 2),
        "period_units": int(period[1]),
        "period_invoices": int(period[2]),
        "total_inventory_cost": round(inv[0], 2),
        "total_inventory_retail": round(inv[1], 2),
        "total_stock_units": int(inv[2]),
        "total_active_skus": int(inv[3]),
        "low_stock_count": low_stock_count,
    }


def get_daily_sales_timeseries(days: int = 45, on_day: Optional[date] = None) -> pd.DataFrame:
    ref = _day(on_day)
    start = (ref - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    query = """
    SELECT DATE(sale_timestamp) as sale_date,
           SUM(total_amount) as daily_revenue,
           SUM(quantity) as daily_units,
           COUNT(DISTINCT invoice_no) as transaction_count
    FROM sales
    WHERE DATE(sale_timestamp) >= ? AND DATE(sale_timestamp) <= ?
    GROUP BY DATE(sale_timestamp)
    ORDER BY sale_date;
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=(start, ref.strftime("%Y-%m-%d")))


def get_hourly_traffic(on_day: Optional[date] = None) -> pd.DataFrame:
    """Visitors and bills for each opening hour of one day."""
    ds = _day(on_day).strftime("%Y-%m-%d")
    query = """
    WITH v AS (
        SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour, SUM(count_in) AS visitors
        FROM footfall_logs WHERE DATE(timestamp) = ? GROUP BY hour
    ),
    b AS (
        SELECT CAST(strftime('%H', sale_timestamp) AS INTEGER) AS hour,
               COUNT(DISTINCT invoice_no) AS bills, SUM(total_amount) AS revenue
        FROM sales WHERE DATE(sale_timestamp) = ? GROUP BY hour
    )
    SELECT v.hour, v.visitors, COALESCE(b.bills, 0) AS bills, COALESCE(b.revenue, 0) AS revenue
    FROM v LEFT JOIN b ON v.hour = b.hour
    ORDER BY v.hour;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(ds, ds))
    df["conversion_pct"] = (df["bills"] / df["visitors"].where(df["visitors"] > 0) * 100).round(1).fillna(0.0)
    return df


def get_daily_traffic(days: int = 14, on_day: Optional[date] = None) -> pd.DataFrame:
    """
    Visitors and bills per day. Visitors and sales are totalled SEPARATELY and
    then joined — joining the raw tables first multiplied visitors by the
    number of sales rows (371 visitors × 30 sales = the bogus 11,130).
    """
    ref = _day(on_day)
    start = (ref - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    end = ref.strftime("%Y-%m-%d")
    query = """
    WITH v AS (
        SELECT DATE(timestamp) AS day, SUM(count_in) AS visitors
        FROM footfall_logs WHERE DATE(timestamp) BETWEEN ? AND ? GROUP BY day
    ),
    b AS (
        SELECT DATE(sale_timestamp) AS day, COUNT(DISTINCT invoice_no) AS bills
        FROM sales WHERE DATE(sale_timestamp) BETWEEN ? AND ? GROUP BY day
    )
    SELECT v.day, v.visitors, COALESCE(b.bills, 0) AS bills
    FROM v LEFT JOIN b ON v.day = b.day
    ORDER BY v.day;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(start, end, start, end))
    df["conversion_pct"] = (df["bills"] / df["visitors"].where(df["visitors"] > 0) * 100).round(1).fillna(0.0)
    return df


def get_footfall_pos_fusion(days: int = 14, on_day: Optional[date] = None) -> pd.DataFrame:
    """Hourly visitors joined with hourly bills (kept for the AI assistant & tests)."""
    ref = _day(on_day)
    start = (ref - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    end = ref.strftime("%Y-%m-%d")
    query = """
    WITH hourly_sales AS (
        SELECT strftime('%Y-%m-%d %H:00:00', sale_timestamp) as hour_slot,
               COUNT(DISTINCT invoice_no) as transactions,
               SUM(total_amount) as hourly_revenue,
               SUM(quantity) as hourly_units
        FROM sales GROUP BY hour_slot
    ),
    hourly_traffic AS (
        SELECT strftime('%Y-%m-%d %H:00:00', timestamp) as hour_slot,
               SUM(count_in) as visitors, MAX(occupancy) as occupancy, AVG(dwell_time_mins) as dwell_time_mins
        FROM footfall_logs
        WHERE DATE(timestamp) BETWEEN ? AND ?
        GROUP BY hour_slot
    )
    SELECT t.hour_slot, t.visitors,
           COALESCE(s.transactions, 0) as transactions,
           COALESCE(s.hourly_revenue, 0.0) as revenue,
           COALESCE(s.hourly_units, 0) as units_sold,
           t.occupancy, t.dwell_time_mins
    FROM hourly_traffic t
    LEFT JOIN hourly_sales s ON t.hour_slot = s.hour_slot
    ORDER BY t.hour_slot;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(start, end))
    df["conversion_rate_pct"] = (df["transactions"] / df["visitors"].where(df["visitors"] > 0) * 100).round(1).fillna(0.0)
    df["atv"] = (df["revenue"] / df["transactions"].where(df["transactions"] > 0)).round(1).fillna(0.0)
    return df


def get_top_and_slow_shades(limit: int = 10, on_day: Optional[date] = None, window_days: int = 90) -> Dict[str, pd.DataFrame]:
    ref = _day(on_day)
    start = (ref - timedelta(days=window_days)).strftime("%Y-%m-%d")
    with sqlite3.connect(DB_PATH) as conn:
        top_df = pd.read_sql_query(
            """
            SELECT p.shade_name, p.brand, p.category, p.hex_code,
                   SUM(s.quantity) as total_units_sold, SUM(s.total_amount) as total_revenue
            FROM sales s JOIN products p ON s.product_id = p.id
            WHERE DATE(s.sale_timestamp) > ?
            GROUP BY p.shade_name, p.brand, p.category, p.hex_code
            ORDER BY total_units_sold DESC
            LIMIT ?;
            """,
            conn,
            params=(start, limit),
        )
        slow_df = pd.read_sql_query(
            """
            SELECT p.shade_name, p.brand, p.category, p.hex_code, p.pack_size_litres,
                   i.current_stock, (i.current_stock * p.cost_price) as tied_up_capital,
                   COALESCE(SUM(s.quantity), 0) as total_units_sold
            FROM products p
            JOIN inventory i ON p.id = i.product_id
            LEFT JOIN sales s ON p.id = s.product_id AND DATE(s.sale_timestamp) > ?
            WHERE i.current_stock > 0
            GROUP BY p.id
            ORDER BY total_units_sold ASC, tied_up_capital DESC
            LIMIT ?;
            """,
            conn,
            params=(start, limit),
        )
    return {"top_shades": top_df, "slow_shades": slow_df}


# --- Seasonal calendar --------------------------------------------------------------

def get_seasonal_events() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM seasonal_calendar ORDER BY start_date;", conn)


def get_season_status(on_day: Optional[date] = None) -> Dict[str, Any]:
    """Which seasonal events are running today, and which comes next."""
    ref = _day(on_day)
    df = get_seasonal_events()
    if df.empty:
        return {"active": [], "next": None}
    df["start"] = pd.to_datetime(df["start_date"]).dt.date
    df["end"] = pd.to_datetime(df["end_date"]).dt.date
    active = df[(df["start"] <= ref) & (df["end"] >= ref)].sort_values("expected_multiplier", ascending=False)
    upcoming = df[df["start"] > ref].sort_values("start")
    return {
        "active": active.to_dict("records"),
        "next": upcoming.iloc[0].to_dict() if not upcoming.empty else None,
    }


# --- Security events & visitor logging ----------------------------------------------

def get_recent_security_events(limit: int = 10) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            "SELECT * FROM security_events ORDER BY timestamp DESC LIMIT ?;", conn, params=(limit,)
        )


def log_security_event(event_type: str, zone: str, severity: str, details: str, at: Optional[datetime] = None) -> int:
    ts = (at or clock.now_naive()).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO security_events (timestamp, event_type, zone, severity, status, details)
            VALUES (?, ?, ?, ?, 'NEW', ?);
            """,
            (ts, event_type, zone, severity, details),
        )
        conn.commit()
        return cur.lastrowid


def update_security_event_status(event_id: int, status: str) -> None:
    """status: NEW, SEEN or FALSE_ALARM."""
    with get_connection() as conn:
        conn.execute("UPDATE security_events SET status = ? WHERE id = ?;", (status, int(event_id)))
        conn.commit()


def log_footfall(count_in: int = 0, count_out: int = 0, occupancy: Optional[int] = None,
                 at: Optional[datetime] = None) -> None:
    """
    Add people counted at the door to the current hour's row (creating it if
    needed). Called with small increments from the camera, so one hour = one row.
    """
    slot = (at or clock.now_naive()).strftime("%Y-%m-%d %H:00:00")
    with get_connection() as conn:
        row = conn.execute("SELECT id, count_in, count_out FROM footfall_logs WHERE timestamp = ?;", (slot,)).fetchone()
        if row:
            new_in = row["count_in"] + count_in
            new_out = row["count_out"] + count_out
            occ = occupancy if occupancy is not None else max(0, new_in - new_out)
            conn.execute(
                "UPDATE footfall_logs SET count_in = ?, count_out = ?, occupancy = ? WHERE id = ?;",
                (new_in, new_out, occ, row["id"]),
            )
        else:
            occ = occupancy if occupancy is not None else max(0, count_in - count_out)
            conn.execute(
                "INSERT INTO footfall_logs (timestamp, count_in, count_out, occupancy, dwell_time_mins) VALUES (?, ?, ?, ?, ?);",
                (slot, count_in, count_out, occ, 0.0),
            )
        conn.commit()


def get_door_counts_today(on_day: Optional[date] = None) -> Dict[str, int]:
    ds = _day(on_day).strftime("%Y-%m-%d")
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(count_in),0), COALESCE(SUM(count_out),0) FROM footfall_logs WHERE DATE(timestamp) = ?;",
            (ds,),
        ).fetchone()
    return {"in": int(row[0]), "out": int(row[1]), "inside": max(0, int(row[0]) - int(row[1]))}


# Backwards-compatible name used by older code
def log_footfall_entry(count_in: int, count_out: int, occupancy: int, dwell_mins: float = 12.0):
    log_footfall(count_in, count_out, occupancy)


# --- App tables (created on startup if missing) ----------------------------------

def ensure_app_tables() -> None:
    """Small tables the app writes to (kept separate from the seeded data)."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS alert_actions (
                alert_id TEXT PRIMARY KEY,
                action TEXT NOT NULL,
                acted_at TEXT NOT NULL,
                hidden_until TEXT NOT NULL
            );
            """
        )
        conn.commit()


def record_alert_action(alert_id: str, action: str, hide_days: int = 3) -> None:
    """'Mark done' / 'Start offer' etc. The alert is hidden for `hide_days`."""
    now = clock.now_naive()
    until = (clock.today() + timedelta(days=hide_days)).strftime("%Y-%m-%d")
    with get_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO alert_actions (alert_id, action, acted_at, hidden_until) VALUES (?, ?, ?, ?);",
            (alert_id, action, now.strftime("%Y-%m-%d %H:%M:%S"), until),
        )
        conn.commit()


def get_hidden_alert_ids(on_day: Optional[date] = None) -> set:
    ds = _day(on_day).strftime("%Y-%m-%d")
    with get_connection() as conn:
        rows = conn.execute("SELECT alert_id FROM alert_actions WHERE hidden_until >= ?;", (ds,)).fetchall()
    return {r[0] for r in rows}


def mark_ordered(product_id: int, qty: int) -> None:
    """Record that cans were ordered from the supplier (adds to 'on order')."""
    with get_connection() as conn:
        conn.execute("UPDATE inventory SET on_order = on_order + ? WHERE product_id = ?;", (int(qty), int(product_id)))
        conn.commit()

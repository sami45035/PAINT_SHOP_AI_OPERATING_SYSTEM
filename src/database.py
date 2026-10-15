"""
Data Access Layer (Layer 1) for Paint Shop AI Operating System
Provides high-performance SQLite queries, inventory calculations, and analytics aggregations.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from src.config import DB_PATH, LOW_STOCK_DAYS_THRESHOLD, EXPIRY_WARNING_DAYS

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# --- Products & Catalog Queries ---

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
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_product_by_sku(sku: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM products WHERE sku = ?", (sku,))
        row = cursor.fetchone()
        return dict(row) if row else None

# --- Inventory & Stock Queries ---

def get_inventory_status() -> pd.DataFrame:
    query = """
    SELECT 
        i.id as inventory_id,
        p.id as product_id,
        p.sku,
        p.brand,
        p.sub_brand,
        p.shade_name,
        p.shade_code,
        p.hex_code,
        p.category,
        p.finish,
        p.pack_size_litres,
        p.cost_price,
        p.selling_price,
        i.batch_no,
        i.current_stock,
        i.on_order,
        i.min_threshold,
        i.max_threshold,
        i.purchase_date,
        i.expiry_date,
        i.shelf_location,
        s.name as supplier_name,
        s.lead_time_days,
        s.moq_units
    FROM inventory i
    JOIN products p ON i.product_id = p.id
    LEFT JOIN suppliers s ON p.supplier_id = s.id;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)
    
    # Calculate stock health classification
    df["stock_value"] = df["current_stock"] * df["cost_price"]
    df["retail_value"] = df["current_stock"] * df["selling_price"]
    
    today = datetime.now().date()
    df["expiry_date_dt"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["days_to_expiry"] = (df["expiry_date_dt"] - today).apply(lambda d: d.days)

    def classify_status(row):
        if row["current_stock"] <= 0:
            return "OUT_OF_STOCK"
        elif row["current_stock"] <= row["min_threshold"]:
            return "LOW_STOCK"
        elif row["days_to_expiry"] <= EXPIRY_WARNING_DAYS:
            return "EXPIRING_SOON"
        elif row["current_stock"] > row["max_threshold"]:
            return "OVERSTOCK"
        else:
            return "HEALTHY"

    df["health_status"] = df.apply(classify_status, axis=1)
    return df

# --- Sales & Conversion Analytics ---

def get_sales_summary(days: int = 30) -> Dict[str, Any]:
    with get_connection() as conn:
        cursor = conn.cursor()
        # Today's reference: 2026-10-03 (or current date)
        cursor.execute("SELECT MAX(DATE(sale_timestamp)) FROM sales;")
        max_date_str = cursor.fetchone()[0] or "2026-10-03"

        # Today's metrics
        cursor.execute("""
        SELECT 
            COALESCE(SUM(total_amount), 0) as today_revenue,
            COALESCE(SUM(quantity), 0) as today_units,
            COUNT(DISTINCT invoice_no) as today_invoices
        FROM sales
        WHERE DATE(sale_timestamp) = ?;
        """, (max_date_str,))
        today_row = cursor.fetchone()

        # Total 30-day metrics
        start_date_str = (datetime.strptime(max_date_str, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
        cursor.execute("""
        SELECT 
            COALESCE(SUM(total_amount), 0) as period_revenue,
            COALESCE(SUM(quantity), 0) as period_units,
            COUNT(DISTINCT invoice_no) as period_invoices
        FROM sales
        WHERE DATE(sale_timestamp) >= ?;
        """, (start_date_str,))
        period_row = cursor.fetchone()

        # Inventory valuation
        cursor.execute("""
        SELECT 
            COALESCE(SUM(i.current_stock * p.cost_price), 0) as total_inventory_cost,
            COALESCE(SUM(i.current_stock * p.selling_price), 0) as total_inventory_retail,
            COALESCE(SUM(i.current_stock), 0) as total_stock_units,
            COUNT(DISTINCT i.product_id) as total_active_skus
        FROM inventory i
        JOIN products p ON i.product_id = p.id;
        """)
        inv_row = cursor.fetchone()

        # Critical alerts count
        cursor.execute("""
        SELECT COUNT(*) FROM inventory
        WHERE current_stock <= min_threshold;
        """)
        low_stock_count = cursor.fetchone()[0]

        return {
            "max_date": max_date_str,
            "today_revenue": round(today_row["today_revenue"], 2),
            "today_units": int(today_row["today_units"]),
            "today_invoices": int(today_row["today_invoices"]),
            "period_revenue": round(period_row["period_revenue"], 2),
            "period_units": int(period_row["period_units"]),
            "period_invoices": int(period_row["period_invoices"]),
            "total_inventory_cost": round(inv_row["total_inventory_cost"], 2),
            "total_inventory_retail": round(inv_row["total_inventory_retail"], 2),
            "total_stock_units": int(inv_row["total_stock_units"]),
            "total_active_skus": int(inv_row["total_active_skus"]),
            "low_stock_count": low_stock_count
        }

def get_daily_sales_timeseries(days: int = 60) -> pd.DataFrame:
    query = """
    SELECT 
        DATE(sale_timestamp) as sale_date,
        SUM(total_amount) as daily_revenue,
        SUM(quantity) as daily_units,
        COUNT(DISTINCT invoice_no) as transaction_count
    FROM sales
    GROUP BY DATE(sale_timestamp)
    ORDER BY sale_date DESC
    LIMIT ?;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(days,))
    return df.sort_values("sale_date")

def get_footfall_pos_fusion(days: int = 14) -> pd.DataFrame:
    """
    Merges hourly Footfall with POS sales transactions to calculate
    hourly conversion rate (Transactions / Visitors) and Average Transaction Value.
    """
    query = """
    WITH hourly_sales AS (
        SELECT 
            strftime('%Y-%m-%d %H:00:00', sale_timestamp) as hour_slot,
            COUNT(DISTINCT invoice_no) as transactions,
            SUM(total_amount) as hourly_revenue,
            SUM(quantity) as hourly_units
        FROM sales
        GROUP BY hour_slot
    ),
    hourly_traffic AS (
        SELECT 
            timestamp as hour_slot,
            count_in as visitors,
            occupancy,
            dwell_time_mins
        FROM footfall_logs
    )
    SELECT 
        t.hour_slot,
        t.visitors,
        COALESCE(s.transactions, 0) as transactions,
        COALESCE(s.hourly_revenue, 0.0) as revenue,
        COALESCE(s.hourly_units, 0) as units_sold,
        t.occupancy,
        t.dwell_time_mins
    FROM hourly_traffic t
    LEFT JOIN hourly_sales s ON t.hour_slot = s.hour_slot
    ORDER BY t.hour_slot DESC
    LIMIT ?;
    """
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn, params=(days * 12,))
    
    df["conversion_rate_pct"] = df.apply(
        lambda r: round((r["transactions"] / r["visitors"] * 100), 1) if r["visitors"] > 0 else 0.0,
        axis=1
    )
    df["atv"] = df.apply(
        lambda r: round(r["revenue"] / r["transactions"], 1) if r["transactions"] > 0 else 0.0,
        axis=1
    )
    return df.sort_values("hour_slot")

def get_top_and_slow_shades(limit: int = 10) -> Dict[str, pd.DataFrame]:
    with sqlite3.connect(DB_PATH) as conn:
        top_query = """
        SELECT 
            p.shade_name,
            p.brand,
            p.category,
            p.hex_code,
            SUM(s.quantity) as total_units_sold,
            SUM(s.total_amount) as total_revenue
        FROM sales s
        JOIN products p ON s.product_id = p.id
        WHERE DATE(s.sale_timestamp) >= DATE('now', '-90 days')
        GROUP BY p.shade_name, p.brand, p.category, p.hex_code
        ORDER BY total_units_sold DESC
        LIMIT ?;
        """
        top_df = pd.read_sql_query(top_query, conn, params=(limit,))

        slow_query = """
        SELECT 
            p.shade_name,
            p.brand,
            p.category,
            p.hex_code,
            i.current_stock,
            (i.current_stock * p.cost_price) as tied_up_capital,
            COALESCE(SUM(s.quantity), 0) as total_units_sold
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        LEFT JOIN sales s ON p.id = s.product_id AND DATE(s.sale_timestamp) >= DATE('now', '-90 days')
        GROUP BY p.shade_name, p.brand, p.category, p.hex_code, i.current_stock, tied_up_capital
        ORDER BY total_units_sold ASC, tied_up_capital DESC
        LIMIT ?;
        """
        slow_df = pd.read_sql_query(slow_query, conn, params=(limit,))

    return {"top_shades": top_df, "slow_shades": slow_df}

# --- Security Events & Seasonal Calendar ---

def get_recent_security_events(limit: int = 10) -> pd.DataFrame:
    query = """
    SELECT * FROM security_events
    ORDER BY timestamp DESC
    LIMIT ?;
    """
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn, params=(limit,))

def log_security_event(event_type: str, zone: str, severity: str, details: str):
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO security_events (timestamp, event_type, zone, severity, status, details)
        VALUES (?, ?, ?, ?, 'NEW', ?);
        """, (now_str, event_type, zone, severity, details))
        conn.commit()

def log_footfall_entry(count_in: int, count_out: int, occupancy: int, dwell_mins: float = 12.0):
    now_str = datetime.now().strftime("%Y-%m-%d %H:00:00")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO footfall_logs (timestamp, count_in, count_out, occupancy, dwell_time_mins)
        VALUES (?, ?, ?, ?, ?);
        """, (now_str, count_in, count_out, occupancy, dwell_mins))
        conn.commit()

def get_seasonal_events() -> pd.DataFrame:
    query = "SELECT * FROM seasonal_calendar ORDER BY start_date;"
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(query, conn)

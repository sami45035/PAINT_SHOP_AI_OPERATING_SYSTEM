"""
Smart Alerts & Dead-Stock Remediation Engine (Layer 4)
Monitors inventory, sales anomalies, and expiring batches to generate
structured actionable alerts for the store owner.
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any
from src.config import (
    DB_PATH,
    LOW_STOCK_DAYS_THRESHOLD,
    EXPIRY_WARNING_DAYS,
    OVERSTOCK_TURNOVER_DAYS,
    DEAD_STOCK_DAYS,
    SURGE_VELOCITY_MULTIPLIER
)

def generate_smart_alerts() -> Dict[str, List[Dict[str, Any]]]:
    """
    Scans the entire database and produces classified alerts:
    - Critical Low Stock
    - Expiring Stock (< 45 days) with Discount Suggestions
    - Dead Stock & Overstock with Bundling/Return Recommendations
    - Traffic & Sales Conversion Anomalies
    """
    alerts = {
        "critical": [],
        "expiring": [],
        "dead_stock": [],
        "anomalies": []
    }

    with sqlite3.connect(DB_PATH) as conn:
        # 1. Low Stock Scan
        low_stock_query = """
        SELECT 
            p.id as product_id,
            p.sku,
            p.brand,
            p.shade_name,
            p.pack_size_litres,
            p.cost_price,
            p.selling_price,
            i.current_stock,
            i.on_order,
            i.min_threshold,
            s.name as supplier_name,
            s.lead_time_days,
            s.moq_units
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        LEFT JOIN suppliers s ON p.supplier_id = s.id
        WHERE i.current_stock <= i.min_threshold;
        """
        low_df = pd.read_sql_query(low_stock_query, conn)

        for _, row in low_df.iterrows():
            alerts["critical"].append({
                "type": "LOW_STOCK",
                "severity": "CRITICAL" if row["current_stock"] <= 2 else "WARNING",
                "sku": row["sku"],
                "title": f"Low Stock: {row['brand']} {row['shade_name']} ({row['pack_size_litres']}L)",
                "description": f"Only {row['current_stock']} units left (Safety threshold: {row['min_threshold']}). On-order: {row['on_order']} units.",
                "action": f"Order minimum {max(row['moq_units'] or 10, 10)} units from {row['supplier_name']} (Lead time: {row['lead_time_days']} days).",
                "product_id": int(row["product_id"])
            })

        # 2. Expiring Stock Scan (< 45 days)
        # Using 2026-10-03 reference
        expiry_query = """
        SELECT 
            p.id as product_id,
            p.sku,
            p.brand,
            p.shade_name,
            p.pack_size_litres,
            p.cost_price,
            p.selling_price,
            i.batch_no,
            i.current_stock,
            i.expiry_date,
            (i.current_stock * p.cost_price) as capital_at_risk
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        WHERE i.current_stock > 0
          AND DATE(i.expiry_date) <= DATE('2026-10-03', '+45 days');
        """
        exp_df = pd.read_sql_query(expiry_query, conn)

        today_dt = datetime(2026, 10, 3)
        for _, row in exp_df.iterrows():
            exp_dt = datetime.strptime(row["expiry_date"], "%Y-%m-%d")
            days_left = (exp_dt - today_dt).days
            suggested_discount = 25 if days_left <= 20 else 15
            discounted_price = round(row["selling_price"] * (1 - suggested_discount / 100.0), 2)

            alerts["expiring"].append({
                "type": "EXPIRING_BATCH",
                "severity": "CRITICAL" if days_left <= 20 else "WARNING",
                "sku": row["sku"],
                "batch_no": row["batch_no"],
                "title": f"Expiring Batch ({days_left}d left): {row['brand']} {row['shade_name']} ({row['pack_size_litres']}L)",
                "description": f"Batch {row['batch_no']} has {row['current_stock']} units expiring on {row['expiry_date']}. Capital at risk: ₹{row['capital_at_risk']:,.2f}.",
                "action": f"Launch instant {suggested_discount}% clearance promo (Offer at ₹{discounted_price} instead of ₹{row['selling_price']}) to liquidate stock.",
                "product_id": int(row["product_id"]),
                "days_left": days_left
            })

        # 3. Dead Stock / Overstock Scan (Stagnant > 60 days)
        dead_query = """
        SELECT 
            p.id as product_id,
            p.sku,
            p.brand,
            p.shade_name,
            p.category,
            p.pack_size_litres,
            p.cost_price,
            i.current_stock,
            (i.current_stock * p.cost_price) as tied_up_capital,
            COALESCE(SUM(s.quantity), 0) as recent_sales
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        LEFT JOIN sales s ON p.id = s.product_id AND DATE(s.sale_timestamp) >= DATE('2026-10-03', '-60 days')
        WHERE i.current_stock > 10
        GROUP BY p.id, p.sku, p.brand, p.shade_name, p.category, p.pack_size_litres, p.cost_price, i.current_stock
        HAVING recent_sales <= 2;
        """
        dead_df = pd.read_sql_query(dead_query, conn)

        for _, row in dead_df.iterrows():
            action_plan = (
                "Bundle deal: Offer 1L free primer with this item for contractors"
                if "Enamel" in row["category"] or "Exterior" in row["category"]
                else "Request supplier exchange for fast-moving interior white emulsion"
            )
            alerts["dead_stock"].append({
                "type": "DEAD_STOCK",
                "severity": "WARNING",
                "sku": row["sku"],
                "title": f"Dead Capital: {row['brand']} {row['shade_name']} ({row['pack_size_litres']}L)",
                "description": f"{row['current_stock']} units stagnant with only {row['recent_sales']} sales in 60 days. Tied-up capital: ₹{row['tied_up_capital']:,.2f}.",
                "action": action_plan,
                "product_id": int(row["product_id"])
            })

        # 4. Traffic & Conversion Anomaly Scan
        # Look for days where Footfall was up but Conversion was down significantly
        traffic_query = """
        WITH daily_stats AS (
            SELECT 
                DATE(t.timestamp) as day,
                SUM(t.count_in) as daily_footfall,
                COUNT(DISTINCT s.invoice_no) as daily_transactions
            FROM footfall_logs t
            LEFT JOIN sales s ON DATE(t.timestamp) = DATE(s.sale_timestamp)
            GROUP BY day
            ORDER BY day DESC
            LIMIT 7
        )
        SELECT * FROM daily_stats;
        """
        daily_df = pd.read_sql_query(traffic_query, conn)
        
        for _, row in daily_df.iterrows():
            ff = row["daily_footfall"] or 1
            tx = row["daily_transactions"] or 0
            conv = (tx / ff) * 100
            if ff > 120 and conv < 15.0:
                alerts["anomalies"].append({
                    "type": "CONVERSION_DROP",
                    "severity": "WARNING",
                    "title": f"Traffic Conversion Drop on {row['day']}",
                    "description": f"High footfall of {ff} visitors resulted in only {tx} transactions (Conversion: {conv:.1f}% vs ~35% average).",
                    "action": "Check if requested white primer or popular rollers were out of stock, or if billing queue caused customer drop-off."
                })

    return alerts

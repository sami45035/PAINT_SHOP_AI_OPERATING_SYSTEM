"""
Sales Predictor, Lead-Time Purchase Engine & Supplier Clearance Evaluator (Layer 3)
Answers:
1. What should I stock for the next 7 / 30 days?
2. When should I place the order before supplier lead time runs out?
3. Should the shop purchase a discounted supplier clearance batch?
"""
import math
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List
from src.config import DB_PATH

def compute_sales_forecast(forecast_days: int = 30) -> pd.DataFrame:
    """
    Computes baseline sales velocity, applies active seasonal festival multipliers,
    and calculates 7-day and 30-day demand forecasts per product.
    """
    with sqlite3.connect(DB_PATH) as conn:
        # 1. Fetch products & inventory
        prod_query = """
        SELECT 
            p.id as product_id,
            p.sku,
            p.brand,
            p.sub_brand,
            p.shade_name,
            p.category,
            p.pack_size_litres,
            p.cost_price,
            p.selling_price,
            i.current_stock,
            i.on_order,
            i.min_threshold,
            s.id as supplier_id,
            s.name as supplier_name,
            s.lead_time_days,
            s.moq_units
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        LEFT JOIN suppliers s ON p.supplier_id = s.id;
        """
        df = pd.read_sql_query(prod_query, conn)

        # 2. Fetch sales over past 30 days
        sales_query = """
        SELECT 
            product_id,
            SUM(quantity) as units_sold_30d,
            COUNT(DISTINCT invoice_no) as order_count_30d
        FROM sales
        WHERE DATE(sale_timestamp) >= DATE('2026-10-03', '-30 days')
        GROUP BY product_id;
        """
        sales_df = pd.read_sql_query(sales_query, conn)

        # 3. Fetch active/upcoming seasonal events
        cal_query = "SELECT * FROM seasonal_calendar;"
        cal_df = pd.read_sql_query(cal_query, conn)

    df = df.merge(sales_df, on="product_id", how="left").fillna({"units_sold_30d": 0, "order_count_30d": 0})

    # Baseline daily velocity
    df["daily_velocity"] = df["units_sold_30d"] / 30.0

    # Determine seasonal multiplier based on current date (Diwali season active in Oct-Nov)
    today = datetime(2026, 10, 3)
    
    def get_seasonal_multiplier(category):
        mult = 1.0
        for _, event in cal_df.iterrows():
            start = datetime.strptime(event["start_date"], "%Y-%m-%d")
            end = datetime.strptime(event["end_date"], "%Y-%m-%d")
            # If within 15 days before start or during event
            if (start - timedelta(days=15)) <= today <= end:
                if "Interior" in category and ("Interior" in event["impact_category"] or "Emulsion" in event["impact_category"]):
                    mult = max(mult, event["expected_multiplier"])
                elif "Exterior" in category and "Exterior" in event["impact_category"]:
                    mult = max(mult, event["expected_multiplier"])
                elif "Waterproofing" in category and "Waterproofing" in event["impact_category"]:
                    mult = max(mult, event["expected_multiplier"])
        return mult

    df["seasonal_multiplier"] = df["category"].apply(get_seasonal_multiplier)
    df["effective_velocity"] = df["daily_velocity"] * df["seasonal_multiplier"]

    # Forecast demand
    df["forecast_7d"] = (df["effective_velocity"] * 7).apply(math.ceil)
    df["forecast_30d"] = (df["effective_velocity"] * 30).apply(math.ceil)

    # Days to stockout
    df["days_to_stockout"] = df.apply(
        lambda r: round(r["current_stock"] / max(r["effective_velocity"], 0.05), 1),
        axis=1
    )

    # Reorder calculation incorporating supplier lead time & MOQ
    def calculate_reorder(r):
        lead_time = r["lead_time_days"] or 7
        moq = r["moq_units"] or 10
        safety_stock = math.ceil(r["effective_velocity"] * 5)  # 5 days safety buffer
        
        # Gross requirement = 30-day forecast + safety stock - current stock - on_order
        net_needed = r["forecast_30d"] + safety_stock - (r["current_stock"] + r["on_order"])
        
        if net_needed <= 0:
            return 0, "SUFFICIENT_STOCK", "N/A"
        
        # Enforce MOQ
        order_qty = math.ceil(net_needed / moq) * moq
        
        # Urgency based on days to stockout vs supplier lead time
        days_left = r["days_to_stockout"]
        if days_left <= lead_time:
            urgency = "CRITICAL_ORDER_NOW"
            order_date = "TODAY (URGENT)"
        elif days_left <= (lead_time + 4):
            urgency = "ORDER_SOON"
            order_date = f"Within {int(days_left - lead_time)} days"
        else:
            urgency = "PLANNED"
            order_date = f"In {int(days_left - lead_time)} days"

        return order_qty, urgency, order_date

    results = [calculate_reorder(row) for _, row in df.iterrows()]
    df["suggested_reorder_qty"] = [res[0] for res in results]
    df["reorder_urgency"] = [res[1] for res in results]
    df["suggested_order_date"] = [res[2] for res in results]
    df["estimated_order_cost"] = df["suggested_reorder_qty"] * df["cost_price"]

    return df.sort_values(by=["reorder_urgency", "days_to_stockout"], ascending=[True, True])


def evaluate_supplier_clearance_deal(
    product_id: int,
    offered_unit_cost: float,
    lot_quantity: int,
    shelf_life_months: int = 12
) -> Dict[str, Any]:
    """
    Evaluates whether a paint shop should purchase a discounted supplier clearance lot.
    Balances:
    - Clearance Margin vs Standard Margin
    - Forward Demand (will it sell before expiring?)
    - Working Capital & Holding Costs
    - Expiry Risk
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT p.*, i.current_stock
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        WHERE p.id = ?;
        """, (product_id,))
        row = cursor.fetchone()
        if not row:
            return {"error": "Product not found"}
        
        col_names = [d[0] for d in cursor.description]
        prod = dict(zip(col_names, row))

        # 30-day velocity
        cursor.execute("""
        SELECT COALESCE(SUM(quantity), 0) / 30.0 
        FROM sales 
        WHERE product_id = ? AND DATE(sale_timestamp) >= DATE('2026-10-03', '-30 days');
        """, (product_id,))
        daily_velocity = cursor.fetchone()[0] or 0.2

    regular_cost = prod["cost_price"]
    selling_price = prod["selling_price"]
    discount_pct = round(((regular_cost - offered_unit_cost) / regular_cost) * 100, 1)

    # Expected months to sell the lot + existing stock
    total_stock_to_sell = prod["current_stock"] + lot_quantity
    days_to_sell_all = total_stock_to_sell / max(daily_velocity, 0.05)
    months_to_sell_all = round(days_to_sell_all / 30.0, 1)

    # Financial analysis
    total_investment = round(offered_unit_cost * lot_quantity, 2)
    expected_revenue = round(selling_price * lot_quantity, 2)
    gross_profit = round(expected_revenue - total_investment, 2)
    profit_margin_pct = round((gross_profit / expected_revenue) * 100, 1)

    # Annual holding cost estimated at 12% of inventory capital
    estimated_holding_cost = round(total_investment * 0.12 * (months_to_sell_all / 12.0), 2)
    net_projected_profit = round(gross_profit - estimated_holding_cost, 2)

    # Risk assessment
    if months_to_sell_all > shelf_life_months:
        recommendation = "REJECT / HIGH EXPIRY RISK"
        verdict = f"High Risk: It will take ~{months_to_sell_all} months to sell, which exceeds the remaining {shelf_life_months}-month shelf-life."
        action = "Do not purchase this batch; stock will expire before depletion."
        badge = "critical"
    elif months_to_sell_all > (shelf_life_months * 0.75):
        recommendation = "BUY WITH CLEARANCE PROMOTION"
        verdict = f"Moderate Risk: Stock turnover (~{months_to_sell_all} months) is tight against {shelf_life_months}-month expiry."
        action = "Negotiate an extra 10% discount and plan a 15% customer promotional bundle to accelerate sales."
        badge = "warning"
    else:
        recommendation = "STRONG BUY (EXCELLENT ROI)"
        verdict = f"Safe Deal: High {discount_pct}% discount on standard cost. Projected turnover is ~{months_to_sell_all} months (well within {shelf_life_months}m shelf life)."
        action = f"Proceed with purchase of {lot_quantity} units. Estimated net profit: Rs. {net_projected_profit:,.2f}."
        badge = "success"

    return {
        "sku": prod["sku"],
        "shade_name": prod["shade_name"],
        "brand": prod["brand"],
        "pack_size": f"{prod['pack_size_litres']}L",
        "regular_cost": regular_cost,
        "offered_cost": offered_unit_cost,
        "discount_pct": discount_pct,
        "lot_quantity": lot_quantity,
        "total_investment": total_investment,
        "gross_profit": gross_profit,
        "profit_margin_pct": profit_margin_pct,
        "months_to_sell": months_to_sell_all,
        "shelf_life_months": shelf_life_months,
        "net_projected_profit": net_projected_profit,
        "recommendation": recommendation,
        "verdict": verdict,
        "action": action,
        "badge": badge
    }

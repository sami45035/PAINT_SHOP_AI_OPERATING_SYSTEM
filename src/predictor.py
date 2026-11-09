"""
Sales Predictor, Reorder Planner & Supplier Deal Checker (Layer 3)

Answers:
1. What should I stock for the next 7 / 30 days?
2. When must I order, given the supplier's lead time?
3. Should the shop buy a discounted supplier clearance lot?

Method (kept simple on purpose, as the project plan suggests):
- Take the last 30 days of sales for each product.
- Remove the season effect from that history (Diwali sales would otherwise be
  counted twice), giving a "normal day" rate.
- Walk forward day by day, multiplying the normal rate by whichever seasonal
  events are active on THAT day — so a festival ending on 15 Nov only boosts
  the forecast until 15 Nov.
"""
import math
import sqlite3
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src import clock
from src.config import DB_PATH

LOOKBACK_DAYS = 30
SAFETY_DAYS = 5
URGENCY_ORDER = {"CRITICAL_ORDER_NOW": 0, "ORDER_SOON": 1, "ON_ORDER": 2, "PLANNED": 3, "SUFFICIENT_STOCK": 4}
LUXURY_FINISH_WORDS = ("Sheen", "Silk", "Luxury", "Satin", "Glo", "Gloss")


def event_applies(category: str, finish: str, impact: str) -> bool:
    """Does a seasonal event (by its impact_category text) affect this product?"""
    category, finish, impact = category or "", finish or "", impact or ""
    if "Emulsion" in impact and "Emulsion" in category:
        if ("Interior" in impact and "Interior" in category) or ("Exterior" in impact and "Exterior" in category):
            return True
    if "Waterproofing" in impact and "Waterproofing" in category:
        return True
    if "Primer" in impact and category == "Primer":
        return True
    if ("Luxury" in impact or "Accent" in impact) and "Emulsion" in category and any(w in finish for w in LUXURY_FINISH_WORDS):
        return True
    return False


def _multiplier_series(cal_df: pd.DataFrame, category: str, finish: str, days: List[date]) -> List[float]:
    """Seasonal multiplier for each day (the strongest active event wins)."""
    relevant = [
        (ev["start"], ev["end"], float(ev["expected_multiplier"]))
        for _, ev in cal_df.iterrows()
        if event_applies(category, finish, ev["impact_category"])
    ]
    out = []
    for d in days:
        active = [m for s, e, m in relevant if s <= d <= e]
        ups = [m for m in active if m >= 1]
        downs = [m for m in active if m < 1]
        out.append((max(ups) if ups else 1.0) * (min(downs) if downs else 1.0))
    return out


def _load_calendar(conn) -> pd.DataFrame:
    cal = pd.read_sql_query("SELECT * FROM seasonal_calendar;", conn)
    cal["start"] = pd.to_datetime(cal["start_date"]).dt.date
    cal["end"] = pd.to_datetime(cal["end_date"]).dt.date
    return cal


def compute_sales_forecast(forecast_days: int = 30, on_day: Optional[date] = None) -> pd.DataFrame:
    ref = on_day or clock.today()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            """
            SELECT p.id as product_id, p.sku, p.brand, p.sub_brand, p.shade_name, p.hex_code,
                   p.category, p.finish, p.pack_size_litres, p.cost_price, p.selling_price,
                   i.current_stock, i.on_order, i.min_threshold,
                   s.id as supplier_id, s.name as supplier_name, s.lead_time_days, s.moq_units
            FROM products p
            JOIN inventory i ON p.id = i.product_id
            LEFT JOIN suppliers s ON p.supplier_id = s.id;
            """,
            conn,
        )
        sales_df = pd.read_sql_query(
            """
            SELECT product_id, SUM(quantity) as units_sold_30d, COUNT(DISTINCT invoice_no) as order_count_30d
            FROM sales
            WHERE DATE(sale_timestamp) > DATE(?, ?) AND DATE(sale_timestamp) <= DATE(?) AND sale_timestamp <= ?
            GROUP BY product_id;
            """,
            conn,
            params=(ref.strftime("%Y-%m-%d"), f"-{LOOKBACK_DAYS} days", ref.strftime("%Y-%m-%d"),
                    clock.now_naive().strftime("%Y-%m-%d %H:%M:%S")),
        )
        cal = _load_calendar(conn)

    df = df.merge(sales_df, on="product_id", how="left").fillna({"units_sold_30d": 0, "order_count_30d": 0})

    past_days = [ref - timedelta(days=i) for i in range(LOOKBACK_DAYS)]
    future_days = [ref + timedelta(days=i) for i in range(1, max(forecast_days, 30) + 1)]

    # Multipliers only depend on (category, finish) — compute once per pair.
    cache: Dict[tuple, Dict[str, List[float]]] = {}
    for key in df[["category", "finish"]].drop_duplicates().itertuples(index=False):
        cache[(key.category, key.finish)] = {
            "past": _multiplier_series(cal, key.category, key.finish, past_days),
            "future": _multiplier_series(cal, key.category, key.finish, future_days),
        }

    rows = []
    for r in df.itertuples(index=False):
        m = cache[(r.category, r.finish)]
        normal_rate = r.units_sold_30d / sum(m["past"])            # season-neutral cans/day
        demand = [normal_rate * x for x in m["future"]]
        f7, f30 = sum(demand[:7]), sum(demand[:30])

        # Days until the shelf is empty, walking forward with the daily demand
        stock, days_left = float(r.current_stock), None
        for i, dmd in enumerate(demand):
            stock -= dmd
            if stock <= 0:
                days_left = i + (1 + stock / dmd if dmd else 1)
                break
        if days_left is None:
            days_left = len(demand) + (stock / max(demand[-1], 0.01))

        lead = int(r.lead_time_days or 7)
        moq = int(r.moq_units or 10)
        safety = math.ceil(max(demand[:SAFETY_DAYS] or [0]) * SAFETY_DAYS)
        net_needed = math.ceil(f30) + safety - (r.current_stock + r.on_order)

        if net_needed <= 0:
            qty = 0
            urgency = "ON_ORDER" if r.on_order > 0 and days_left <= lead else "SUFFICIENT_STOCK"
            when = "Already ordered" if urgency == "ON_ORDER" else "Not needed"
        else:
            qty = math.ceil(net_needed / moq) * moq
            if days_left <= lead:
                urgency, when = "CRITICAL_ORDER_NOW", "Order today"
            elif days_left <= lead + 4:
                n = max(1, int(days_left - lead))
                urgency, when = "ORDER_SOON", f"Order within {n} day{'s' if n != 1 else ''}"
            else:
                n = int(days_left - lead)
                urgency, when = "PLANNED", f"Order in about {n} days"

        rows.append({
            "daily_velocity": round(r.units_sold_30d / LOOKBACK_DAYS, 2),
            "normal_daily_rate": round(normal_rate, 3),
            "seasonal_multiplier": round(max(m["future"][:7]), 2),
            "effective_velocity": round(f7 / 7, 2),
            "forecast_7d": math.ceil(f7),
            "forecast_30d": math.ceil(f30),
            "days_to_stockout": round(days_left, 1),
            "safety_stock": safety,
            "suggested_reorder_qty": qty,
            "reorder_urgency": urgency,
            "suggested_order_date": when,
        })

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["estimated_order_cost"] = out["suggested_reorder_qty"] * out["cost_price"]
    out["_rank"] = out["reorder_urgency"].map(URGENCY_ORDER)
    return out.sort_values(["_rank", "days_to_stockout"]).drop(columns="_rank").reset_index(drop=True)


def evaluate_supplier_clearance_deal(
    product_id: int,
    offered_unit_cost: float,
    lot_quantity: int,
    shelf_life_months: int = 12,
    on_day: Optional[date] = None,
) -> Dict[str, Any]:
    """
    Should the shop buy a discounted lot? Balances discount, how long the lot
    (plus current stock) takes to sell, money tied up, and expiry risk.
    """
    ref = on_day or clock.today()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT p.*, i.current_stock FROM products p JOIN inventory i ON p.id = i.product_id WHERE p.id = ?;",
            (product_id,),
        )
        row = cur.fetchone()
        if not row:
            return {"error": "Product not found"}
        prod = dict(zip([d[0] for d in cur.description], row))
        units_30d = cur.execute(
            """
            SELECT COALESCE(SUM(quantity), 0) FROM sales
            WHERE product_id = ? AND DATE(sale_timestamp) > DATE(?, ?) AND DATE(sale_timestamp) <= DATE(?);
            """,
            (product_id, ref.strftime("%Y-%m-%d"), f"-{LOOKBACK_DAYS} days", ref.strftime("%Y-%m-%d")),
        ).fetchone()[0]
        cal = _load_calendar(conn)

    past = _multiplier_series(cal, prod["category"], prod["finish"], [ref - timedelta(days=i) for i in range(LOOKBACK_DAYS)])
    normal_rate = max(units_30d / sum(past), 0.05)  # cans per normal day

    regular_cost = prod["cost_price"]
    selling_price = prod["selling_price"]
    discount_pct = round((regular_cost - offered_unit_cost) / regular_cost * 100, 1)

    total_to_sell = prod["current_stock"] + lot_quantity
    months_to_sell = round(total_to_sell / normal_rate / 30.0, 1)

    total_investment = round(offered_unit_cost * lot_quantity, 2)
    expected_revenue = round(selling_price * lot_quantity, 2)
    gross_profit = round(expected_revenue - total_investment, 2)
    profit_margin_pct = round(gross_profit / expected_revenue * 100, 1) if expected_revenue else 0.0
    holding_cost = round(total_investment * 0.12 * (months_to_sell / 12.0), 2)   # 12% a year
    net_profit = round(gross_profit - holding_cost, 2)

    if offered_unit_cost >= regular_cost:
        recommendation, badge = "NO DISCOUNT", "warning"
        verdict = "This price is not below your usual supplier cost."
        action = "Ask for at least 10% off before considering it."
    elif months_to_sell > shelf_life_months:
        recommendation, badge = "SKIP — WILL EXPIRE FIRST", "critical"
        verdict = f"At today's pace it takes about {months_to_sell} months to sell, but it expires in {shelf_life_months}."
        action = "Don't buy this lot, or buy fewer cans."
    elif months_to_sell > shelf_life_months * 0.75:
        recommendation, badge = "BUY, BUT PLAN AN OFFER", "warning"
        verdict = f"It sells in about {months_to_sell} months — close to the {shelf_life_months}-month expiry."
        action = "Negotiate another 10% off and plan a customer offer to move it faster."
    else:
        recommendation, badge = "GOOD DEAL — BUY IT", "success"
        verdict = f"{discount_pct}% below usual cost, and it sells in about {months_to_sell} months — well before the {shelf_life_months}-month expiry."
        action = f"Buy {lot_quantity} cans. Expected profit after holding cost: ₹{net_profit:,.0f}."

    return {
        "sku": prod["sku"], "shade_name": prod["shade_name"], "brand": prod["brand"],
        "pack_size": f"{prod['pack_size_litres']:g}L", "regular_cost": regular_cost,
        "offered_cost": offered_unit_cost, "discount_pct": discount_pct, "lot_quantity": lot_quantity,
        "total_investment": total_investment, "gross_profit": gross_profit,
        "profit_margin_pct": profit_margin_pct, "months_to_sell": months_to_sell,
        "shelf_life_months": shelf_life_months, "holding_cost": holding_cost,
        "net_projected_profit": net_profit, "recommendation": recommendation,
        "verdict": verdict, "action": action, "badge": badge,
    }

"""
Smart Alerts (Layer 4)

Turns stock, expiry, sales and visitor data into a short list of alerts a shop
owner can act on: what's wrong, why it matters, and the one thing to do.

Levels: "act_now" (red), "soon" (amber), "look_into" (blue).
"""
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src import clock
from src.config import (
    CONVERSION_DROP_RATIO,
    CONVERSION_LOOKBACK_DAYS,
    DB_PATH,
    DEAD_STOCK_DAYS,
    DEAD_STOCK_MAX_SALES,
    DEAD_STOCK_MIN_UNITS,
    EXPIRY_WARNING_DAYS,
)
from src.database import get_daily_traffic, get_hourly_traffic
from src.formatting import rupees
from src.predictor import compute_sales_forecast


def _fmt_pack(litres: float) -> str:
    return f"{litres:g} L"


_rupees = rupees


def _hour_label(h: int) -> str:
    suffix = "AM" if h < 12 else "PM"
    return f"{(h % 12) or 12} {suffix}"


def _low_stock_alerts(ref: date) -> List[Dict[str, Any]]:
    fc = compute_sales_forecast(30, on_day=ref)
    need = fc[fc["reorder_urgency"].isin(["CRITICAL_ORDER_NOW", "ORDER_SOON"])]
    alerts = []
    for r in need.itertuples(index=False):
        days_left = max(0.0, r.days_to_stockout)
        rate = r.effective_velocity
        critical = r.reorder_urgency == "CRITICAL_ORDER_NOW"
        if days_left < 1:
            runs_out = "Will run out today."
        elif days_left < 2:
            runs_out = "Will run out tomorrow."
        else:
            runs_out = f"Runs out in about {days_left:.0f} days."
        why = (
            f"Sells about {rate:.1f} a day; {r.supplier_name} needs {int(r.lead_time_days)} days to deliver. "
            + runs_out
        )
        if r.on_order:
            why += f" {int(r.on_order)} already on order."
        alerts.append({
            "type": "LOW_STOCK",
            "group": "Running out",
            "level": "act_now" if critical else "soon",
            "severity": "CRITICAL" if critical else "WARNING",
            "sku": r.sku,
            "product_id": int(r.product_id),
            "hex_code": r.hex_code,
            "title": f"{r.shade_name} {_fmt_pack(r.pack_size_litres)} — {int(r.current_stock)} can{'s' if r.current_stock != 1 else ''} left",
            "description": why,
            "action": f"Order {int(r.suggested_reorder_qty)} cans",
            "detail": {"supplier": r.supplier_name, "qty": int(r.suggested_reorder_qty), "when": r.suggested_order_date},
        })
    return alerts


def _expiry_alerts(conn, ref: date) -> List[Dict[str, Any]]:
    df = pd.read_sql_query(
        """
        SELECT p.id as product_id, p.sku, p.brand, p.shade_name, p.hex_code, p.pack_size_litres,
               p.cost_price, p.selling_price, i.batch_no, i.current_stock, i.expiry_date,
               (i.current_stock * p.cost_price) as capital_at_risk
        FROM inventory i JOIN products p ON i.product_id = p.id
        WHERE i.current_stock > 0 AND DATE(i.expiry_date) <= DATE(?, ?)
        ORDER BY DATE(i.expiry_date);
        """,
        conn,
        params=(ref.strftime("%Y-%m-%d"), f"+{EXPIRY_WARNING_DAYS} days"),
    )
    alerts = []
    for r in df.itertuples(index=False):
        days_left = (pd.to_datetime(r.expiry_date).date() - ref).days
        name = f"{r.shade_name} {_fmt_pack(r.pack_size_litres)}"
        if days_left < 0:
            alerts.append({
                "type": "EXPIRED_BATCH", "group": "Expiring", "level": "act_now", "severity": "CRITICAL",
                "sku": r.sku, "product_id": int(r.product_id), "hex_code": r.hex_code, "batch_no": r.batch_no,
                "days_left": days_left,
                "title": f"{name} — {r.current_stock} cans expired {abs(days_left)} days ago",
                "description": f"Batch {r.batch_no}. {_rupees(r.capital_at_risk)} of stock. Take it off the shelf and ask the supplier about returns.",
                "action": "Remove from shelf",
            })
            continue
        discount = 25 if days_left <= 20 else 15
        offer = round(r.selling_price * (1 - discount / 100.0))
        alerts.append({
            "type": "EXPIRING_BATCH", "group": "Expiring",
            "level": "act_now" if days_left <= 20 else "soon",
            "severity": "CRITICAL" if days_left <= 20 else "WARNING",
            "sku": r.sku, "product_id": int(r.product_id), "hex_code": r.hex_code, "batch_no": r.batch_no,
            "days_left": days_left,
            "title": f"{name} — {r.current_stock} cans expire in {days_left} days",
            "description": (
                f"Batch {r.batch_no}, {_rupees(r.capital_at_risk)} at risk. "
                f"A {discount}% offer ({_rupees(offer)} instead of {_rupees(r.selling_price)}) usually clears it in time."
            ),
            "action": f"Start {discount}% offer",
            "detail": {"discount_pct": discount, "offer_price": offer},
        })
    return alerts


def _dead_stock_alerts(conn, ref: date) -> List[Dict[str, Any]]:
    df = pd.read_sql_query(
        """
        SELECT p.id as product_id, p.sku, p.brand, p.shade_name, p.hex_code, p.category,
               p.pack_size_litres, p.cost_price, i.current_stock,
               (i.current_stock * p.cost_price) as tied_up_capital,
               COALESCE(SUM(s.quantity), 0) as recent_sales
        FROM products p
        JOIN inventory i ON p.id = i.product_id
        LEFT JOIN sales s ON p.id = s.product_id
             AND DATE(s.sale_timestamp) > DATE(?, ?) AND DATE(s.sale_timestamp) <= DATE(?)
             AND s.sale_timestamp <= ?
        WHERE i.current_stock >= ?
        GROUP BY p.id
        HAVING recent_sales <= ?
        ORDER BY tied_up_capital DESC;
        """,
        conn,
        params=(ref.strftime("%Y-%m-%d"), f"-{DEAD_STOCK_DAYS} days", ref.strftime("%Y-%m-%d"),
                clock.now_naive().strftime("%Y-%m-%d %H:%M:%S"), DEAD_STOCK_MIN_UNITS, DEAD_STOCK_MAX_SALES),
    )
    alerts = []
    for r in df.itertuples(index=False):
        bundle = any(k in r.category for k in ("Enamel", "Exterior")) or r.pack_size_litres >= 10
        alerts.append({
            "type": "DEAD_STOCK", "group": "Money stuck", "level": "soon", "severity": "WARNING",
            "sku": r.sku, "product_id": int(r.product_id), "hex_code": r.hex_code,
            "title": f"{r.shade_name} {_fmt_pack(r.pack_size_litres)} — {_rupees(r.tied_up_capital)} sitting on the shelf",
            "description": (
                f"{r.current_stock} cans, "
                + (f"none sold in {DEAD_STOCK_DAYS} days." if r.recent_sales == 0 else f"only {r.recent_sales} sold in {DEAD_STOCK_DAYS} days.")
            ),
            "action": "Bundle with free primer" if bundle else "Ask supplier to exchange",
        })
    return alerts


def _busiest_run(weak_hours: pd.DataFrame) -> List[int]:
    """Longest run of back-to-back weak hours (ties go to the one with more visitors)."""
    hours = weak_hours.set_index("hour")["visitors"].to_dict()
    runs, current = [], []
    for h in sorted(hours):
        if current and h == current[-1] + 1:
            current.append(h)
        else:
            if current:
                runs.append(current)
            current = [h]
    if current:
        runs.append(current)
    if not runs:
        return []
    return max(runs, key=lambda r: (len(r), sum(hours[h] for h in r)))


def _visitor_alerts(ref: date) -> List[Dict[str, Any]]:
    daily = get_daily_traffic(CONVERSION_LOOKBACK_DAYS, on_day=ref)
    if len(daily) < 5:
        return []
    alerts = []
    recent = daily.tail(7)
    median_visitors = daily["visitors"].median()
    for r in recent.itertuples(index=False):
        others = daily[daily["day"] != r.day]
        normal = others["bills"].sum() / max(others["visitors"].sum(), 1)
        conv = r.bills / r.visitors if r.visitors else 0
        if r.visitors < median_visitors or conv >= normal * CONVERSION_DROP_RATIO:
            continue
        day = pd.to_datetime(r.day).date()
        hourly = get_hourly_traffic(on_day=day)
        weak = hourly[(hourly["visitors"] >= 5) & (hourly["conversion_pct"] < normal * 100 * 0.5)]
        when = clock.fmt_day(day)
        run = _busiest_run(weak)
        if run:
            when += f", {_hour_label(run[0])}–{_hour_label(run[-1] + 1)}"
        alerts.append({
            "type": "CONVERSION_DROP", "group": "Visitors vs bills", "level": "look_into", "severity": "INFO",
            "day": r.day,
            "title": f"Many visitors, few bills on {when}",
            "description": (
                f"{int(r.visitors)} visitors, {int(r.bills)} bills ({conv * 100:.0f}% bought vs your usual {normal * 100:.0f}%). "
                "Was a popular item out of stock, or was the billing queue long?"
            ),
            "action": "See that day",
        })
    return alerts


def _camera_alerts(ref: date) -> List[Dict[str, Any]]:
    """Shelf refills, walk-outs and shelf-vs-till gaps from the shop camera."""
    from src import shop_insights as si

    try:
        source = si.best_source(ref)
        po = si.preorder(source, ref)
        th = si.theft(source, ref, days=3)
        fm = si.fast_movers(source, 7, day=ref)
    except Exception:  # noqa: BLE001 - no camera tables yet
        return []
    alerts = []
    for r in po["refill"]:
        alerts.append({
            "type": "REFILL_SHELF", "group": "Shelves", "level": r["level"],
            "severity": "WARNING", "sku": f"{source}:{r['slot_id']}", "product_id": int(r["product_id"] or 0),
            "hex_code": r.get("hex_code"), "source": source, "slot_id": r["slot_id"], "move": int(r["move"]),
            "title": f"{r['slot_name']} nearly empty — {r['label']}",
            "description": r["message"] + " An empty shelf loses the customer who doesn't ask.",
            "action": f"Refilled ({r['move']} cans)" if r["move"] else "Order more",
        })
    for e in th["walkouts"].to_dict("records") if not th["walkouts"].empty else []:
        if e["status"] != "NEW":
            continue
        when = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
        alerts.append({
            "type": "UNPAID_EXIT", "group": "Theft", "level": "act_now", "severity": "CRITICAL",
            "sku": f"ev{e['id']}", "event_id": int(e["id"]),
            "title": f"Possible unpaid can — {clock.fmt_day(when.date())}, {clock.fmt_time(when)}",
            "description": e["details"].split(" Photo:")[0] + " Check the photo and the bills from that time.",
            "action": "Checked",
        })
    for g in th["shelf_vs_till"]:
        alerts.append({
            "type": "SHELF_TILL_GAP", "group": "Theft", "level": "soon", "severity": "WARNING",
            "sku": f"gap{g['product_id']}:{ref.isoformat()}", "product_id": g["product_id"],
            "title": f"{g['label']}: {g['missing']} can{'s' if g['missing'] != 1 else ''} left the shelf without a bill today",
            "description": (f"The camera saw {g['taken']} taken off the shelf (not put back); the till sold {g['sold']}. "
                            f"About {rupees(g['value'])} unaccounted — a missed bill, a can moved to the store room, or theft."),
            "action": "Checked",
        })
    for h in fm["hesitant"][:2]:
        alerts.append({
            "type": "PICKED_NOT_BOUGHT", "group": "Shelves", "level": "look_into", "severity": "INFO",
            "sku": f"hes{h['product_id']}:{ref.isoformat()}", "product_id": h["product_id"],
            "title": f"{h['label']} — picked up often, rarely bought",
            "description": h["message"],
            "action": "Noted",
        })
    return alerts


def generate_smart_alerts(on_day: Optional[date] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns alerts grouped as:
    - critical:   products running out before the supplier can deliver
    - expiring:   batches expiring within EXPIRY_WARNING_DAYS (or already expired)
    - dead_stock: money stuck in slow sellers
    - anomalies:  days with many visitors but few bills
    - camera:     shelves to refill, walk-outs with unpaid cans, shelf-vs-till gaps
    """
    from src.safety import guard

    ref = on_day or clock.today()

    def with_conn(fn):
        with sqlite3.connect(DB_PATH, timeout=30) as conn:
            return fn(conn, ref)

    # each group on its own: one failing group must not hide the others (or crash the home page)
    return {
        "critical": guard("Alerts · running out", _low_stock_alerts, ref, show=False, default=[]),
        "expiring": guard("Alerts · expiring", with_conn, _expiry_alerts, show=False, default=[]),
        "dead_stock": guard("Alerts · money stuck", with_conn, _dead_stock_alerts, show=False, default=[]),
        "anomalies": guard("Alerts · visitors vs bills", _visitor_alerts, ref, show=False, default=[]),
        "camera": guard("Alerts · camera", _camera_alerts, ref, show=False, default=[]),
    }


def flatten_alerts(alerts: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """All alerts in one list, most urgent first."""
    order = {"act_now": 0, "soon": 1, "look_into": 2}
    items = [a for group in ("critical", "camera", "expiring", "dead_stock", "anomalies") for a in alerts.get(group, [])]
    return sorted(items, key=lambda a: order.get(a.get("level"), 3))


def alert_id(alert: Dict[str, Any]) -> str:
    """Stable id so 'Mark done' survives a page refresh."""
    return f"{alert['type']}:{alert.get('sku') or alert.get('day') or alert['title']}:{alert.get('batch_no', '')}"

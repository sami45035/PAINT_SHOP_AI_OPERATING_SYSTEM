"""
Shop insights from the camera (Layer 5 → business answers).

Everything here joins what the camera saw (visits, shelf picks, shelf counts,
door counts, security events) with the till (sales), the stock book
(inventory) and the forecast (predictor). Each function answers one of the
five questions the camera is for:

1. footfall()          How many people came in, when?
2. buyers()            How many of them actually bought? Where do we lose them?
3. theft()             Did anything leave without being paid for?
4. fast_movers()       What moves fastest — and what do people pick up but not buy?
5. shelves(), preorder()  Which shelf needs refilling now, what must be ordered?

`source` is 'camera' (a real camera), 'demo' (the drawn demo shop) or
'sample' (seeded history). Nothing after "now" is counted.
"""
from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from src import clock
from src.config import DB_PATH


def _now() -> str:
    return clock.now_naive().strftime("%Y-%m-%d %H:%M:%S")


def _ds(d: Optional[date]) -> str:
    return (d or clock.today()).strftime("%Y-%m-%d")


def _q(sql: str, params=()) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn, params=params)


def _hour(h: int) -> str:
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def product_label(row) -> str:
    return f"{row['shade_name']} {float(row['pack_size_litres']):g} L"


def best_source(day: Optional[date] = None) -> str:
    """Real camera data if there is any today, otherwise the sample history."""
    with sqlite3.connect(DB_PATH) as conn:
        ds = _ds(day)
        real = conn.execute("SELECT 1 FROM visits WHERE source = 'camera' AND DATE(entered_at) = ? "
                            "UNION SELECT 1 FROM footfall_logs WHERE source = 'camera' AND DATE(timestamp) = ? LIMIT 1;",
                            (ds, ds)).fetchone()
    return "camera" if real else "sample"


# ----------------------------------------------------------------------------- 1. footfall

def footfall(source: str, day: Optional[date] = None) -> Dict[str, Any]:
    ds, now = _ds(day), _now()
    hourly = _q("SELECT CAST(strftime('%H', timestamp) AS INTEGER) AS hour, SUM(count_in) AS came_in, "
                "SUM(count_out) AS left_shop FROM footfall_logs WHERE source = ? AND DATE(timestamp) = ? "
                "AND timestamp <= ? GROUP BY hour ORDER BY hour;", (source, ds, now))
    visits = _q("SELECT dwell_s FROM visits WHERE source = ? AND DATE(entered_at) = ? AND left_at <= ?;",
                (source, ds, now))
    came_in = int(hourly["came_in"].sum()) if not hourly.empty else 0
    left = int(hourly["left_shop"].sum()) if not hourly.empty else 0
    peak = None
    if not hourly.empty and came_in:
        r = hourly.loc[hourly["came_in"].idxmax()]
        peak = {"hour": int(r["hour"]), "label": f"{_hour(int(r['hour']))}–{_hour(int(r['hour']) + 1)}",
                "visitors": int(r["came_in"])}
    # same time last week, for comparison
    last_week = (datetime.strptime(ds, "%Y-%m-%d") - timedelta(days=7))
    lw_until = last_week.strftime("%Y-%m-%d") + " " + clock.now_naive().strftime("%H:%M:%S")
    lw = _q("SELECT COALESCE(SUM(count_in),0) AS n FROM footfall_logs WHERE source = ? AND DATE(timestamp) = ? "
            "AND timestamp <= ?;", (source, last_week.strftime("%Y-%m-%d"), lw_until))
    lw_n = int(lw["n"].iloc[0]) if not lw.empty else 0
    return {
        "came_in": came_in, "left": left, "inside": max(0, came_in - left),
        "avg_stay_min": round(float(visits["dwell_s"].mean()) / 60, 1) if not visits.empty else None,
        "peak": peak, "hourly": hourly,
        "vs_last_week_pct": (round((came_in - lw_n) / lw_n * 100) if lw_n else None),
    }


def daily_footfall(source: str, days: int = 14, day: Optional[date] = None) -> pd.DataFrame:
    end = datetime.strptime(_ds(day), "%Y-%m-%d").date()
    start = (end - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    now = _now()
    v = _q("SELECT DATE(timestamp) AS day, SUM(count_in) AS visitors FROM footfall_logs WHERE source = ? "
           "AND DATE(timestamp) BETWEEN ? AND ? AND timestamp <= ? GROUP BY day;", (source, start, end.isoformat(), now))
    b = _q("SELECT DATE(sale_timestamp) AS day, COUNT(DISTINCT invoice_no) AS bills FROM sales "
           "WHERE DATE(sale_timestamp) BETWEEN ? AND ? AND sale_timestamp <= ? GROUP BY day;", (start, end.isoformat(), now))
    out = v.merge(b, on="day", how="left").fillna({"bills": 0})
    out["bought_pct"] = (out["bills"] / out["visitors"].where(out["visitors"] > 0) * 100).round(1)
    return out


# ----------------------------------------------------------------------------- 2. buyers

def buyers(source: str, day: Optional[date] = None) -> Dict[str, Any]:
    """
    The visitor funnel: came in → stood at a shelf → reached the counter → bill.
    Bills come from the till; for the demo shop (no real till) the counter is
    the best stand-in and `bills` is None.
    """
    ds, now = _ds(day), _now()
    foot = footfall(source, day)
    v = _q("SELECT CAST(strftime('%H', entered_at) AS INTEGER) AS hour, reached_counter, shelves, picked, put_back, "
           "outcome FROM visits WHERE source = ? AND DATE(entered_at) = ? AND left_at <= ?;", (source, ds, now))
    sales = _q("SELECT invoice_no, CAST(strftime('%H', MIN(sale_timestamp)) AS INTEGER) AS hour, SUM(total_amount) AS amount "
               "FROM sales WHERE DATE(sale_timestamp) = ? AND sale_timestamp <= ? GROUP BY invoice_no;", (ds, now))
    real_till = source != "demo"
    visitors = foot["came_in"]
    finished = len(v)
    at_shelves = int((v["shelves"].fillna("") != "").sum()) if finished else 0
    at_counter = int(v["reached_counter"].sum()) if finished else 0
    bills = int(len(sales)) if real_till else None
    walkouts = int((v["outcome"] == "unpaid_exit").sum()) if finished else 0
    left_without_counter = finished - at_counter
    avg_bill = float(sales["amount"].mean()) if real_till and len(sales) else None

    hourly = foot["hourly"][["hour", "came_in"]].rename(columns={"came_in": "visitors"}) if not foot["hourly"].empty \
        else pd.DataFrame(columns=["hour", "visitors"])
    if finished:
        hourly = hourly.merge(v.groupby("hour")["reached_counter"].sum().rename("at_counter").reset_index(),
                              on="hour", how="outer")
    if real_till and len(sales):
        hourly = hourly.merge(sales.groupby("hour").size().rename("bills").reset_index(), on="hour", how="outer")
    for c in ("visitors", "at_counter", "bills"):
        if c not in hourly:
            hourly[c] = 0
    hourly = hourly.fillna(0).sort_values("hour").reset_index(drop=True)
    hourly["bought_pct"] = (hourly["bills"] / hourly["visitors"].where(hourly["visitors"] > 0) * 100).round(0)

    rate = (bills / visitors) if (real_till and visitors) else None
    # Hours where many came in but few bought (half the day's rate or worse)
    weak = []
    if rate:
        for r in hourly.itertuples(index=False):
            if r.visitors >= 4 and r.bills / r.visitors < rate * 0.5:
                weak.append({"hour": int(r.hour), "label": f"{_hour(int(r.hour))}–{_hour(int(r.hour) + 1)}",
                             "visitors": int(r.visitors), "bills": int(r.bills),
                             "missed_bills": int(round(r.visitors * rate - r.bills))})
    missed = sum(w["missed_bills"] for w in weak)
    return {
        "visitors": visitors, "finished_visits": finished, "at_shelves": at_shelves, "at_counter": at_counter,
        "bills": bills, "bought_pct": round(rate * 100, 1) if rate is not None else None,
        "left_without_counter": left_without_counter, "walkouts": walkouts, "avg_bill": avg_bill,
        "weak_hours": weak, "missed_bills": missed,
        "missed_value": round(missed * avg_bill) if (missed and avg_bill) else 0,
        "hourly": hourly, "real_till": real_till,
    }


# ----------------------------------------------------------------------------- shelves (picks, counts)

def _shelf_events(source: str, start: str, end: str) -> pd.DataFrame:
    return _q("SELECT e.*, p.shade_name, p.pack_size_litres, p.brand, p.selling_price FROM shelf_events e "
              "LEFT JOIN products p ON p.id = e.product_id WHERE e.source = ? AND e.at > ? AND e.at <= ?;",
              (source, start, end))


def shelves(source: str) -> pd.DataFrame:
    """
    Cans on each watched shelf right now: the last saved count plus every pick,
    put-back and refill since. Joined with the stock book (total cans in the
    shop) so we know what's left in the store room.
    """
    now = _now()
    st = _q("SELECT s.*, p.shade_name, p.pack_size_litres, p.brand, p.hex_code, p.selling_price "
            "FROM shelf_status s LEFT JOIN products p ON p.id = s.product_id WHERE s.source = ?;", (source,))
    if st.empty:
        return st
    ev = _q("SELECT slot_id, kind, qty, at FROM shelf_events WHERE source = ? AND at <= ?;", (source, now))
    inv = _q("SELECT product_id, SUM(current_stock) AS in_shop, SUM(on_order) AS on_order FROM inventory GROUP BY product_id;")
    rows = []
    for r in st.to_dict("records"):
        e = ev[(ev["slot_id"] == r["slot_id"]) & (ev["at"] > r["updated_at"])]
        change = int(e.loc[e["kind"].isin(["restock", "put_back"]), "qty"].sum() - e.loc[e["kind"] == "pick", "qty"].sum())
        cans = max(0, min(int(r["capacity"]), int(r["cans"]) + change))
        last = ev[ev["slot_id"] == r["slot_id"]]["at"].max() if not ev.empty else None
        stock = inv[inv["product_id"] == r["product_id"]]
        in_shop = int(stock["in_shop"].iloc[0]) if not stock.empty else 0
        on_order = int(stock["on_order"].iloc[0]) if not stock.empty else 0
        rows.append({**r, "cans": cans, "fill_pct": round(cans / max(1, int(r["capacity"])) * 100),
                     "label": product_label(r) if r.get("shade_name") else r["slot_name"],
                     "in_shop": in_shop, "store_room": max(0, in_shop - cans), "on_order": on_order,
                     "last_change": last if isinstance(last, str) else r["updated_at"]})
    return pd.DataFrame(rows).sort_values("slot_id").reset_index(drop=True)


def interest(source: str, days: int = 7, day: Optional[date] = None) -> pd.DataFrame:
    """
    Per watched product over the last `days`: cans picked up, put back, kept,
    sold at the till, and the same for the week before (for the trend).
    """
    end = datetime.strptime(_ds(day), "%Y-%m-%d") + timedelta(days=1)
    end_s = min(end.strftime("%Y-%m-%d %H:%M:%S"), _now())
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    prev = (end - timedelta(days=2 * days)).strftime("%Y-%m-%d %H:%M:%S")
    ev = _shelf_events(source, prev, end_s)
    slots = _q("SELECT slot_id, product_id FROM shelf_status WHERE source = ?;", (source,))
    if slots.empty:
        return pd.DataFrame()
    pids = [int(p) for p in slots["product_id"].dropna()]
    sold = _q(f"SELECT product_id, SUM(CASE WHEN sale_timestamp > ? THEN quantity ELSE 0 END) AS sold, "
              f"SUM(CASE WHEN sale_timestamp <= ? THEN quantity ELSE 0 END) AS sold_prev FROM sales "
              f"WHERE sale_timestamp > ? AND sale_timestamp <= ? AND product_id IN ({','.join('?' * len(pids)) or 'NULL'}) "
              f"GROUP BY product_id;", (start, start, prev, end_s, *pids))
    prods = _q(f"SELECT id AS product_id, shade_name, pack_size_litres, brand, hex_code, selling_price FROM products "
               f"WHERE id IN ({','.join('?' * len(pids)) or 'NULL'});", tuple(pids))
    rows = []
    for s in slots.to_dict("records"):
        e = ev[ev["slot_id"] == s["slot_id"]]
        cur, old = e[e["at"] > start], e[e["at"] <= start]
        p = prods[prods["product_id"] == s["product_id"]]
        if p.empty:
            continue
        p = p.iloc[0].to_dict()
        sd = sold[sold["product_id"] == s["product_id"]]
        picks = int(cur.loc[cur["kind"] == "pick", "qty"].sum())
        back = int(cur.loc[cur["kind"] == "put_back", "qty"].sum())
        picks_prev = int(old.loc[old["kind"] == "pick", "qty"].sum())
        rows.append({
            "slot_id": s["slot_id"], "product_id": int(s["product_id"]), "label": product_label(p), "brand": p["brand"],
            "hex_code": p["hex_code"], "picked": picks, "put_back": back, "kept": max(0, picks - back),
            "sold": int(sd["sold"].iloc[0]) if not sd.empty else 0,
            "sold_prev": int(sd["sold_prev"].iloc[0]) if not sd.empty else 0,
            "picked_prev": picks_prev,
            "put_back_pct": round(back / picks * 100) if picks else 0,
            "pick_trend_pct": round((picks - picks_prev) / picks_prev * 100) if picks_prev >= 3 else None,
            "price": float(p["selling_price"]),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 4. fast movers

def fast_movers(source: str, days: int = 7, limit: int = 8, day: Optional[date] = None) -> Dict[str, Any]:
    """
    Fastest-moving products from the till, with the camera's view added for the
    products on watched shelves: how often they're picked up, and which ones
    people pick up and put back (interest that isn't turning into sales).
    """
    end = datetime.strptime(_ds(day), "%Y-%m-%d") + timedelta(days=1)
    end_s = min(end.strftime("%Y-%m-%d %H:%M:%S"), _now())
    start = (end - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
    prev = (end - timedelta(days=2 * days)).strftime("%Y-%m-%d %H:%M:%S")
    till = _q("SELECT p.id AS product_id, p.shade_name, p.pack_size_litres, p.brand, p.hex_code, "
              "SUM(CASE WHEN s.sale_timestamp > ? THEN s.quantity ELSE 0 END) AS units, "
              "SUM(CASE WHEN s.sale_timestamp <= ? THEN s.quantity ELSE 0 END) AS units_prev, "
              "SUM(CASE WHEN s.sale_timestamp > ? THEN s.total_amount ELSE 0 END) AS revenue "
              "FROM sales s JOIN products p ON p.id = s.product_id WHERE s.sale_timestamp > ? AND s.sale_timestamp <= ? "
              "GROUP BY p.id ORDER BY units DESC;", (start, start, start, prev, end_s))
    till["label"] = till.apply(product_label, axis=1)
    till["per_day"] = (till["units"] / days).round(1)
    till["trend_pct"] = ((till["units"] - till["units_prev"]) / till["units_prev"].where(till["units_prev"] >= 3) * 100).round(0)
    cam = interest(source, days, day)
    top = till.head(limit).copy()
    if not cam.empty:
        top = top.merge(cam[["product_id", "picked", "put_back", "put_back_pct"]], on="product_id", how="left")
    else:
        top["picked"] = top["put_back"] = top["put_back_pct"] = None
    hesitant = []
    if not cam.empty:
        for r in cam.sort_values("put_back_pct", ascending=False).to_dict("records"):
            if r["picked"] >= 6 and r["put_back_pct"] >= 50:
                hesitant.append({**r, "message": (
                    f"Picked up {r['picked']} times this week, put back {r['put_back']} times "
                    f"({r['put_back_pct']}%). People like it but don't buy — check the price against similar shades, "
                    f"keep a small tester tin open, or suggest the 1 L size.")})
    rising = []
    if not cam.empty:
        for r in cam.to_dict("records"):
            if r["pick_trend_pct"] is not None and r["pick_trend_pct"] >= 25:
                rising.append({**r, "message": f"Picked up {r['pick_trend_pct']}% more than the week before "
                                               f"({r['picked']} vs {r['picked_prev']}) — demand is building."})
    return {"top": top, "hesitant": hesitant, "rising": rising, "watched": cam}


# ----------------------------------------------------------------------------- 5. shelves → refill & pre-order

def preorder(source: str, day: Optional[date] = None) -> Dict[str, Any]:
    """
    What to do about stock, from the camera's shelf counts + the stock book +
    the sales forecast:
    - refill: shelf low, cans waiting in the store room → move them now
    - order:  the forecast says it runs out before the supplier can deliver;
              for watched products the forecast is nudged by the camera's
              pick-up trend (more people picking it up this week = more demand coming)
    """
    from src.predictor import compute_sales_forecast

    sh = shelves(source)
    cam = interest(source, 7, day)
    fc = compute_sales_forecast(30, on_day=day)
    refill, order = [], []
    if not sh.empty:
        for r in sh.to_dict("records"):
            cap = int(r["capacity"])
            if r["cans"] <= max(1, round(cap * 0.3)):
                move = min(cap - r["cans"], r["store_room"])
                if move > 0:
                    refill.append({**r, "move": int(move), "level": "act_now" if r["cans"] == 0 else "soon",
                                   "message": f"{r['slot_name']}: {r['cans']} of {cap} left on the shelf — "
                                              f"bring {int(move)} from the store room ({r['store_room']} there)."})
                else:
                    refill.append({**r, "move": 0, "level": "act_now",
                                   "message": f"{r['slot_name']}: {r['cans']} of {cap} on the shelf and none in the store room."})
    factors = {}
    if not cam.empty:
        for r in cam.to_dict("records"):
            if r["picked"] >= 5 and r["picked_prev"] >= 3:
                factors[r["product_id"]] = max(0.8, min(1.5, r["picked"] / r["picked_prev"]))
    watched = set(int(p) for p in sh["product_id"].dropna()) if not sh.empty else set()
    for r in fc.to_dict("records"):
        pid = int(r["product_id"])
        f = factors.get(pid, 1.0)
        rate = r["effective_velocity"] * f
        demand30 = r["forecast_30d"] * f
        have = r["current_stock"] + r["on_order"]
        days_left = r["current_stock"] / rate if rate > 0 else 999
        lead = int(r["lead_time_days"] or 7)
        need = math.ceil(demand30 + r["safety_stock"] - have)
        if need <= 0 or days_left > lead + 7:
            continue
        moq = int(r["moq_units"] or 1)
        qty = math.ceil(need / moq) * moq
        why = f"Sells about {rate:.1f}/day; {r['supplier_name']} takes {lead} days; {int(r['current_stock'])} in the shop"
        if r["on_order"]:
            why += f", {int(r['on_order'])} on order"
        if f > 1.05:
            why += f". The camera shows {round((f - 1) * 100)}% more pick-ups this week, so the order is sized up"
        elif f < 0.95:
            why += f". Fewer pick-ups this week ({round((1 - f) * 100)}% down), so the order is trimmed"
        order.append({
            "product_id": pid, "label": f"{r['shade_name']} {float(r['pack_size_litres']):g} L", "brand": r["brand"],
            "hex_code": r["hex_code"], "qty": int(qty), "cost": float(qty * r["cost_price"]),
            "days_left": round(days_left, 1), "lead": lead, "camera_factor": round(f, 2),
            "watched": pid in watched,
            "level": "act_now" if days_left <= lead else "soon",
            "when": "Order today" if days_left <= lead else (
                f"Order within {max(1, int(days_left - lead))} day{'s' if max(1, int(days_left - lead)) != 1 else ''}"),
            "message": why + ".",
        })
    order.sort(key=lambda o: (o["level"] != "act_now", not o["watched"], o["days_left"]))
    return {"refill": refill, "order": order}


# ----------------------------------------------------------------------------- 3. theft

def theft(source: str, day: Optional[date] = None, days: int = 7) -> Dict[str, Any]:
    """
    - Walk-outs: someone took a can and left without reaching the counter.
    - Shelf vs till: cans that left the watched shelves today (picked, not put
      back) compared with what the till sold of those products today. More gone
      than sold = cans that may have left unpaid (or were moved without a note).
    - After-hours movement and staff-only area entries.
    """
    ds, now = _ds(day), _now()
    since = (datetime.strptime(ds, "%Y-%m-%d") - timedelta(days=days - 1)).strftime("%Y-%m-%d")
    sec = _q("SELECT * FROM security_events WHERE COALESCE(source,'sample') = ? AND DATE(timestamp) >= ? "
             "AND timestamp <= ? ORDER BY timestamp DESC;", (source, since, now))
    start = ds + " 00:00:00"
    # A can picked in the last 20 minutes may still be on its way to the counter: picks in that window
    # that the till hasn't billed yet (in the same window) are given the benefit of the doubt.
    grace = (clock.now_naive() - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
    ev = _shelf_events(source, start, now)
    gap = []
    if not ev.empty:
        pids = sorted(set(int(p) for p in ev["product_id"].dropna()))
        sold = _q(f"SELECT product_id, SUM(quantity) AS sold, SUM(CASE WHEN sale_timestamp > ? THEN quantity ELSE 0 END) "
                  f"AS sold_recent FROM sales WHERE sale_timestamp > ? AND sale_timestamp <= ? "
                  f"AND product_id IN ({','.join('?' * len(pids))}) GROUP BY product_id;", (grace, start, now, *pids))
        for pid, g in ev.groupby("product_id"):
            net = lambda d: int(d.loc[d["kind"] == "pick", "qty"].sum() - d.loc[d["kind"] == "put_back", "qty"].sum())
            taken, taken_recent = net(g), net(g[g["at"] > grace])
            s_ = sold[sold["product_id"] == pid]
            n_sold = int(s_["sold"].iloc[0]) if not s_.empty else 0
            sold_recent = int(s_["sold_recent"].iloc[0]) if not s_.empty else 0
            if source == "demo":
                n_sold = sold_recent = 0       # no till in the demo shop
            pending = max(0, taken_recent - sold_recent)
            missing = taken - n_sold - pending
            if missing > 0:
                r = g.iloc[0]
                gap.append({"product_id": int(pid), "label": product_label(r), "taken": taken, "sold": n_sold,
                            "missing": int(missing), "value": float(missing * (r["selling_price"] or 0))})
    walkouts = sec[sec["event_type"] == "UNPAID_EXIT"] if not sec.empty else sec
    return {
        "events": sec,
        "walkouts": walkouts,
        "walkouts_today": int((walkouts["timestamp"].str[:10] == ds).sum()) if not walkouts.empty else 0,
        "open": int((sec["status"] == "NEW").sum()) if not sec.empty else 0,
        "after_hours": int((sec["event_type"] == "AFTER_HOURS_MOTION").sum()) if not sec.empty else 0,
        "staff_zone": int((sec["event_type"] == "RESTRICTED_ZONE_ENTRY").sum()) if not sec.empty else 0,
        "shelf_vs_till": sorted(gap, key=lambda g: -g["value"]),
        "unbilled_value": round(sum(g["value"] for g in gap)),
    }

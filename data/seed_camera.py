"""
Sample shop-camera history (source = 'sample'), built from the sample sales.

So the camera screens have 30 days to show on day one, every visit, shelf
pick and walk-out here is consistent with the sales table:
- each bill in an hour is one visitor who reached the counter; the rest of
  that hour's visitors browsed (a few asked something at the counter)
- buyers of a 1 L / 4 L pack that sits on a camera-watched shelf took those
  cans off the shelf; browsers sometimes pick a can up and put it back
- shelves are refilled each morning; two are left short so "refill" and
  "pre-order" have something real to find
- three walk-outs with an unpaid can (one recent, two already handled)

Run on its own to add the history to an existing database:
    python data/seed_camera.py
"""
import random
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.database import CAMERA_TABLES_SQL  # noqa: E402
from src.shop_vision import DEMO_SLOT_PRODUCTS, demo_layout  # noqa: E402

DAYS = 30
CAPACITY = 8                       # cans that fit side by side on one shelf slot
# How often a browser picks this shelf's can up (and puts it back)
BROWSE_INTEREST = {"A1": 0.08, "A2": 0.45, "A3": 0.03, "B1": 0.06, "B2": 0.05, "B3": 0.04}
REASON_MIX = {"price": 25, "not_in_stock": 20, "just_asking": 38, "come_back": 17}   # why browsers didn't buy
NEGLECTED = {"B1": 4, "B3": 6}     # slot -> refills skipped for the last N days (shelves running low now)


def seed_camera_history(conn: sqlite3.Connection, days: int = DAYS, seed: int = 7) -> int:
    rnd = random.Random(seed)
    cur = conn.cursor()
    cur.executescript(CAMERA_TABLES_SQL)
    cols = [r[1] for r in cur.execute("PRAGMA table_info(security_events);").fetchall()]
    if "source" not in cols:
        cur.execute("ALTER TABLE security_events ADD COLUMN source TEXT NOT NULL DEFAULT 'sample';")
    from src.billing import ensure_tables as billing_tables

    billing_tables(conn)                     # bills / didn't-buy tables and the visit columns they need
    for table in ("visits", "shelf_events", "shelf_status", "no_sales"):
        cur.execute(f"DELETE FROM {table} WHERE source = 'sample';")
    why_rnd = random.Random(seed + 1)        # separate stream: reasons don't change the rest of the story
    no_sales = []
    # shades customers asked for that the shop is short of (they turn up under "not in stock")
    low_stock = [r[0] for r in cur.execute(
        "SELECT p.id FROM products p JOIN inventory i ON i.product_id = p.id WHERE p.category != 'Waterproofing' "
        "GROUP BY p.id HAVING SUM(i.current_stock) <= MIN(i.min_threshold) ORDER BY p.id;").fetchall()][:5]
    cur.execute("DELETE FROM security_events WHERE event_type = 'UNPAID_EXIT' AND source = 'sample';")

    latest = cur.execute("SELECT MAX(DATE(sale_timestamp)) FROM sales;").fetchone()[0]
    if not latest:
        return 0
    today = datetime.strptime(latest, "%Y-%m-%d")
    start = today - timedelta(days=days - 1)

    # Shelf slots -> products (same shelves as the demo shop)
    pid = {}
    for sid, shade, size in DEMO_SLOT_PRODUCTS:
        row = cur.execute("SELECT id FROM products WHERE shade_name = ? AND pack_size_litres = ?;", (shade, size)).fetchone()
        if row:
            pid[sid] = int(row[0])
    slot_of = {p: s for s, p in pid.items()}
    names = {s["id"]: s["name"] for s in demo_layout()["shelves"]}

    level = {sid: CAPACITY for sid in pid}
    first = start.replace(hour=9)
    for sid in pid:
        cur.execute("INSERT INTO shelf_status (source, slot_id, slot_name, product_id, cans, capacity, updated_at) "
                    "VALUES ('sample', ?, ?, ?, ?, ?, ?);",
                    (sid, names.get(sid, sid), pid[sid], CAPACITY, CAPACITY, first.strftime("%Y-%m-%d %H:%M:%S")))

    events, visits = [], []

    def ev(at, sid, kind, qty):
        events.append(("sample", at.strftime("%Y-%m-%d %H:%M:%S"), sid, pid[sid], kind, qty))

    # Walk-outs: (days ago, hour, slot, cans, status)
    walkouts = {(0, 10): ("B2", 1, "NEW"), (2, 18): ("B3", 1, "SEEN"), (9, 12): ("B1", 2, "FALSE_ALARM"),
                (17, 19): ("A2", 1, "RESOLVED")}

    day = start
    while day <= today:
        ago = (today - day).days
        recent_boost = 3.0 if ago < 7 else 1.0          # more people picking up Royale Blue this week
        # morning refill
        for sid in pid:
            skip = sid in NEGLECTED and NEGLECTED[sid] > 0 and ago < NEGLECTED[sid]
            if not skip and level[sid] < CAPACITY:
                ev(day.replace(hour=9, minute=10), sid, "restock", CAPACITY - level[sid])
                level[sid] = CAPACITY
        ds = day.strftime("%Y-%m-%d")
        foot = cur.execute("SELECT timestamp, count_in, dwell_time_mins FROM footfall_logs "
                           "WHERE DATE(timestamp) = ? AND COALESCE(source,'sample') = 'sample' ORDER BY timestamp;",
                           (ds,)).fetchall()
        for ts, visitors, dwell in foot:
            hour_start = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            hr = hour_start.hour
            if hr == 15:      # afternoon top-up of nearly empty shelves (not B1 today)
                for sid in pid:
                    if level[sid] <= 2 and not (sid in NEGLECTED and ago < NEGLECTED[sid]):
                        ev(hour_start.replace(minute=5), sid, "restock", CAPACITY - level[sid])
                        level[sid] = CAPACITY
            bills = cur.execute(
                "SELECT invoice_no, MIN(sale_timestamp) FROM sales WHERE sale_timestamp >= ? AND sale_timestamp < ? "
                "GROUP BY invoice_no;", (ts, (hour_start + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
            lines = defaultdict(list)
            for inv, p, q, size in cur.execute(
                    "SELECT s.invoice_no, s.product_id, s.quantity, p.pack_size_litres FROM sales s JOIN products p "
                    "ON p.id = s.product_id WHERE s.sale_timestamp >= ? AND s.sale_timestamp < ?;",
                    (ts, (hour_start + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"))).fetchall():
                lines[inv].append((p, q, size))
            wo = walkouts.get((ago, hr))
            n_visitors = max(int(visitors), len(bills) + (1 if wo else 0))
            buyers = sorted(bills, key=lambda b: b[1])
            for k in range(n_visitors):
                if k < len(buyers):
                    inv, paid_at = buyers[k]
                    paid = datetime.strptime(paid_at, "%Y-%m-%d %H:%M:%S")
                    stay = max(4.0, rnd.gauss(dwell or 14, 4))
                    entered = paid - timedelta(minutes=stay * 0.75)
                    left = paid + timedelta(minutes=stay * 0.25)
                    shelves, picked, back = set(), 0, 0
                    for p, q, size in lines[inv]:
                        sid = slot_of.get(p)
                        if sid and size <= 4:
                            shelves.add(sid)
                            if level[sid] < q and not (sid in NEGLECTED and ago < NEGLECTED[sid]):
                                # staff top the shelf up from the store room when it runs out (the camera sees it)
                                ev(entered + timedelta(minutes=stay * 0.3), sid, "restock", CAPACITY - level[sid])
                                level[sid] = CAPACITY
                            take = min(q, level[sid])
                            if take:
                                ev(entered + timedelta(minutes=stay * 0.4), sid, "pick", take)
                                level[sid] -= take
                                picked += take
                    if not shelves and rnd.random() < 0.4:
                        shelves.add(rnd.choice(list(pid)))
                    visits.append(("sample", entered, left, True, shelves, picked, back, "reached_counter",
                                   {"bill_no": inv, "counter_from": paid - timedelta(minutes=2), "counter_to": paid + timedelta(seconds=40)}))
                    continue
                entered = hour_start + timedelta(minutes=rnd.uniform(0, 50))
                stay = max(2.0, rnd.gauss((dwell or 14) * 0.6, 3))
                left = entered + timedelta(minutes=stay)
                if wo and k == len(buyers):
                    sid, n, _ = wo
                    take = min(n, level[sid])
                    ev(entered + timedelta(minutes=stay * 0.5), sid, "pick", take)
                    level[sid] -= take
                    visits.append(("sample", entered, left, False, {sid}, take, 0, "unpaid_exit", {}))
                    continue
                shelves, back = set(), 0
                if rnd.random() < 0.7:
                    for sid in rnd.sample(list(pid), k=rnd.choice([1, 1, 2])):
                        shelves.add(sid)
                        p_pick = BROWSE_INTEREST.get(sid, 0.1) * (recent_boost if sid == "B2" else 1.0)
                        if level[sid] > 0 and rnd.random() < p_pick:
                            at = entered + timedelta(minutes=stay * 0.4)
                            ev(at, sid, "pick", 1)
                            ev(at + timedelta(minutes=1), sid, "put_back", 1)
                            back += 1
                counter = rnd.random() < 0.15        # asked something at the counter, didn't buy
                extra = {}
                if counter:
                    mid = entered + (left - entered) / 2
                    extra = {"counter_from": mid - timedelta(minutes=1), "counter_to": mid + timedelta(minutes=1)}
                if why_rnd.random() < 0.45:                  # staff tapped "Didn't buy" with a reason
                    reason = why_rnd.choices(list(REASON_MIX), weights=list(REASON_MIX.values()))[0]
                    pid_wanted = (why_rnd.choice(low_stock) if reason == "not_in_stock" and low_stock
                                  and why_rnd.random() < 0.5 else None)    # staff don't always note the shade
                    at = left - timedelta(seconds=30)
                    no_sales.append(("sample", at.strftime("%Y-%m-%d %H:%M:%S"), reason, pid_wanted))
                    extra["no_sale_reason"] = reason
                visits.append(("sample", entered, left, counter, shelves, 0, back,
                               "reached_counter" if counter else "browsed", extra))
        day += timedelta(days=1)

    cur.executemany("INSERT INTO shelf_events (source, at, slot_id, product_id, kind, qty) VALUES (?, ?, ?, ?, ?, ?);",
                    events)
    fmt = lambda d: d.strftime("%Y-%m-%d %H:%M:%S") if d else None
    cur.executemany(
        "INSERT INTO visits (source, entered_at, left_at, dwell_s, reached_counter, shelves, picked, put_back, outcome, "
        "counter_from, counter_to, bill_no, no_sale_reason) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
        [(src, fmt(a), fmt(b), (b - a).total_seconds(), int(c), ",".join(sorted(sh)), pk, bk, out,
          fmt(x.get("counter_from")), fmt(x.get("counter_to")), x.get("bill_no"), x.get("no_sale_reason"))
         for src, a, b, c, sh, pk, bk, out, x in visits])
    cur.executemany("INSERT INTO no_sales (at, source, reason, product_id) VALUES (?, ?, ?, ?);",
                    [(at, src, r, pid) for src, at, r, pid in no_sales])

    labels = {sid: f"{shade} {size:g} L" for sid, shade, size in DEMO_SLOT_PRODUCTS}
    for (ago, hr), (sid, n, status) in walkouts.items():
        at = (today - timedelta(days=ago)).replace(hour=hr, minute=37, second=12)
        cur.execute(
            "INSERT INTO security_events (timestamp, event_type, zone, severity, status, details, source) "
            "VALUES (?, 'UNPAID_EXIT', 'Front door', 'CRITICAL', ?, ?, 'sample');",
            (at.strftime("%Y-%m-%d %H:%M:%S"), status,
             f"Someone took {n} × {labels[sid]} ({names.get(sid, sid)}) and left at {at.strftime('%I:%M %p').lstrip('0')} "
             f"without going to the counter."))
    conn.commit()
    return len(visits)


if __name__ == "__main__":
    from src.config import DB_PATH

    with sqlite3.connect(DB_PATH) as c:
        n = seed_camera_history(c)
    print(f"Added {n} sample camera visits to {DB_PATH}")

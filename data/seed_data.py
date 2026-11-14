"""
Seed script to initialize and populate SQLite database with realistic paint shop data.
Creates suppliers, 100+ paint SKUs, batches, 180 days of sales transactions,
hourly footfall logs, seasonal events, and security events.
"""
import os
import sys
import json
import sqlite3
import random
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "data"))

from src.config import SAMPLE_DB_PATH as DB_PATH, SHADES_CATALOG_PATH
from src import clock

def init_db():
    if DB_PATH.exists():
        os.remove(DB_PATH)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. Suppliers Table
    cursor.execute("""
    CREATE TABLE suppliers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        lead_time_days INTEGER NOT NULL,
        moq_units INTEGER NOT NULL,
        contact_email TEXT,
        contact_phone TEXT,
        reliability_score REAL NOT NULL
    );
    """)

    # 2. Products Table
    cursor.execute("""
    CREATE TABLE products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        sku TEXT UNIQUE NOT NULL,
        brand TEXT NOT NULL,
        sub_brand TEXT,
        shade_name TEXT NOT NULL,
        shade_code TEXT NOT NULL,
        hex_code TEXT NOT NULL,
        category TEXT NOT NULL,
        finish TEXT NOT NULL,
        pack_size_litres REAL NOT NULL,
        cost_price REAL NOT NULL,
        selling_price REAL NOT NULL,
        supplier_id INTEGER REFERENCES suppliers(id),
        room_tags TEXT
    );
    """)

    # 3. Inventory Table
    cursor.execute("""
    CREATE TABLE inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL REFERENCES products(id),
        batch_no TEXT NOT NULL,
        current_stock INTEGER NOT NULL,
        on_order INTEGER DEFAULT 0,
        min_threshold INTEGER NOT NULL,
        max_threshold INTEGER NOT NULL,
        purchase_date TEXT NOT NULL,
        expiry_date TEXT NOT NULL,
        shelf_location TEXT
    );
    """)

    # 4. Sales Table
    cursor.execute("""
    CREATE TABLE sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        invoice_no TEXT NOT NULL,
        product_id INTEGER NOT NULL REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        total_amount REAL NOT NULL,
        sale_timestamp TEXT NOT NULL,
        customer_type TEXT NOT NULL
    );
    """)

    # 5. Footfall Logs Table
    cursor.execute("""
    CREATE TABLE footfall_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        count_in INTEGER NOT NULL,
        count_out INTEGER NOT NULL,
        occupancy INTEGER NOT NULL,
        dwell_time_mins REAL NOT NULL,
        source TEXT NOT NULL DEFAULT 'sample'
    );
    """)

    # 6. Security Events Table
    cursor.execute("""
    CREATE TABLE security_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        event_type TEXT NOT NULL,
        zone TEXT NOT NULL,
        severity TEXT NOT NULL,
        status TEXT NOT NULL,
        details TEXT,
        source TEXT NOT NULL DEFAULT 'sample'
    );
    """)

    # 7. Seasonal Calendar Table
    cursor.execute("""
    CREATE TABLE seasonal_calendar (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_name TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        impact_category TEXT NOT NULL,
        expected_multiplier REAL NOT NULL,
        description TEXT
    );
    """)

    conn.commit()
    return conn

def populate_database(conn):
    cursor = conn.cursor()
    random.seed(42)

    # 1. Populate Suppliers
    suppliers_data = [
        ("Metro Coatings Wholesale", 7, 10, "orders@metrocoatings.in", "+91-98201-11223", 0.94),
        ("Asian Regional Supply Hub", 5, 15, "dispatch@asiandepot.in", "+91-98202-33445", 0.98),
        ("Berger Logistics Hub", 12, 12, "logistics@bergerhub.in", "+91-98203-55667", 0.91),
        ("Nerolac North Distributors", 10, 8, "support@nerolacnorth.in", "+91-98204-77889", 0.89)
    ]
    cursor.executemany("""
    INSERT INTO suppliers (name, lead_time_days, moq_units, contact_email, contact_phone, reliability_score)
    VALUES (?, ?, ?, ?, ?, ?);
    """, suppliers_data)
    conn.commit()

    # 2. Populate Products from Catalog
    with open(SHADES_CATALOG_PATH, "r", encoding="utf-8") as f:
        shades = json.load(f)

    pack_sizes = [
        (1.0, 1.0, 1.0),      # (litres, cost_multiplier, price_multiplier)
        (4.0, 3.8, 3.85),
        (10.0, 9.2, 9.3),
        (20.0, 17.8, 18.0)
    ]

    brand_supplier_map = {
        "Asian Paints": 2,
        "Berger": 3,
        "Dulux": 1,
        "Nerolac": 4
    }

    products_inserted = []
    sku_counter = 1001

    for shade in shades:
        supp_id = brand_supplier_map.get(shade["brand"], 1)
        base_cost = shade["base_cost_1l"]
        base_price = shade["base_price_1l"]
        tags_str = ", ".join(shade.get("room_tags", []))

        # We create 1L, 4L, 10L, 20L for each shade
        for size_litres, cost_mult, price_mult in pack_sizes:
            sku = f"SKU-{shade['brand'][:2].upper()}-{sku_counter}"
            sku_counter += 1
            cost = round(base_cost * cost_mult, 2)
            price = round(base_price * price_mult, 2)

            cursor.execute("""
            INSERT INTO products (sku, brand, sub_brand, shade_name, shade_code, hex_code, category, finish, pack_size_litres, cost_price, selling_price, supplier_id, room_tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                sku, shade["brand"], shade["sub_brand"], shade["shade_name"],
                shade["shade_code"], shade["hex_code"], shade["category"],
                shade["finish"], size_litres, cost, price, supp_id, tags_str
            ))
            products_inserted.append({
                "id": cursor.lastrowid,
                "sku": sku,
                "shade_name": shade["shade_name"],
                "pack_size": size_litres,
                "cost": cost,
                "price": price,
                "category": shade["category"],
                "finish": shade["finish"],
            })
    conn.commit()

    # 3. Populate Inventory with Batches & Realistic Statuses
    # The sample data ends on the day it is generated (store time zone).
    # In demo mode the app keeps moving it forward so it always ends "today".
    today = datetime.combine(clock.today(), datetime.min.time())

    slow_ids = set()   # deliberately slow sellers ("money stuck" demo)
    for idx, prod in enumerate(products_inserted):
        # We craft specific corner cases for realistic prototype demonstrations:
        # Case A: Low Stock (reorder triggered)
        if idx in [2, 7, 14, 25, 40]:
            stock = random.randint(1, 3)
            min_thresh = 8
            max_thresh = 35
            on_order = 0
            expiry = today + timedelta(days=random.randint(180, 400))
        # Case B: Expiring Soon (< 45 days)
        elif idx in [5, 12, 19, 33]:
            stock = random.randint(8, 16)
            min_thresh = 5
            max_thresh = 30
            on_order = 0
            expiry = today + timedelta(days=random.randint(12, 38))
        # Case C: Overstock / Slow-Moving Dead Stock
        elif idx in [8, 22, 39, 60, 78]:
            slow_ids.add(prod["id"])
            stock = random.randint(35, 50)
            min_thresh = 6
            max_thresh = 20
            on_order = 0
            expiry = today + timedelta(days=random.randint(120, 300))
        # Case D: In-Transit on_order shipment
        elif idx in [4, 11, 28]:
            stock = random.randint(4, 7)
            min_thresh = 8
            max_thresh = 30
            on_order = 20
            expiry = today + timedelta(days=random.randint(200, 450))
        # Case E: Healthy Stock (fast sellers are stocked deeper)
        else:
            fast = any(k in prod["shade_name"] for k in ("White", "Beige", "Royale Blue")) or prod["category"] == "Primer"
            stock = random.randint(40, 70) if fast else random.randint(10, 28)
            min_thresh = 12 if fast else 6
            max_thresh = 90 if fast else 35
            on_order = 0
            expiry = today + timedelta(days=random.randint(180, 500))

        purchase_date = today - timedelta(days=random.randint(30, 240))
        batch_no = f"B26-{random.randint(100, 999)}"
        shelf = f"Aisle {random.choice(['A','B','C','D'])}-Shelf {random.randint(1, 5)}"

        cursor.execute("""
        INSERT INTO inventory (product_id, batch_no, current_stock, on_order, min_threshold, max_threshold, purchase_date, expiry_date, shelf_location)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            prod["id"], batch_no, stock, on_order, min_thresh, max_thresh,
            purchase_date.strftime("%Y-%m-%d"), expiry.strftime("%Y-%m-%d"), shelf
        ))
    conn.commit()

    # 4. Populate 180 Days of Historical Sales & Hourly Footfall
    start_date = today - timedelta(days=180)
    customer_types = ["Retail Walk-in", "Retail Walk-in", "Contractor", "Interior Designer"]

    inv_count = 10001
    
    # Pre-select popular products (e.g. Pure White, Royale Blue, Warm Beige, Primer)
    popular_ids = [p["id"] for p in products_inserted if "White" in p["shade_name"] or "Beige" in p["shade_name"] or "Royale Blue" in p["shade_name"] or "Primer" in p["category"]]
    regular_ids = [p["id"] for p in products_inserted if p["id"] not in popular_ids]

    # Generate hourly visitors first, then bills from those visitors, so that
    # "bills per visitor" stays realistic (about 35-45%) hour by hour.
    # Two recent evenings get a deliberate dip (busy store, few bills) so the
    # "Many visitors, few bills" alert has something real to find.
    dip_days = {(today - timedelta(days=1)).date(), (today - timedelta(days=4)).date()}

    # What festivals and seasons REALLY do to this shop's sales. The shop's calendar
    # (seasonal_calendar, below) only has rough guesses — e.g. it says Diwali is 2.2x,
    # in this shop it is nearer 1.8x for emulsions. The sales predictor learns the
    # real effect from these sales.
    from src.predictor import event_applies
    REAL = [  # (start, end, impact text as in the calendar, shop-wide visitors x, product x)
        ("2026-10-10", "2026-11-15", "Interior & Exterior Emulsions", 1.30, 1.45),
        ("2026-09-01", "2026-10-15", "Waterproofing & Exterior Primer", 1.10, 2.10),
        ("2026-11-20", "2027-01-15", "Luxury Sheen & Accent Shades", 1.10, 1.30),
        ("2026-06-15", "2026-08-15", "Exterior Emulsion", 0.92, 0.45),
    ]
    REAL = [(datetime.strptime(a, "%Y-%m-%d").date(), datetime.strptime(b, "%Y-%m-%d").date(), imp, v, pm)
            for a, b, imp, v, pm in REAL]
    base_w = {}
    for p in products_inserted:
        w = 0.65 / len(popular_ids) if p["id"] in popular_ids else 0.35 / max(1, len(regular_ids))
        base_w[p["id"]] = w * (0.03 if p["id"] in slow_ids else 1.0)
    by_id = {p["id"]: p for p in products_inserted}
    day_cache = {}

    def day_weights(d):
        if d not in day_cache:
            active = [(imp, pm) for a, b, imp, v, pm in REAL if a <= d <= b]
            ids = list(base_w)
            ws = []
            for pid in ids:
                pr = by_id[pid]
                m = 1.0
                for imp, pm in active:
                    if event_applies(pr["category"], pr["finish"], imp):
                        m = max(m, pm) if pm >= 1 else min(m, pm)
                if "Royale Blue" in pr["shade_name"] and (today.date() - d).days < 10:
                    m *= 2.0          # Royale Blue is suddenly in demand (a viral home-makeover reel)
                ws.append(base_w[pid] * m)
            day_cache[d] = (ids, ws)
        return day_cache[d]

    cur_date = start_date
    while cur_date <= today:
        date_str = cur_date.strftime("%Y-%m-%d")
        is_weekend = cur_date.weekday() >= 5
        season_mult = 1.0
        for a, b, imp, v, pm in REAL:
            if a <= cur_date.date() <= b:
                season_mult *= v

        for hour in range(9, 21):
            is_peak = (11 <= hour <= 13) or (17 <= hour <= 20)
            visitors = random.randint(3, 7) if is_peak else random.randint(1, 3)
            if is_weekend:
                visitors = int(round(visitors * 1.4))
            visitors = int(round(visitors * season_mult))
            buy_rate = random.uniform(0.32, 0.46)
            if cur_date.date() in dip_days and 17 <= hour <= 19:
                visitors = int(visitors * 1.8) + 3
                buy_rate = 0.12

            bills = sum(1 for _ in range(visitors) if random.random() < buy_rate)
            count_out = max(0, visitors - random.randint(0, 2))
            occupancy = max(0, visitors - count_out)
            dwell = round(random.uniform(8.5, 24.0), 1)
            cursor.execute("""
            INSERT INTO footfall_logs (timestamp, count_in, count_out, occupancy, dwell_time_mins)
            VALUES (?, ?, ?, ?, ?);
            """, (f"{date_str} {hour:02d}:00:00", visitors, count_out, occupancy, dwell))

            for _ in range(bills):
                ids, ws = day_weights(cur_date.date())
                prod_id = random.choices(ids, weights=ws, k=1)[0]
                p_info = next(p for p in products_inserted if p["id"] == prod_id)
                c_type = random.choice(customer_types)
                # Contractors buy higher volume (2 to 6 cans)
                qty = random.randint(2, 6) if c_type == "Contractor" else random.randint(1, 2)
                unit_price = p_info["price"]
                sale_time = f"{date_str} {hour:02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}"
                cursor.execute("""
                INSERT INTO sales (invoice_no, product_id, quantity, unit_price, total_amount, sale_timestamp, customer_type)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """, (f"INV-{inv_count}", prod_id, qty, unit_price, round(qty * unit_price, 2), sale_time, c_type))
                inv_count += 1

        cur_date += timedelta(days=1)
    conn.commit()

    # 5. Populate Seasonal Calendar (real calendar dates — update every year)
    seasonal_data = [
        ("Diwali Pre-Festival Surge", "2026-10-10", "2026-11-15", "Interior & Exterior Emulsions", 2.2, "Massive annual repainting tradition across households & commercial shops."),
        ("Post-Monsoon Waterproofing", "2026-09-01", "2026-10-15", "Waterproofing & Exterior Primer", 1.8, "Fixing moisture dampness, roof cracks, and exterior weathering."),
        ("Winter Wedding Season", "2026-11-20", "2027-01-15", "Luxury Sheen & Accent Shades", 1.5, "Home renovations, bridal room transformations, and festive makeovers."),
        ("Monsoon Exterior Lull", "2026-06-15", "2026-08-15", "Exterior Emulsion", 0.45, "Heavy monsoon rainfall delays exterior painting projects.")
    ]
    cursor.executemany("""
    INSERT INTO seasonal_calendar (event_name, start_date, end_date, impact_category, expected_multiplier, description)
    VALUES (?, ?, ?, ?, ?, ?);
    """, seasonal_data)
    conn.commit()

    # 6. Populate Realistic Security Events
    security_data = [
        (
            (today - timedelta(days=1, hours=2, minutes=17)).strftime("%Y-%m-%d 21:43:12"),
            "AFTER_HOURS_MOTION",
            "Main Showroom Floor",
            "WARNING",
            "VERIFIED",
            "Movement detected at 9:43 PM (Shop closed at 9:00 PM). Verified as night cleaning staff."
        ),
        (
            (today - timedelta(days=3, hours=5)).strftime("%Y-%m-%d 18:31:05"),
            "RESTRICTED_ZONE_ENTRY",
            "High-Value Tinting Storage",
            "CRITICAL",
            "RESOLVED",
            "Unauthorized entry into pigment formulation zone during busy footfall period."
        ),
        (
            (today - timedelta(days=5, hours=10)).strftime("%Y-%m-%d 02:15:40"),
            "CAMERA_DISCONNECTED",
            "Rear Gate Camera",
            "CRITICAL",
            "RESOLVED",
            "Stream heartbeat interrupted for 42 minutes. Reconnected after Wi-Fi router reboot."
        ),
        (
            (today - timedelta(days=8, hours=1)).strftime("%Y-%m-%d 22:10:00"),
            "AFTER_HOURS_MOTION",
            "Storage Backdoor",
            "CRITICAL",
            "VERIFIED",
            "Movement logged near stock loading dock after store closing."
        )
    ]
    cursor.executemany("""
    INSERT INTO security_events (timestamp, event_type, zone, severity, status, details)
    VALUES (?, ?, ?, ?, ?, ?);
    """, security_data)
    conn.commit()

    # 7. Sample shop-camera history (visits, shelf picks, walk-outs) matching the sales above
    from seed_camera import seed_camera_history
    n_visits = seed_camera_history(conn)

    # 8. Tables the app writes to, and a stamp so running apps know the sample data changed
    from src.database import CAMERA_TABLES_SQL
    conn.executescript(CAMERA_TABLES_SQL)
    conn.execute("CREATE TABLE IF NOT EXISTS alert_actions (alert_id TEXT PRIMARY KEY, action TEXT NOT NULL, "
                 "acted_at TEXT NOT NULL, hidden_until TEXT NOT NULL);")
    conn.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('seed_id', ?);",
                 (datetime.now().strftime("%Y%m%d%H%M%S"),))
    conn.commit()

    print(f"Successfully initialized and populated database at {DB_PATH}")
    print(f"Sample camera visits: {n_visits}")
    print(f"Total Products: {len(products_inserted)}")
    print(f"Total Sales Records: {cursor.execute('SELECT COUNT(*) FROM sales;').fetchone()[0]}")
    print(f"Total Footfall Logs: {cursor.execute('SELECT COUNT(*) FROM footfall_logs;').fetchone()[0]}")
    print(f"Total Security Events: {cursor.execute('SELECT COUNT(*) FROM security_events;').fetchone()[0]}")
    conn.close()

def build(path) -> None:
    """Write the sample database (dated so its last day is today) to `path`."""
    global DB_PATH
    saved, DB_PATH = DB_PATH, Path(path)
    try:
        populate_database(init_db())
    finally:
        DB_PATH = saved


if __name__ == "__main__":
    conn = init_db()
    populate_database(conn)

"""
Scan & Bill (Layer 1 + 5): billing at the counter, wired to stock and the camera.

- Each can carries a QR label (APX:<SKU>) — or its own printed barcode, linked to
  a product once. Scanning adds it to the bill.
- Saving a bill, in ONE database transaction:
    1. writes the sale lines (so sales, forecast and fast movers see it at once),
    2. takes the cans off stock, oldest expiry batch first (FIFO — less paint
       expires on the shelf), remembering which batch each can came from,
    3. records the bill (time, items, total, customer type, payment).
  Voiding a bill puts the cans back into the exact batches they came from.
- "Didn't buy" records a lost customer with a reason (price, shade not in stock,
  just asking, will come back) — and the shade they wanted, if any.
- Bills and didn't-buys are matched to camera visits by time (see match_visits):
  the customer standing at the counter when the bill is saved is the buyer;
  people who arrived together share the bill.
"""
from __future__ import annotations

import sqlite3
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src import clock
from src.config import DB_PATH, STORE_SHORT_NAME

CODE_PREFIX = "APX:"
REASONS = {
    "price": "Price too high",
    "not_in_stock": "Shade / pack not in stock",
    "just_asking": "Just asking / comparing",
    "come_back": "Will come back later",
    "other": "Other",
}
CUSTOMER_TYPES = ["Retail Walk-in", "Contractor", "Painter"]
PAYMENT_MODES = ["Cash", "UPI", "Card", "Credit"]

BILLING_SQL = """
CREATE TABLE IF NOT EXISTS bills (
    invoice_no TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,                 -- 'app' (Scan & Bill) | 'sample'
    customer_type TEXT NOT NULL,
    customer_phone TEXT,
    payment_mode TEXT,
    items INTEGER NOT NULL,
    total REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'paid'   -- paid | void
);
CREATE INDEX IF NOT EXISTS ix_bills_day ON bills (created_at);
CREATE TABLE IF NOT EXISTS bill_stock_moves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT NOT NULL,
    inventory_id INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    qty INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS no_sales (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    source TEXT NOT NULL,                 -- 'app' | 'sample'
    reason TEXT NOT NULL,
    product_id INTEGER,
    note TEXT
);
CREATE INDEX IF NOT EXISTS ix_no_sales_day ON no_sales (at);
CREATE TABLE IF NOT EXISTS product_codes (
    code TEXT PRIMARY KEY,                -- a barcode printed on the can, linked to a product once
    product_id INTEGER NOT NULL
);
"""

VISIT_COLUMNS = {"counter_from": "TEXT", "counter_to": "TEXT", "bill_no": "TEXT", "no_sale_reason": "TEXT"}


class StockError(Exception):
    """Not enough cans in stock for a bill line."""

    def __init__(self, short: List[Dict[str, Any]]):
        self.short = short
        super().__init__("; ".join(f"{s['label']}: {s['wanted']} wanted, {s['in_stock']} in stock" for s in short))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_tables(conn: Optional[sqlite3.Connection] = None) -> None:
    own = conn is None
    conn = conn or _conn()
    try:
        conn.executescript(BILLING_SQL)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(visits);").fetchall()}
        if cols:
            for c, t in VISIT_COLUMNS.items():
                if c not in cols:
                    conn.execute(f"ALTER TABLE visits ADD COLUMN {c} {t};")
        conn.commit()
    finally:
        if own:
            conn.close()


def _ts(at: Optional[datetime] = None) -> str:
    return (at or clock.now_naive()).strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------- products & codes

def label_of(row) -> str:
    return f"{row['shade_name']} {float(row['pack_size_litres']):g} L"


def products() -> pd.DataFrame:
    """Every product with its stock (all batches) — what the bill screen picks from."""
    with _conn() as conn:
        df = pd.read_sql_query(
            "SELECT p.id AS product_id, p.sku, p.brand, p.shade_name, p.hex_code, p.category, p.pack_size_litres, "
            "p.selling_price, COALESCE(SUM(i.current_stock), 0) AS in_stock FROM products p "
            "LEFT JOIN inventory i ON i.product_id = p.id GROUP BY p.id ORDER BY p.brand, p.shade_name, p.pack_size_litres;",
            conn)
    df["label"] = df.apply(label_of, axis=1)
    return df


def code_for(sku: str) -> str:
    return CODE_PREFIX + sku


def lookup_code(code: str) -> Optional[Dict[str, Any]]:
    """A scanned code -> product (our QR labels, a raw SKU, or a barcode linked earlier)."""
    code = (code or "").strip()
    if not code:
        return None
    sku = code[len(CODE_PREFIX):] if code.upper().startswith(CODE_PREFIX) else code
    with _conn() as conn:
        row = conn.execute("SELECT id FROM products WHERE UPPER(sku) = UPPER(?);", (sku,)).fetchone()
        if row is None:
            try:
                row = conn.execute("SELECT product_id AS id FROM product_codes WHERE code = ?;", (code,)).fetchone()
            except sqlite3.OperationalError:
                row = None
    if row is None:
        return None
    df = products()
    hit = df[df["product_id"] == int(row["id"])]
    return hit.iloc[0].to_dict() if not hit.empty else None


def link_code(code: str, product_id: int) -> None:
    """Remember a can's own barcode for next time."""
    with _conn() as conn:
        ensure_tables(conn)
        conn.execute("INSERT OR REPLACE INTO product_codes (code, product_id) VALUES (?, ?);", (code.strip(), int(product_id)))
        conn.commit()


# ----------------------------------------------------------------------------- bills

def _next_invoice(conn: sqlite3.Connection, at: datetime) -> str:
    prefix = f"APX-{at:%Y%m%d}-"
    row = conn.execute("SELECT invoice_no FROM bills WHERE invoice_no LIKE ? ORDER BY invoice_no DESC LIMIT 1;",
                       (prefix + "%",)).fetchone()
    n = int(row[0].rsplit("-", 1)[1]) + 1 if row else 1
    return f"{prefix}{n:03d}"


def create_bill(lines: Sequence[Dict[str, Any]], customer_type: str = "Retail Walk-in", phone: str = "",
                payment_mode: str = "Cash", at: Optional[datetime] = None, allow_short: bool = False) -> Dict[str, Any]:
    """
    lines: [{"product_id", "qty", "unit_price" (optional: catalogue price)}]
    Writes sales + takes stock off (oldest expiry first) + records the bill, all or nothing.
    Raises StockError if a line wants more cans than are in stock (unless allow_short).
    """
    lines = [dict(l) for l in lines if int(l.get("qty", 0)) > 0]
    if not lines:
        raise ValueError("The bill is empty.")
    at = at or clock.now_naive()
    conn = _conn()
    try:
        ensure_tables(conn)
        conn.execute("BEGIN IMMEDIATE;")                       # one bill at a time: no double-selling the last can
        prices = {r["id"]: r for r in conn.execute(
            f"SELECT id, shade_name, pack_size_litres, selling_price FROM products WHERE id IN "
            f"({','.join('?' * len(lines))});", [int(l["product_id"]) for l in lines]).fetchall()}
        # merge repeated products
        merged: Dict[int, Dict[str, Any]] = {}
        for l in lines:
            pid = int(l["product_id"])
            if pid not in prices:
                raise ValueError(f"Unknown product {pid}.")
            m = merged.setdefault(pid, {"product_id": pid, "qty": 0,
                                        "unit_price": float(l.get("unit_price") or prices[pid]["selling_price"])})
            m["qty"] += int(l["qty"])
        short = []
        for pid, m in merged.items():
            have = conn.execute("SELECT COALESCE(SUM(current_stock), 0) FROM inventory WHERE product_id = ?;", (pid,)).fetchone()[0]
            if have < m["qty"]:
                short.append({"product_id": pid, "label": label_of(prices[pid]), "wanted": m["qty"], "in_stock": int(have)})
        if short and not allow_short:
            raise StockError(short)
        invoice = _next_invoice(conn, at)
        ts = _ts(at)
        moves, stock_after = [], {}
        total = 0.0
        for pid, m in merged.items():
            need = m["qty"]
            # oldest expiry first, so the shop sells the cans that would expire first
            for b in conn.execute("SELECT id, current_stock FROM inventory WHERE product_id = ? AND current_stock > 0 "
                                  "ORDER BY expiry_date ASC, purchase_date ASC, id ASC;", (pid,)).fetchall():
                if need <= 0:
                    break
                take = min(need, int(b["current_stock"]))
                conn.execute("UPDATE inventory SET current_stock = current_stock - ? WHERE id = ?;", (take, b["id"]))
                moves.append((invoice, b["id"], pid, take))
                need -= take
            amount = round(m["qty"] * m["unit_price"], 2)
            total += amount
            conn.execute("INSERT INTO sales (invoice_no, product_id, quantity, unit_price, total_amount, sale_timestamp, "
                         "customer_type) VALUES (?, ?, ?, ?, ?, ?, ?);",
                         (invoice, pid, m["qty"], m["unit_price"], amount, ts, customer_type))
            stock_after[pid] = int(conn.execute("SELECT COALESCE(SUM(current_stock), 0) FROM inventory WHERE product_id = ?;",
                                                (pid,)).fetchone()[0])
        conn.executemany("INSERT INTO bill_stock_moves (invoice_no, inventory_id, product_id, qty) VALUES (?, ?, ?, ?);", moves)
        conn.execute("INSERT INTO bills (invoice_no, created_at, source, customer_type, customer_phone, payment_mode, items, "
                     "total, status) VALUES (?, ?, 'app', ?, ?, ?, ?, ?, 'paid');",
                     (invoice, ts, customer_type, (phone or "").strip() or None, payment_mode,
                      sum(m["qty"] for m in merged.values()), round(total, 2)))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "invoice_no": invoice, "created_at": ts, "total": round(total, 2), "customer_type": customer_type,
        "payment_mode": payment_mode, "phone": phone,
        "lines": [{**m, "label": label_of(prices[pid]), "amount": round(m["qty"] * m["unit_price"], 2),
                   "stock_after": stock_after[pid]} for pid, m in merged.items()],
    }


def void_bill(invoice_no: str) -> bool:
    """Cancel a bill: cans go back to the batches they came from, the sale lines are removed."""
    conn = _conn()
    try:
        conn.execute("BEGIN IMMEDIATE;")
        row = conn.execute("SELECT status FROM bills WHERE invoice_no = ?;", (invoice_no,)).fetchone()
        if row is None or row["status"] == "void":
            conn.rollback()
            return False
        for m in conn.execute("SELECT inventory_id, qty FROM bill_stock_moves WHERE invoice_no = ?;", (invoice_no,)).fetchall():
            conn.execute("UPDATE inventory SET current_stock = current_stock + ? WHERE id = ?;", (m["qty"], m["inventory_id"]))
        conn.execute("DELETE FROM sales WHERE invoice_no = ?;", (invoice_no,))
        conn.execute("UPDATE bills SET status = 'void' WHERE invoice_no = ?;", (invoice_no,))
        try:
            conn.execute("UPDATE visits SET bill_no = NULL WHERE bill_no = ?;", (invoice_no,))
        except sqlite3.OperationalError:
            pass
        conn.commit()
        return True
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def record_no_sale(reason: str, product_id: Optional[int] = None, note: str = "", at: Optional[datetime] = None,
                   source: str = "app") -> None:
    if reason not in REASONS:
        raise ValueError(f"Unknown reason {reason!r}")
    with _conn() as conn:
        ensure_tables(conn)
        conn.execute("INSERT INTO no_sales (at, source, reason, product_id, note) VALUES (?, ?, ?, ?, ?);",
                     (_ts(at), source, reason, int(product_id) if product_id else None, note or None))
        conn.commit()


def bills_on(day: Optional[date] = None, source: str = "app") -> pd.DataFrame:
    ds = (day or clock.today()).strftime("%Y-%m-%d")
    with _conn() as conn:
        ensure_tables(conn)
        return pd.read_sql_query("SELECT * FROM bills WHERE source = ? AND DATE(created_at) = ? ORDER BY created_at DESC;",
                                 conn, params=(source, ds))


def bill_lines(invoice_no: str) -> pd.DataFrame:
    with _conn() as conn:
        df = pd.read_sql_query("SELECT s.product_id, s.quantity, s.unit_price, s.total_amount, p.shade_name, "
                               "p.pack_size_litres, p.brand FROM sales s JOIN products p ON p.id = s.product_id "
                               "WHERE s.invoice_no = ?;", conn, params=(invoice_no,))
    if not df.empty:
        df["label"] = df.apply(label_of, axis=1)
    return df


def no_sales_on(day: Optional[date] = None, source: str = "app") -> pd.DataFrame:
    ds = (day or clock.today()).strftime("%Y-%m-%d")
    with _conn() as conn:
        ensure_tables(conn)
        return pd.read_sql_query(
            "SELECT n.*, p.shade_name, p.pack_size_litres FROM no_sales n LEFT JOIN products p ON p.id = n.product_id "
            "WHERE n.source = ? AND DATE(n.at) = ? ORDER BY n.at DESC;", conn, params=(source, ds))


def whatsapp_link(bill: Dict[str, Any], phone: str = "") -> str:
    """wa.me link with the bill as a message (no WhatsApp API needed; the shop's phone sends it)."""
    lines = [f"*{STORE_SHORT_NAME}* — bill {bill['invoice_no']}", bill["created_at"][:16], ""]
    for l in bill["lines"]:
        lines.append(f"{l['qty']} × {l['label']} @ ₹{l['unit_price']:,.0f} = ₹{l['amount']:,.0f}")
    lines += ["", f"*Total: ₹{bill['total']:,.0f}* ({bill.get('payment_mode', '')})", "Thank you!"]
    digits = "".join(ch for ch in (phone or "") if ch.isdigit())
    if len(digits) == 10:
        digits = "91" + digits                                  # Indian mobile without country code
    return f"https://wa.me/{digits}?text=" + urllib.parse.quote("\n".join(lines))


def asked_not_stocked(days: int = 7, day: Optional[date] = None) -> pd.DataFrame:
    """Shades customers asked for that weren't in stock (from 'Didn't buy')."""
    end = (day or clock.today()) + timedelta(days=1)
    start = end - timedelta(days=days)
    with _conn() as conn:
        ensure_tables(conn)
        df = pd.read_sql_query(
            "SELECT n.product_id, p.shade_name, p.pack_size_litres, p.brand, p.hex_code, COUNT(*) AS times, "
            "MAX(n.at) AS last_at FROM no_sales n JOIN products p ON p.id = n.product_id "
            "WHERE n.reason = 'not_in_stock' AND n.at >= ? AND n.at < ? GROUP BY n.product_id ORDER BY times DESC;",
            conn, params=(start.isoformat(), end.isoformat()))
    if not df.empty:
        df["label"] = df.apply(label_of, axis=1)
    return df


# ----------------------------------------------------------------------------- QR labels

def qr_image(text: str, size: int = 240) -> np.ndarray:
    """A QR code (white background, black modules) as a grey image, using OpenCV's encoder."""
    import cv2

    enc = cv2.QRCodeEncoder.create()
    q = enc.encode(text)
    q = cv2.resize(q, (size, size), interpolation=cv2.INTER_NEAREST)
    return cv2.copyMakeBorder(q, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)


def label_sheet(product_ids: Sequence[int], copies: int = 1, fmt: str = "PDF") -> bytes:
    """
    A4 sheet(s) of can labels: QR code, shade, pack, price, a colour strip.
    3 x 7 labels per page; print, cut, stick on the can or the shelf edge.
    """
    import io

    from PIL import Image, ImageDraw, ImageFont

    def font(sz, bold=False):
        for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "Arial.ttf"):
            try:
                return ImageFont.truetype(name, sz)
            except OSError:
                continue
        try:
            return ImageFont.load_default(size=sz)
        except TypeError:
            return ImageFont.load_default()

    df = products().set_index("product_id")
    items = [int(p) for p in product_ids for _ in range(max(1, copies)) if int(p) in df.index]
    W, H = 1240, 1754                                             # A4 at 150 dpi
    cols, rows, mx, my = 3, 7, 50, 60
    cw, ch = (W - 2 * mx) // cols, (H - 2 * my) // rows
    pages = []
    for start in range(0, max(1, len(items)), cols * rows):
        page = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(page)
        for k, pid in enumerate(items[start:start + cols * rows]):
            r = df.loc[pid]
            x0, y0 = mx + (k % cols) * cw, my + (k // cols) * ch
            d.rectangle([x0 + 4, y0 + 4, x0 + cw - 4, y0 + ch - 4], outline=(200, 200, 200), width=1)
            q = Image.fromarray(qr_image(code_for(r["sku"]), 150)).convert("RGB")
            page.paste(q, (x0 + 10, y0 + (ch - q.height) // 2))
            tx = x0 + 20 + q.width
            hx = r["hex_code"].lstrip("#")
            d.rectangle([tx, y0 + 22, tx + 30, y0 + 52], fill=tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4)),
                        outline=(120, 120, 120))
            d.text((tx + 40, y0 + 24), f"{float(r['pack_size_litres']):g} L", fill="black", font=font(22, True))
            room = x0 + cw - 14 - tx                       # keep text inside the label
            name, nf = r["shade_name"], font(20, True)
            for size in (18, 16):                         # smaller type before cutting the name
                if d.textlength(name, font=nf) <= room:
                    break
                nf = font(size, True)
            if d.textlength(name, font=nf) > room:
                while name and d.textlength(name + "…", font=nf) > room:
                    name = name[:-1]
                name = name.rstrip() + "…"
            d.text((tx, y0 + 64), name, fill="black", font=nf)
            d.text((tx, y0 + 92), r["brand"][:18], fill=(90, 90, 90), font=font(16))
            d.text((tx, y0 + 116), f"₹{float(r['selling_price']):,.0f}", fill="black", font=font(20, True))
            d.text((tx, y0 + 146), r["sku"], fill=(120, 120, 120), font=font(13))
        pages.append(page)
    buf = io.BytesIO()
    if fmt.upper() == "PDF":
        pages[0].save(buf, format="PDF", save_all=True, append_images=pages[1:], resolution=150)
    else:
        pages[0].save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------------------------------- bills <-> camera visits

GRACE_BEFORE_S = 30       # a bill saved just before the camera confirmed the person at the counter
GRACE_AFTER_S = 150       # ... or up to 2.5 minutes after they stepped away (printing, change, packing)
GROUP_S = 90              # people who came in within 90 s of each other and were at the counter together = one group


def _window(v) -> Tuple[Optional[datetime], Optional[datetime]]:
    f = v.get("counter_from") or (v["entered_at"] if v.get("reached_counter") else None)
    t = v.get("counter_to") or (v["left_at"] if v.get("reached_counter") else None)
    if not f or not t:
        return None, None
    return datetime.strptime(f, "%Y-%m-%d %H:%M:%S"), datetime.strptime(t, "%Y-%m-%d %H:%M:%S")


def match_visits(source: str = "camera", day: Optional[date] = None) -> Dict[str, int]:
    """
    Give each finished camera visit its bill (or its didn't-buy reason), by time:
    - a bill goes to the customer who was at the counter when it was saved
      (closest counter time wins); people who came in with them share it;
    - a didn't-buy goes to the customer at the counter then who has no bill.
    Only fills what isn't set yet, so it can run on every refresh.
    """
    ds = (day or clock.today()).strftime("%Y-%m-%d")
    bill_source = "sample" if source == "sample" else "app"
    with _conn() as conn:
        ensure_tables(conn)
        visits = [dict(r) for r in conn.execute(
            "SELECT id, entered_at, left_at, reached_counter, counter_from, counter_to, bill_no, no_sale_reason "
            "FROM visits WHERE source = ? AND DATE(entered_at) = ? ORDER BY entered_at;", (source, ds)).fetchall()]
        bills = conn.execute("SELECT invoice_no, created_at FROM bills WHERE source = ? AND status = 'paid' "
                             "AND DATE(created_at) = ? ORDER BY created_at;", (bill_source, ds)).fetchall()
        nos = conn.execute("SELECT id, at, reason FROM no_sales WHERE source = ? AND DATE(at) = ? ORDER BY at;",
                           (bill_source, ds)).fetchall()
        taken = {v["bill_no"] for v in visits if v["bill_no"]}
        done = 0
        for b in bills:
            if b["invoice_no"] in taken:
                continue
            t = datetime.strptime(b["created_at"], "%Y-%m-%d %H:%M:%S")
            cands = []
            for v in visits:
                if v["bill_no"]:
                    continue
                f, to = _window(v)
                if f is None:
                    continue
                if f - timedelta(seconds=GRACE_BEFORE_S) <= t <= to + timedelta(seconds=GRACE_AFTER_S):
                    gap = 0 if f <= t <= to else min(abs((t - f).total_seconds()), abs((t - to).total_seconds()))
                    cands.append((gap, v))
            if not cands:
                continue
            cands.sort(key=lambda c: c[0])
            buyer = cands[0][1]
            entered = datetime.strptime(buyer["entered_at"], "%Y-%m-%d %H:%M:%S")
            group = [buyer] + [v for g, v in cands[1:]
                               if abs((datetime.strptime(v["entered_at"], "%Y-%m-%d %H:%M:%S") - entered).total_seconds())
                               <= GROUP_S and g - cands[0][0] <= 30]   # came in together, at the counter together
            for v in group:
                v["bill_no"] = b["invoice_no"]
                conn.execute("UPDATE visits SET bill_no = ? WHERE id = ?;", (b["invoice_no"], v["id"]))
                done += 1
            taken.add(b["invoice_no"])
        for n in nos:
            t = datetime.strptime(n["at"], "%Y-%m-%d %H:%M:%S")
            best, best_gap = None, None
            for v in visits:
                if v["bill_no"] or v["no_sale_reason"]:
                    continue
                f, to = _window(v)
                if f is None:
                    f = datetime.strptime(v["entered_at"], "%Y-%m-%d %H:%M:%S")
                    to = datetime.strptime(v["left_at"], "%Y-%m-%d %H:%M:%S")
                if f - timedelta(seconds=GRACE_BEFORE_S) <= t <= to + timedelta(seconds=GRACE_AFTER_S):
                    gap = 0 if f <= t <= to else min(abs((t - f).total_seconds()), abs((t - to).total_seconds()))
                    if best_gap is None or gap < best_gap:
                        best, best_gap = v, gap
            if best is not None:
                best["no_sale_reason"] = n["reason"]
                conn.execute("UPDATE visits SET no_sale_reason = ? WHERE id = ?;", (n["reason"], best["id"]))
                done += 1
        conn.commit()
    return {"matched": done}

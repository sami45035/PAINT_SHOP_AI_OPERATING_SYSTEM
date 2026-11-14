"""Scan & Bill: bills, stock, codes, labels, and matching bills to camera visits."""
import shutil
import sqlite3
import tempfile
import threading
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock


from src import billing, clock
from src import config as cfg
from src import database as db

_PATCHES = []
TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp()) / "paint_shop.db"
    shutil.copy(cfg.SAMPLE_DB_PATH, TMP)
    import src.colour_assistant as ca
    import src.predictor as predictor
    import src.sales_forecast as sf
    import src.shop_insights as si
    import src.smart_alerts as sa

    for mod in (cfg, db, billing, ca, predictor, sf, si, sa):
        p = mock.patch.object(mod, "DB_PATH", TMP)
        p.start()
        _PATCHES.append(p)
    db.ensure_app_tables()


def tearDownModule():
    for p in _PATCHES:
        p.stop()


def q(sql, params=()):
    with sqlite3.connect(TMP) as c:
        return c.execute(sql, params).fetchall()


def new_product(stock_batches, price=500.0):
    """A product of our own with known batches: [(stock, expiry)]."""
    with sqlite3.connect(TMP) as c:
        cur = c.execute("INSERT INTO products (sku, brand, sub_brand, shade_name, shade_code, hex_code, category, finish, "
                        "pack_size_litres, cost_price, selling_price, supplier_id) SELECT 'TEST-' || (MAX(id) + 1), "
                        "'TestBrand', 'T', 'Test Shade', 'T-1', '#AABBCC', 'Interior Emulsion', 'Matt', 4.0, 300, ?, "
                        "supplier_id FROM products;", (price,))
        pid = cur.lastrowid
        for i, (n, exp) in enumerate(stock_batches):
            c.execute("INSERT INTO inventory (product_id, batch_no, current_stock, on_order, min_threshold, max_threshold, "
                      "purchase_date, expiry_date, shelf_location) VALUES (?, ?, ?, 0, 2, 20, '2026-01-01', ?, 'T');",
                      (pid, f"TB{i}", n, exp))
        c.commit()
    return pid


def stock(pid):
    return {b: n for b, n in q("SELECT batch_no, current_stock FROM inventory WHERE product_id = ?;", (pid,))}


class TestBills(unittest.TestCase):
    def test_bill_sells_oldest_expiry_first_and_writes_sales(self):
        pid = new_product([(5, "2027-06-01"), (3, "2027-01-01")])      # TB1 expires first
        bill = billing.create_bill([{"product_id": pid, "qty": 4}], customer_type="Painter", payment_mode="UPI")
        self.assertEqual(stock(pid), {"TB0": 4, "TB1": 0})              # 3 from TB1, then 1 from TB0
        rows = q("SELECT product_id, quantity, unit_price, total_amount, customer_type FROM sales WHERE invoice_no = ?;",
                 (bill["invoice_no"],))
        self.assertEqual(rows, [(pid, 4, 500.0, 2000.0, "Painter")])
        self.assertEqual(bill["total"], 2000.0)
        self.assertEqual(bill["lines"][0]["stock_after"], 4)
        self.assertEqual(q("SELECT items, total, status FROM bills WHERE invoice_no = ?;", (bill["invoice_no"],)),
                         [(4, 2000.0, "paid")])
        self.assertTrue(bill["invoice_no"].startswith("APX-" + clock.now_naive().strftime("%Y%m%d")))

    def test_repeated_lines_merge_and_numbers_increase(self):
        pid = new_product([(10, "2027-06-01")])
        b1 = billing.create_bill([{"product_id": pid, "qty": 1}, {"product_id": pid, "qty": 2}])
        b2 = billing.create_bill([{"product_id": pid, "qty": 1}])
        self.assertEqual(b1["lines"][0]["qty"], 3)
        self.assertEqual(stock(pid), {"TB0": 6})
        self.assertGreater(b2["invoice_no"], b1["invoice_no"])

    def test_not_enough_stock(self):
        pid = new_product([(2, "2027-06-01")])
        with self.assertRaises(billing.StockError) as ctx:
            billing.create_bill([{"product_id": pid, "qty": 3}])
        self.assertEqual(ctx.exception.short[0]["in_stock"], 2)
        self.assertEqual(stock(pid), {"TB0": 2})                        # nothing changed
        self.assertEqual(q("SELECT COUNT(*) FROM sales WHERE product_id = ?;", (pid,)), [(0,)])
        billing.create_bill([{"product_id": pid, "qty": 3}], allow_short=True)
        self.assertEqual(stock(pid), {"TB0": 0})                        # never below zero

    def test_void_puts_cans_back_where_they_came_from(self):
        pid = new_product([(5, "2027-06-01"), (3, "2027-01-01")])
        bill = billing.create_bill([{"product_id": pid, "qty": 6}])
        self.assertEqual(stock(pid), {"TB0": 2, "TB1": 0})
        self.assertTrue(billing.void_bill(bill["invoice_no"]))
        self.assertEqual(stock(pid), {"TB0": 5, "TB1": 3})
        self.assertEqual(q("SELECT COUNT(*) FROM sales WHERE invoice_no = ?;", (bill["invoice_no"],)), [(0,)])
        self.assertFalse(billing.void_bill(bill["invoice_no"]))         # only once

    def test_two_counters_cannot_both_sell_the_last_can(self):
        pid = new_product([(1, "2027-06-01")])
        results = []

        def sell():
            try:
                billing.create_bill([{"product_id": pid, "qty": 1}])
                results.append("sold")
            except billing.StockError:
                results.append("short")

        ts = [threading.Thread(target=sell) for _ in range(4)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        self.assertEqual(sorted(results), ["short", "short", "short", "sold"])
        self.assertEqual(stock(pid), {"TB0": 0})

    def test_empty_bill(self):
        with self.assertRaises(ValueError):
            billing.create_bill([])

    def test_mark_ordered_counts_once_per_order(self):
        pid = new_product([(1, "2027-06-01"), (1, "2027-07-01")])
        db.mark_ordered(pid, 12)
        self.assertEqual(q("SELECT SUM(on_order) FROM inventory WHERE product_id = ?;", (pid,)), [(12,)])


class TestCodesAndLabels(unittest.TestCase):
    def test_lookup(self):
        sku = q("SELECT sku FROM products WHERE id = 2;")[0][0]
        self.assertEqual(billing.lookup_code("APX:" + sku)["product_id"], 2)
        self.assertEqual(billing.lookup_code(sku.lower())["product_id"], 2)
        self.assertIsNone(billing.lookup_code("8901234567890"))
        billing.link_code("8901234567890", 5)                           # the can's own barcode, linked once
        self.assertEqual(billing.lookup_code("8901234567890")["product_id"], 5)

    def test_qr_reads_back_and_label_sheet(self):
        import cv2

        sku = q("SELECT sku FROM products WHERE id = 7;")[0][0]
        img = billing.qr_image(billing.code_for(sku))
        text, _, _ = cv2.QRCodeDetector().detectAndDecode(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        self.assertEqual(text, "APX:" + sku)
        pdf = billing.label_sheet(list(range(1, 25)), fmt="PDF")
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 10_000)

    def test_whatsapp_link(self):
        bill = {"invoice_no": "APX-20261111-007", "created_at": "2026-11-11 18:30:00", "total": 4466.0,
                "payment_mode": "UPI", "lines": [{"qty": 2, "label": "Royale Blue 4 L", "unit_price": 2233.0, "amount": 4466.0}]}
        link = billing.whatsapp_link(bill, "98765 43210")
        self.assertTrue(link.startswith("https://wa.me/919876543210?text="))
        self.assertIn("APX-20261111-007", link)
        self.assertIn("4%2C466", link)                                   # ₹4,466 url-encoded


class TestMatching(unittest.TestCase):
    """A bill goes to the customer at the counter when it was saved; companions share it; didn't-buys get reasons."""

    def setUp(self):
        with sqlite3.connect(TMP) as c:
            c.execute("DELETE FROM visits WHERE source = 'camera';")
            c.execute("DELETE FROM bills WHERE source = 'app';")
            c.execute("DELETE FROM no_sales WHERE source = 'app';")
        self.base = datetime.combine(clock.today(), datetime.min.time()).replace(hour=11)

    def visit(self, enter_min, leave_min, counter=None):
        t = lambda m: self.base + timedelta(minutes=m)
        return db.log_visit("camera", t(enter_min), t(leave_min), counter is not None, [], 0, 0,
                            "reached_counter" if counter else "browsed",
                            counter_from=t(counter[0]) if counter else None, counter_to=t(counter[1]) if counter else None)

    def bill_at(self, minute):
        pid = new_product([(5, "2027-06-01")])
        return billing.create_bill([{"product_id": pid, "qty": 1}], at=self.base + timedelta(minutes=minute))["invoice_no"]

    def outcome(self, vid):
        return q("SELECT bill_no, no_sale_reason FROM visits WHERE id = ?;", (vid,))[0]

    def test_bill_goes_to_the_customer_at_the_counter(self):
        a = self.visit(0, 10, counter=(6, 9))          # at the counter 11:06-11:09
        b = self.visit(3, 12, counter=(9.5, 11))       # at the counter after a
        c = self.visit(4, 8)                            # never at the counter
        inv = self.bill_at(8)
        billing.match_visits("camera", clock.today())
        self.assertEqual(self.outcome(a)[0], inv)
        self.assertIsNone(self.outcome(b)[0])
        self.assertIsNone(self.outcome(c)[0])

    def test_group_shares_a_bill_and_late_bill_still_matches(self):
        a = self.visit(0, 10, counter=(6, 9))
        b = self.visit(0.5, 10, counter=(6.2, 9))      # came in with a, stood at the counter with them
        inv = self.bill_at(10.5)                         # printed 1.5 min after they stepped away
        billing.match_visits("camera", clock.today())
        self.assertEqual(self.outcome(a)[0], inv)
        self.assertEqual(self.outcome(b)[0], inv)

    def test_bill_nobody_was_there_for_stays_unmatched(self):
        a = self.visit(0, 10, counter=(6, 9))
        self.bill_at(30)
        billing.match_visits("camera", clock.today())
        self.assertIsNone(self.outcome(a)[0])

    def test_didnt_buy_reason_goes_to_the_customer_without_a_bill(self):
        a = self.visit(0, 10, counter=(6, 9))
        b = self.visit(2, 9, counter=(7, 8.5))
        self.bill_at(8.8)                                # a pays (closest counter time: a until 9:00 vs b until 8:30 -> both 0 gap)
        billing.record_no_sale("price", at=self.base + timedelta(minutes=8, seconds=45))
        billing.match_visits("camera", clock.today())
        got = {self.outcome(a), self.outcome(b)}
        self.assertEqual(sum(1 for bill, _ in got if bill), 1)          # one bought
        self.assertIn("price", {reason for _, reason in got})          # the other: didn't buy, price

    def test_matching_is_idempotent_and_classification(self):
        from src import shop_insights as si

        a = self.visit(0, 10, counter=(6, 9))
        self.visit(20, 25)
        self.bill_at(8)
        billing.match_visits("camera", clock.today())
        billing.match_visits("camera", clock.today())
        self.assertEqual(q("SELECT COUNT(*) FROM visits WHERE source='camera' AND bill_no IS NOT NULL;"), [(1,)])
        with mock.patch.object(clock, "now_naive", return_value=self.base + timedelta(hours=1)):
            b = si.buyers("camera")
        self.assertEqual((b["bought_visits"], b["didnt_buy"]), (1, 1))
        self.assertTrue(b["classified"])
        self.assertIsNotNone(self.outcome(a)[0])

    def test_void_unlinks_the_visit(self):
        a = self.visit(0, 10, counter=(6, 9))
        inv = self.bill_at(8)
        billing.match_visits("camera", clock.today())
        billing.void_bill(inv)
        self.assertIsNone(self.outcome(a)[0])


class TestLostSales(unittest.TestCase):
    def test_asked_but_not_stocked_becomes_an_alert(self):
        from src.smart_alerts import _lost_sale_alerts

        pid = new_product([(0, "2027-06-01")])
        for _ in range(3):
            billing.record_no_sale("not_in_stock", product_id=pid)
        asked = billing.asked_not_stocked(7)
        self.assertEqual(int(asked[asked["product_id"] == pid]["times"].iloc[0]), 3)
        alerts = [a for a in _lost_sale_alerts(clock.today()) if a["product_id"] == pid]
        self.assertEqual(len(alerts), 1)
        self.assertIn("none in the shop", alerts[0]["description"])
        with self.assertRaises(ValueError):
            billing.record_no_sale("bored")


if __name__ == "__main__":
    unittest.main()

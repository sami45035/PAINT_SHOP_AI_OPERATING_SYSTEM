"""
Shop camera tests — one section per job the camera does:
footfall, buyers, theft, fast movers, shelf stock / pre-order.

Runs on a temporary copy of the sample database, moved forward to today.
"""
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

from src import clock
from src import config as cfg
from src import database as db

_PATCHES = []
TMP_DB = None


def setUpModule():
    global TMP_DB
    TMP_DB = Path(tempfile.mkdtemp()) / "paint_shop.db"
    shutil.copy(cfg.DB_PATH, TMP_DB)
    import src.predictor as predictor
    import src.shop_insights as si
    import src.smart_alerts as sa

    for mod in (cfg, db, si, predictor, sa):
        p = mock.patch.object(mod, "DB_PATH", TMP_DB)
        p.start()
        _PATCHES.append(p)
    db.ensure_app_tables()
    db.ensure_demo_dates_current()


def tearDownModule():
    for p in _PATCHES:
        p.stop()


def _at(hour: int, minute: int = 0):
    """Freeze the store clock at a time today."""
    t = datetime.combine(clock.today(), datetime.min.time()).replace(hour=hour, minute=minute)
    return [mock.patch.object(clock, "now_naive", return_value=t),
            mock.patch.object(clock, "now", return_value=t.replace(tzinfo=clock.STORE_TZ))]


class Frozen:
    def __init__(self, hour, minute=0):
        self.ps = _at(hour, minute)

    def __enter__(self):
        for p in self.ps:
            p.start()

    def __exit__(self, *a):
        for p in self.ps:
            p.stop()


# ----------------------------------------------------------------------------- the demo shop, end to end
class TestDemoShopStory(unittest.TestCase):
    """One loop of the demo shop must produce exactly the story it acts out."""

    @classmethod
    def setUpClass(cls):
        import src.camera_stream as cs

        db.clear_camera_data("demo")
        cls.visitors_before = db.get_sales_summary(30)["today_visitors"]
        t = [1000.0]
        with mock.patch.object(cs.time, "time", side_effect=lambda: t[0]), \
                mock.patch.object(cs.StoreVisionEngine, "_save_snapshot", lambda *a, **k: "snap.jpg"), \
                Frozen(15):
            eng = cs.StoreVisionEngine("demo")
            eng._demo_t0 = 1000.0
            eng.open_stream()
            for _ in range(int(66 / 0.25)):
                cls.frame, cls.m = eng.process_frame()
                t[0] += 0.25
        cls.eng = eng
        with sqlite3.connect(TMP_DB) as c:
            cls.visits = c.execute("SELECT outcome, reached_counter, picked, put_back, shelves FROM visits "
                                   "WHERE source = 'demo' ORDER BY id;").fetchall()
            cls.events = c.execute("SELECT slot_id, kind, qty FROM shelf_events WHERE source = 'demo' ORDER BY id;").fetchall()
            cls.security = c.execute("SELECT event_type, details FROM security_events WHERE source = 'demo';").fetchall()
            cls.door = c.execute("SELECT SUM(count_in), SUM(count_out) FROM footfall_logs WHERE source = 'demo';").fetchone()

    def test_footfall(self):
        self.assertEqual(tuple(self.door), (4, 4))                    # buyer, browser, walk-out, enquiry
        self.assertEqual(self.m["inside_now"], 0)

    def test_staff_coming_from_the_store_room_is_not_a_visitor(self):
        self.assertEqual(len(self.visits), 4)

    def test_buyer_browser_walkout_enquiry(self):
        outcomes = [v[0] for v in self.visits]
        self.assertEqual(outcomes, ["reached_counter", "browsed", "unpaid_exit", "reached_counter"])
        buyer, browser, walkout, enquiry = self.visits
        self.assertEqual(buyer[2], 1)                  # took the Royale Blue it paid for
        self.assertEqual((browser[2], browser[3]), (0, 1))   # picked up and put back
        self.assertEqual(walkout[2], 1)
        self.assertIn("B1", walkout[4])
        self.assertEqual((enquiry[2], enquiry[4]), (0, ""))

    def test_shelf_events(self):
        self.assertEqual(self.events, [("B2", "pick", 1), ("A2", "pick", 1), ("A2", "put_back", 1), ("B1", "pick", 1),
                                       ("B1", "restock", 1), ("B2", "restock", 1)])

    def test_walkout_raises_one_alert_naming_the_can(self):
        self.assertEqual(len(self.security), 1)
        kind, details = self.security[0]
        self.assertEqual(kind, "UNPAID_EXIT")
        self.assertIn("Warm Beige 4 L", details)

    def test_demo_never_counts_as_real(self):
        self.assertEqual(db.get_sales_summary(30)["today_visitors"], self.visitors_before)
        self.assertFalse((db.get_recent_security_events(50)["source"] == "demo").any())

    def test_shelf_counts_match_the_drawing(self):
        truth = self.eng.demo_shop.cans_at(65.75 % 64)
        self.assertEqual({k: v for k, v in self.m["shelves"].items()}, truth)


# ----------------------------------------------------------------------------- shelves
class TestShelfMonitor(unittest.TestCase):
    def setUp(self):
        from src.shop_vision import DemoShop, ShelfMonitor, demo_layout

        self.lay = demo_layout()
        self.shop = DemoShop(self.lay)
        self.mon = ShelfMonitor(self.lay["shelves"])

    def test_counts_cans_without_calibration(self):
        from src.shop_vision import boxes_touch, to_px

        for t in np.arange(0, 64, 1.0):
            f, people = self.shop.frame(t)
            truth = self.shop.cans_at(t)
            for s in self.lay["shelves"]:
                if any(boxes_touch(b, to_px(s["box"], 960, 540), 9) for b in people):
                    continue
                self.assertEqual(self.mon.read(f, s), truth[s["id"]], f"t={t} slot={s['id']}")

    def test_hand_in_front_of_shelf_is_not_a_missing_can(self):
        f, _ = self.shop.frame(1.0)
        for _ in range(3):
            self.mon.update(f, [])
        before = dict(self.mon.counts())
        hidden = f.copy()
        hidden[150:400, 20:330] = (60, 60, 60)          # someone standing right in front of the shelf
        for _ in range(5):
            self.assertEqual(self.mon.update(hidden, [(20, 150, 310, 250)]), [])
        self.assertEqual(self.mon.counts(), before)

    def test_change_needs_steady_readings(self):
        f0, p0 = self.shop.frame(1.0)
        f1, p1 = self.shop.frame(12.0)                   # B2 has one can fewer (buyer took it at 9 s)
        for _ in range(3):
            self.mon.update(f0, p0)
        self.assertEqual(self.mon.update(f1, p1), [])
        self.assertEqual(self.mon.update(f1, p1), [])
        ch = self.mon.update(f1, p1)
        self.assertEqual([(s["id"], d) for s, d, _ in ch], [("B2", -1)])

    def test_calibration_uses_the_counts_given(self):
        f, _ = self.shop.frame(1.0)
        done = self.mon.calibrate(f, self.shop.cans_at(1.0))
        self.assertEqual(done, self.shop.cans_at(1.0))
        for s in self.lay["shelves"]:
            self.assertEqual(self.mon.read(f, s), self.shop.cans_at(1.0)[s["id"]])


# ----------------------------------------------------------------------------- footfall: door line
class TestDoorLine(unittest.TestCase):
    def test_walking_along_the_back_wall_is_not_counted(self):
        from src.camera_stream import StoreVisionEngine

        eng = StoreVisionEngine("webcam", log_to_db=False)
        eng.layout["door"] = {"y": 0.5, "x1": 0.4, "x2": 0.6}
        eng._frame_w, h, line = 960, 540, 270
        total = [0, 0]
        for i, yf in enumerate(list(range(200, 340, 12)) + list(range(340, 200, -12))):   # crossing at x = 850 (store room)
            e, x = eng._update_tracks([(830, int(yf - 0.9 * 150), 50, 150)], line, i * 0.2, h)
            total[0] += e
            total[1] += x
        self.assertEqual(total, [0, 0])
        eng2 = StoreVisionEngine("webcam", log_to_db=False)
        eng2.layout["door"] = {"y": 0.5, "x1": 0.4, "x2": 0.6}
        eng2._frame_w = 960
        for i, yf in enumerate(range(200, 400, 12)):                                     # through the doorway, then aside
            x = 480 - i * 25
            e, _ = eng2._update_tracks([(x, int(yf - 0.9 * 150), 50, 150)], line, i * 0.2, h)
            total[0] += e
        self.assertEqual(total[0], 1)


# ----------------------------------------------------------------------------- insights on the sample history
class TestInsights(unittest.TestCase):
    def test_bought_is_bills_over_visitors(self):
        from src import shop_insights as si

        with Frozen(18):
            b = si.buyers("sample")
            self.assertGreater(b["visitors"], 0)
            self.assertAlmostEqual(b["bought_pct"], round(b["bills"] / b["visitors"] * 100, 1))
            self.assertLessEqual(b["at_counter"], b["finished_visits"])
            self.assertEqual(b["finished_visits"] - b["at_counter"], b["left_without_counter"])

    def test_nothing_after_now(self):
        from src import shop_insights as si

        with Frozen(12):
            f = si.footfall("sample")
            self.assertTrue(f["hourly"].empty or f["hourly"]["hour"].max() <= 12)

    def test_shelf_count_is_last_count_plus_changes(self):
        from src import shop_insights as si

        with Frozen(20, 30):
            sh = si.shelves("sample").set_index("slot_id")
            with sqlite3.connect(TMP_DB) as c:
                for sid in sh.index:
                    base, at = c.execute("SELECT cans, updated_at FROM shelf_status WHERE source='sample' AND slot_id=?;",
                                         (sid,)).fetchone()
                    rows = c.execute("SELECT kind, qty FROM shelf_events WHERE source='sample' AND slot_id=? AND at > ? "
                                     "AND at <= ?;", (sid, at, clock.now_naive().strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
                    exp = base + sum(q for k, q in rows if k != "pick") - sum(q for k, q in rows if k == "pick")
                    self.assertEqual(sh.loc[sid, "cans"], max(0, min(sh.loc[sid, "capacity"], exp)))

    def test_low_shelf_gets_a_refill_with_store_room_stock(self):
        from src import shop_insights as si

        with Frozen(20, 30):
            po = si.preorder("sample")
        b3 = [r for r in po["refill"] if r["slot_id"] == "B3"]
        self.assertTrue(b3)
        self.assertGreater(b3[0]["move"], 0)
        self.assertLessEqual(b3[0]["move"], b3[0]["store_room"])

    def test_picked_up_not_bought_and_rising_demand(self):
        from src import shop_insights as si

        with Frozen(20, 30):
            fm = si.fast_movers("sample", 7)
        self.assertIn("Teal Serenity 1 L", [h["label"] for h in fm["hesitant"]])
        self.assertIn("Royale Blue 4 L", [r["label"] for r in fm["rising"]])
        top = fm["top"]
        self.assertTrue((top["units"].diff().dropna() <= 0).all())          # fastest first

    def test_camera_trend_sizes_the_order(self):
        from src import shop_insights as si

        with Frozen(20, 30):
            cam = si.interest("sample", 7)
            po = si.preorder("sample")
        for o in po["order"]:
            r = cam[cam["product_id"] == o["product_id"]]
            if not r.empty and r["picked"].iloc[0] >= 5 and r["picked_prev"].iloc[0] >= 3:
                self.assertAlmostEqual(o["camera_factor"], round(max(0.8, min(1.5, r["picked"].iloc[0] / r["picked_prev"].iloc[0])), 2))
            elif not o["watched"]:
                self.assertEqual(o["camera_factor"], 1.0)

    def test_walkout_today_shows_as_shelf_vs_till_gap(self):
        from src import shop_insights as si

        with Frozen(20, 30):
            th = si.theft("sample")
        self.assertEqual(th["walkouts_today"], 1)
        self.assertIn("Royale Blue 4 L", [g["label"] for g in th["shelf_vs_till"]])
        with Frozen(10, 40):                       # 3 minutes after the walk-out: still in the 20-minute grace
            th = si.theft("sample")
        self.assertEqual(th["shelf_vs_till"], [])

    def test_alerts_and_refill_action(self):
        from src import shop_insights as si
        from src.smart_alerts import generate_smart_alerts

        with Frozen(20, 30):
            cam = generate_smart_alerts()["camera"]
            types = {a["type"] for a in cam}
            self.assertTrue({"REFILL_SHELF", "UNPAID_EXIT", "SHELF_TILL_GAP", "PICKED_NOT_BOUGHT"} <= types)
            refill = next(a for a in cam if a["type"] == "REFILL_SHELF")
            before = si.shelves("sample").set_index("slot_id").loc[refill["slot_id"], "cans"]
            db.log_shelf_event("sample", refill["slot_id"], refill["product_id"], "restock", refill["move"],
                               at=clock.now_naive() - timedelta(seconds=1))
            after = si.shelves("sample").set_index("slot_id").loc[refill["slot_id"], "cans"]
            with sqlite3.connect(TMP_DB) as c:       # undo, so other tests still see the low shelf
                c.execute("DELETE FROM shelf_events WHERE id = (SELECT MAX(id) FROM shelf_events);")
            self.assertEqual(after, before + refill["move"])


class TestAuraCamera(unittest.TestCase):
    def test_routes_camera_questions(self):
        from src.ai_assistant import deterministic_assistant_response as ask

        self.assertIn("came in", ask("How many customers bought today?"))
        self.assertIn("Theft checks", ask("Any theft today?"))
        self.assertTrue("Refill" in ask("Which shelf needs refilling?") or "stocked" in ask("Which shelf needs refilling?"))


if __name__ == "__main__":
    unittest.main()

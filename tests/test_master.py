"""
Test suite for Paint Shop AI — one section per layer.

Run:  python -m pytest -q        (or: python -m unittest discover tests)
"""
import shutil
import sqlite3
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np

from src import clock
from src import database as db
from src.config import DB_PATH, SAMPLE_ROOMS_DIR, SHADES_CATALOG_PATH


def _sample_room(name="master_bedroom.jpg"):
    import cv2

    return cv2.cvtColor(cv2.imread(str(SAMPLE_ROOMS_DIR / name)), cv2.COLOR_BGR2RGB)


class FakeSegFormer:
    """Stands in for the ONNX session: returns ADE20K-style logits for the sample bedroom's layout."""

    class _Inp:
        name = "pixel_values"

    def get_inputs(self):
        return [self._Inp()]

    def run(self, _outputs, feeds):
        import cv2

        assert feeds["pixel_values"].shape == (1, 3, 512, 512)
        h = w = 128

        def P(pts):
            m = np.zeros((h, w), np.uint8)
            cv2.fillPoly(m, [np.array([[int(x * w), int(y * h)] for x, y in pts], np.int32)], 1)
            return m > 0

        bx0, by0, bx1, by1 = .22, .16, .78, .74
        ceiling = P([(0, 0), (1, 0), (bx1, by0), (bx0, by0)])
        floor = P([(0, 1), (bx0, by1), (bx1, by1), (1, 1)])
        bed = P([(.33, .52), (.67, .52), (.67, .66), (.78, .94), (.22, .94), (.30, .66), (.33, .66)])
        door = P([(.05, .27), (.15, .24), (.15, .82), (.05, .92)])
        win = P([(.86, .26), (.95, .20), (.95, .56), (.86, .57)])
        art = P([(.45, .24), (.55, .24), (.55, .41), (.45, .41)])
        wall = ~(ceiling | floor | bed | door | win | art)
        logits = np.zeros((1, 150, h, w), np.float32)
        for cid, m in [(0, wall), (5, ceiling), (3, floor & ~bed), (7, bed), (14, door), (8, win), (22, art)]:
            logits[0, cid][m] = 8.0
        return [logits]


LABELS = {0: "wall", 3: "floor", 5: "ceiling", 7: "bed", 8: "windowpane", 14: "door", 22: "painting"}


# --------------------------------------------------------------------- Clock
class TestClock(unittest.TestCase):
    def test_store_hours_use_store_time(self):
        self.assertTrue(clock.is_store_open(datetime(2026, 11, 7, 10, 0)))
        self.assertFalse(clock.is_store_open(datetime(2026, 11, 7, 22, 57)))
        self.assertFalse(clock.is_store_open(datetime(2026, 11, 7, 8, 59)))
        self.assertIn("closes", clock.opening_label(datetime(2026, 11, 7, 12, 0)))
        self.assertIn("opens", clock.opening_label(datetime(2026, 11, 7, 23, 0)))

    def test_today_is_timezone_aware(self):
        self.assertIsNotNone(clock.now().tzinfo)


# --------------------------------------------------------------------- Layer 1
class TestLayer1Data(unittest.TestCase):
    def test_tables_and_catalog(self):
        self.assertTrue(DB_PATH.exists())
        self.assertTrue(SHADES_CATALOG_PATH.exists())
        self.assertGreaterEqual(len(db.get_all_products()), 90)
        inv = db.get_inventory_status()
        self.assertIn("health_status", inv.columns)
        self.assertGreaterEqual(len(inv), 90)

    def test_demo_dates_shift_forward_together(self):
        tmp = Path(tempfile.mkdtemp()) / "shop.db"
        shutil.copy(DB_PATH, tmp)
        with sqlite3.connect(tmp) as c:
            latest = c.execute("SELECT MAX(DATE(sale_timestamp)) FROM sales").fetchone()[0]
            exp_before = c.execute("SELECT expiry_date FROM inventory WHERE id = 1").fetchone()[0]
        future = datetime.strptime(latest, "%Y-%m-%d").date() + timedelta(days=10)
        with mock.patch.object(db, "DB_PATH", tmp), mock.patch.object(db, "DEMO_MODE", True), \
                mock.patch.object(db.clock, "today", return_value=future):
            self.assertEqual(db.ensure_demo_dates_current(), 10)
            self.assertEqual(db.ensure_demo_dates_current(), 0)  # idempotent
        with sqlite3.connect(tmp) as c:
            self.assertEqual(c.execute("SELECT MAX(DATE(sale_timestamp)) FROM sales").fetchone()[0], future.isoformat())
            exp_after = c.execute("SELECT expiry_date FROM inventory WHERE id = 1").fetchone()[0]
        delta = (datetime.strptime(exp_after, "%Y-%m-%d") - datetime.strptime(exp_before, "%Y-%m-%d")).days
        self.assertEqual(delta, 10)


# --------------------------------------------------------------------- Layer 2
class TestLayer2KPIs(unittest.TestCase):
    def test_conversion_is_bills_over_visitors(self):
        s = db.get_sales_summary(30)
        if s["today_visitors"]:
            self.assertAlmostEqual(s["today_conversion_pct"], round(s["today_invoices"] / s["today_visitors"] * 100, 1))
        self.assertGreater(s["total_inventory_cost"], 10000)

    def test_daily_visitors_not_multiplied_by_sales_rows(self):
        daily = db.get_daily_traffic(7)
        with sqlite3.connect(DB_PATH) as c:
            for r in daily.itertuples(index=False):
                raw = c.execute("SELECT SUM(count_in) FROM footfall_logs WHERE DATE(timestamp) = ?", (r.day,)).fetchone()[0]
                self.assertEqual(int(r.visitors), int(raw))

    def test_season_status_uses_today(self):
        st = db.get_season_status(date(2026, 11, 7))
        self.assertTrue(any("Diwali" in e["event_name"] for e in st["active"]))
        self.assertFalse(any("Monsoon" in e["event_name"] for e in st["active"]))


# --------------------------------------------------------------------- Layer 3
class TestLayer3Forecast(unittest.TestCase):
    def test_forecast_columns_and_order(self):
        from src.predictor import URGENCY_ORDER, compute_sales_forecast

        fc = compute_sales_forecast(30)
        for col in ("forecast_7d", "forecast_30d", "days_to_stockout", "suggested_reorder_qty", "reorder_urgency"):
            self.assertIn(col, fc.columns)
        ranks = fc["reorder_urgency"].map(URGENCY_ORDER).tolist()
        self.assertEqual(ranks, sorted(ranks))
        moq = fc[fc["suggested_reorder_qty"] > 0]
        self.assertTrue(((moq["suggested_reorder_qty"] % moq["moq_units"]) == 0).all())

    def test_festival_boost_stops_when_festival_ends(self):
        import pandas as pd

        from src.predictor import _multiplier_series

        cal = pd.DataFrame([{"start": date(2026, 10, 10), "end": date(2026, 11, 15),
                             "impact_category": "Interior & Exterior Emulsions", "expected_multiplier": 2.2}])
        days = [date(2026, 11, 14), date(2026, 11, 15), date(2026, 11, 16)]
        self.assertEqual(_multiplier_series(cal, "Interior Emulsion", "Matte", days), [2.2, 2.2, 1.0])
        self.assertEqual(_multiplier_series(cal, "Primer", "Undercoat", days), [1.0, 1.0, 1.0])

    def test_deal_checker(self):
        from src.predictor import evaluate_supplier_clearance_deal

        good = evaluate_supplier_clearance_deal(2, 1200.0, 20, 24)
        self.assertEqual(good["badge"], "success")
        risky = evaluate_supplier_clearance_deal(2, 1200.0, 5000, 3)
        self.assertEqual(risky["badge"], "critical")


# --------------------------------------------------------------------- Layer 4
class TestLayer4Alerts(unittest.TestCase):
    def test_alert_groups_and_wording(self):
        from src.smart_alerts import alert_id, flatten_alerts, generate_smart_alerts

        a = generate_smart_alerts()
        self.assertEqual(set(a), {"critical", "expiring", "dead_stock", "anomalies"})
        for x in a["expiring"]:
            self.assertIn("expire", x["title"])
        for x in a["anomalies"]:
            self.assertLess(int(x["description"].split(" visitors")[0]), 1000)  # the old bug said 11,130
        flat = flatten_alerts(a)
        ids = [alert_id(x) for x in flat]
        self.assertEqual(len(ids), len(set(ids)))


# --------------------------------------------------------------------- Layer 5
class TestLayer5Camera(unittest.TestCase):
    def test_demo_counts_without_touching_database(self):
        import src.camera_stream as cs

        t = [1000.0]
        with mock.patch.object(cs.time, "time", side_effect=lambda: t[0]), \
                mock.patch.object(cs, "log_footfall") as lf, mock.patch.object(cs, "log_security_event") as ls:
            eng = cs.StoreVisionEngine("demo")
            eng._demo_t0 = 1000.0
            self.assertTrue(eng.open_stream())
            for _ in range(40):  # 20 seconds of demo
                frame, m = eng.process_frame()
                t[0] += 0.5
            self.assertGreaterEqual(m["entered"], 1)
            self.assertGreaterEqual(m["exited"], 1)
            lf.assert_not_called()
            ls.assert_not_called()

    def test_missing_camera_is_reported(self):
        from src.camera_stream import StoreVisionEngine

        eng = StoreVisionEngine("esp32", "http://10.255.255.1:81/stream")
        self.assertFalse(eng.open_stream())
        frame, m = eng.process_frame()
        self.assertIsNone(frame)
        self.assertFalse(m["connected"])

    def test_alert_cooldown(self):
        import src.camera_stream as cs

        with mock.patch.object(cs, "log_security_event") as ls:
            eng = cs.StoreVisionEngine("webcam", log_to_db=True)
            eng._save_snapshot = lambda *a, **k: "data/security_snapshots/x.jpg"
            frame = np.zeros((240, 320, 3), np.uint8)
            first = eng._maybe_alert("AFTER_HOURS_MOTION", "Main", "x", frame, clock.now())
            second = eng._maybe_alert("AFTER_HOURS_MOTION", "Main", "x", frame, clock.now())
            self.assertIsNotNone(first)
            self.assertIsNone(second)
            self.assertEqual(ls.call_count, 1)


# --------------------------------------------------------------------- Layer 6
class TestLayer6ColourStudio(unittest.TestCase):
    def test_model_path_finds_three_walls(self):
        from src.colour_studio.segmentation import WallSegmenter

        res = WallSegmenter(FakeSegFormer(), LABELS).segment(_sample_room())
        names = sorted(r.name for r in res.regions)
        self.assertEqual(names, ["Back wall", "Left wall", "Right wall"])
        back = [r for r in res.regions if r.name == "Back wall"][0]
        self.assertEqual(res.region_at(0.5, 0.25).id, back.id)

    def test_tap_mode_separates_walls(self):
        from src.colour_studio.segmentation import tap_result

        for room in ("master_bedroom.jpg", "modern_living_room.jpg", "cozy_dining_room.jpg"):
            res = tap_result(_sample_room(room), [(0.5, 0.2), (0.05, 0.12), (0.95, 0.12)])
            self.assertEqual(len(res.regions), 3, room)
            self.assertLess(max(r.area_share for r in res.regions), 0.35, room)  # no leaking round corners

    def test_recolour_keeps_light_and_leaves_rest_alone(self):
        from src.colour_studio.recolour import recolour, srgb_to_linear
        from src.colour_studio.segmentation import WallSegmenter

        res = WallSegmenter(FakeSegFormer(), LABELS).segment(_sample_room())
        back = [r for r in res.regions if r.name == "Back wall"][0]
        alpha = res.selection_alpha([back.id])
        out = recolour(res.image, alpha, "#1A365D")
        untouched = alpha == 0
        self.assertTrue(np.array_equal(out[untouched], res.image[untouched]))
        core = back.mask & (alpha > 0.99)
        y_in = srgb_to_linear(res.image[core] / 255.0) @ np.array([0.2126, 0.7152, 0.0722])
        y_out = srgb_to_linear(out[core] / 255.0) @ np.array([0.2126, 0.7152, 0.0722])
        self.assertGreater(np.corrcoef(y_in, y_out)[0, 1], 0.9)    # same light pattern
        median = np.median(out[core], axis=0)
        self.assertLess(np.abs(median - np.array([0x1A, 0x36, 0x5D])).max(), 12)  # true paint colour where typically lit

    def test_recommendations_are_varied_and_explained(self):
        from src.colour_studio.recommend import delta_e, hex_to_rgb, recommend_shades, rgb_to_lab, room_palette
        from src.colour_studio.segmentation import WallSegmenter

        res = WallSegmenter(FakeSegFormer(), LABELS).segment(_sample_room())
        pal = room_palette(res.image, res.label_map, res.id2label, res.wall_mask)
        self.assertTrue(any("wooden floor" == p["name"] for p in pal))
        recs = recommend_shades(pal, None, "calm bedroom", limit=5)
        self.assertEqual(len(recs), 5)
        self.assertTrue(all(r["reason"] for r in recs))
        labs = [rgb_to_lab(hex_to_rgb(r["hex_code"]))[0] for r in recs]
        self.assertTrue(all(delta_e(a, b) > 15 for i, a in enumerate(labs) for b in labs[i + 1:]))

    def test_paint_quantity(self):
        from src.colour_studio.quantity import best_packs, litres_needed

        need = litres_needed(12, 10, 2, "Interior Emulsion", coverage=100)
        self.assertAlmostEqual(need["litres"], 2.4)
        packs = [{"size_litres": 1, "price": 580, "stock": 10}, {"size_litres": 4, "price": 2233, "stock": 10}]
        plan = best_packs(need["litres"], packs)
        self.assertGreaterEqual(plan["total_litres"], 2.4)
        self.assertEqual(plan["total_price"], 3 * 580)
        self.assertIsNone(best_packs(10, [{"size_litres": 1, "price": 580, "stock": 2}]))

    def test_catalog_search(self):
        from src.colour_assistant import find_nearest_shades_by_hex, search_shades_by_query

        self.assertEqual(search_shades_by_query("royale blue", 1)[0]["shade_name"], "Royale Blue")
        self.assertEqual(find_nearest_shades_by_hex("#1A365D", 1)[0]["shade_name"], "Royale Blue")


# --------------------------------------------------------------------- Layer 7
class TestLayer7Aura(unittest.TestCase):
    def test_offline_routing(self):
        from src.ai_assistant import deterministic_assistant_response as ask

        low = ask("Which shades are running low?")
        self.assertTrue("should be ordered" in low or "Nothing needs ordering" in low)
        self.assertIn("Today", ask("How did we do today?"))
        self.assertIn("Royale Blue", ask("Do we have Royale Blue?"))
        self.assertIn("Deal Checker", ask("Is the Royale Blue deal worth it?"))
        self.assertIn("Colour Studio", ask("Suggest a calm bedroom colour"))


if __name__ == "__main__":
    unittest.main()

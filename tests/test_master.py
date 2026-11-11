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
from src.config import SAMPLE_DB_PATH as DB_PATH, SAMPLE_ROOMS_DIR, SHADES_CATALOG_PATH


_DB_PATCH = None


def setUpModule():
    """Work on a copy of the sample database, moved forward to today like the app does."""
    global DB_PATH, _DB_PATCH
    tmp = Path(tempfile.mkdtemp()) / "paint_shop.db"
    shutil.copy(DB_PATH, tmp)
    DB_PATH = tmp
    import src.colour_assistant as ca
    import src.config as cfg
    import src.predictor as predictor
    import src.sales_forecast as sf
    import src.shop_insights as si
    import src.smart_alerts as sa

    _DB_PATCH = [mock.patch.object(m, "DB_PATH", tmp) for m in (db, cfg, ca, predictor, sf, si, sa)]
    for p in _DB_PATCH:
        p.start()
    db.ensure_app_tables()
    db.ensure_demo_dates_current()


def tearDownModule():
    for p in _DB_PATCH or []:
        p.stop()


def _sample_room(name="master_bedroom.jpg"):
    import cv2

    return cv2.cvtColor(cv2.imread(str(SAMPLE_ROOMS_DIR / name)), cv2.COLOR_BGR2RGB)


import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent))
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from fake_wall_model import LABELS, FakeSegFormer, iou  # noqa: E402
import make_sample_rooms as rooms  # noqa: E402


def _segment(name="master_bedroom.jpg", **kw):
    from src.colour_studio.segmentation import WallSegmenter

    img, lab = rooms.render(name)
    fake = FakeSegFormer(img, lab, **kw)
    res = WallSegmenter(fake, LABELS).segment(img)
    H, W = res.image.shape[:2]
    truth = cv2.resize((lab == 0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    return res, truth, fake


import cv2  # noqa: E402


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
                raw = c.execute("SELECT SUM(count_in) FROM footfall_logs WHERE DATE(timestamp) = ? AND timestamp <= ?",
                                (r.day, clock.now_naive().strftime("%Y-%m-%d %H:%M:%S"))).fetchone()[0]
                self.assertEqual(int(r.visitors), int(raw))

    def test_nothing_after_now_counts(self):
        fake_now = datetime.combine(clock.today(), datetime.min.time()).replace(hour=12, minute=30)
        with mock.patch.object(db.clock, "now_naive", return_value=fake_now), \
                mock.patch.object(db.clock, "now", return_value=fake_now.replace(tzinfo=clock.STORE_TZ)):
            hourly = db.get_hourly_traffic()
            self.assertLessEqual(hourly["hour"].max(), 12)
            s = db.get_sales_summary()
            with sqlite3.connect(DB_PATH) as c:
                late = c.execute("SELECT COUNT(DISTINCT invoice_no) FROM sales WHERE DATE(sale_timestamp) = ? "
                                 "AND sale_timestamp <= ?", (clock.today_str(), fake_now.strftime("%Y-%m-%d %H:%M:%S"))).fetchone()[0]
            self.assertEqual(s["today_invoices"], late)

    def test_camera_counts_exclude_sample_visitors(self):
        tmp = Path(tempfile.mkdtemp()) / "shop.db"
        shutil.copy(DB_PATH, tmp)
        with mock.patch.object(db, "DB_PATH", tmp):
            db.ensure_app_tables()
            self.assertEqual(db.get_door_counts_today(), {"in": 0, "out": 0, "inside": 0})
            db.log_footfall(3, 1)
            self.assertEqual(db.get_door_counts_today(), {"in": 3, "out": 1, "inside": 2})

    def test_season_status_uses_today(self):
        st = db.get_season_status(date(2026, 11, 7))
        self.assertTrue(any("Diwali" in e["event_name"] for e in st["active"]))
        self.assertFalse(any("Monsoon" in e["event_name"] for e in st["active"]))


# --------------------------------------------------------------------- Layer 3
class TestLiveDatabase(unittest.TestCase):
    """The app works on a live copy; a redeploy that replaces the sample database must not break or wipe it."""

    def test_live_copy_is_created_and_survives_a_new_sample(self):
        import src.config as cfg

        d = Path(tempfile.mkdtemp())
        sample, live = d / "paint_shop.db", d / "live" / "paint_shop.db"
        shutil.copy(DB_PATH, sample)
        with mock.patch.object(db, "SAMPLE_DB_PATH", sample), mock.patch.object(db, "DB_PATH", live), \
                mock.patch.object(cfg, "DB_PATH", live):
            self.assertTrue(db.ensure_live_db())                     # first start: copied from the sample
            self.assertFalse(db.ensure_live_db())                    # next runs: nothing to do
            db.log_footfall(2, 0)                                    # camera data in the live copy
            with sqlite3.connect(live) as c:                          # a redeploy with a half-prepared database
                c.execute("DROP TABLE alert_actions;")
            self.assertTrue(db.ensure_live_db())                     # repaired, not crashed
            self.assertEqual(db.get_hidden_alert_ids(), set())
            self.assertEqual(db.get_door_counts_today()["in"], 2)   # camera data kept
            with sqlite3.connect(sample) as c:                        # new sample data pushed (demo mode)
                c.execute("INSERT OR REPLACE INTO app_settings (key, value) VALUES ('seed_id', '\"new\"');")
            with mock.patch.object(db, "DEMO_MODE", True):
                self.assertTrue(db.ensure_live_db())                 # live copy refreshed to the new sample


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
        self.assertEqual(set(a), {"critical", "expiring", "dead_stock", "anomalies", "camera"})
        for x in a["expiring"]:
            self.assertIn("expire", x["title"])
        for x in a["anomalies"]:
            self.assertLess(int(x["description"].split(" visitors")[0]), 1000)  # the old bug said 11,130
        flat = flatten_alerts(a)
        ids = [alert_id(x) for x in flat]
        self.assertEqual(len(ids), len(set(ids)))


# --------------------------------------------------------------------- Layer 5
class TestLayer5Camera(unittest.TestCase):
    def test_demo_counts_are_tagged_demo(self):
        import src.camera_stream as cs

        t = [1000.0]
        with mock.patch.object(cs.time, "time", side_effect=lambda: t[0]), \
                mock.patch.object(cs, "log_footfall") as lf, mock.patch.object(cs, "log_security_event") as ls, \
                mock.patch.object(cs, "log_visit"), mock.patch.object(cs, "log_shelf_event"), \
                mock.patch.object(cs, "set_shelf_status"), \
                mock.patch.object(cs.StoreVisionEngine, "_save_snapshot", lambda *a, **k: "x.jpg"):
            eng = cs.StoreVisionEngine("demo")
            eng._demo_t0 = 1000.0
            self.assertTrue(eng.open_stream())
            for _ in range(60):  # 30 seconds of demo
                frame, m = eng.process_frame()
                t[0] += 0.5
            self.assertGreaterEqual(m["entered"], 1)
            self.assertGreaterEqual(m["exited"], 1)
            self.assertTrue(lf.called)
            self.assertTrue(all(c.kwargs.get("source") == "demo" for c in lf.call_args_list))
            self.assertTrue(all(c.kwargs.get("source") == "demo" for c in ls.call_args_list))

    def test_standing_on_the_line_is_not_counted_again_and_again(self):
        from src.camera_stream import StoreVisionEngine

        eng = StoreVisionEngine("webcam", log_to_db=False)
        eng.layout["count_mode"] = "door"
        h, line = 480, int(0.68 * 480)
        t = 0.0
        # Someone hovering right on the door line for 40 frames: no counts
        for i in range(40):
            y_feet = line + (12 if i % 2 else -12)
            e, x = eng._update_tracks([(300, int(y_feet - 0.9 * 150), 60, 150)], line, t, h)
            self.assertEqual((e, x), (0, 0))
            t += 0.12
        # One clean walk in, then out: exactly one each way
        eng2 = StoreVisionEngine("webcam", log_to_db=False)
        eng2.layout["count_mode"] = "door"
        totals = [0, 0]
        for y_feet in list(range(line - 120, line + 120, 15)) + list(range(line + 120, line - 120, -15)):
            e, x = eng2._update_tracks([(300, int(y_feet - 0.9 * 150), 60, 150)], line, t, h)
            totals[0] += e
            totals[1] += x
            t += 0.12
        self.assertEqual(totals, [1, 1])

    def test_box_cut_off_by_frame_bottom_is_not_counted(self):
        from src.camera_stream import StoreVisionEngine

        eng = StoreVisionEngine("webcam", log_to_db=False)
        eng.layout["count_mode"] = "door"
        h, line = 480, int(0.68 * 480)
        total = 0
        for i, top in enumerate(range(0, 300, 10)):        # someone close to the camera, feet out of view
            e, x = eng._update_tracks([(200, top, 200, h - top)], line, i * 0.1, h)
            total += e + x
        self.assertEqual(total, 0)

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
    def test_walls_are_accurate_despite_model_mistakes(self):
        # The stand-in model makes the real model's mistakes: bites at lamp glow,
        # holes, leaks onto the headboard/sofa, ragged blotches, coarse edges.
        for name in rooms.ROOMS:
            res, truth, fake = _segment(name)
            raw = cv2.resize((fake.pred == 0).astype(np.uint8), truth.shape[::-1],
                             interpolation=cv2.INTER_NEAREST).astype(bool)
            self.assertGreater(iou(res.wall_mask, truth), 0.95, name)
            self.assertGreaterEqual(iou(res.wall_mask, truth), iou(raw, truth) - 0.005, name)

    def test_furniture_and_objects_stay_unpainted(self):
        for name, cls in [("master_bedroom.jpg", 7), ("modern_living_room.jpg", 23), ("master_bedroom.jpg", 14),
                          ("master_bedroom.jpg", 22)]:
            res, truth, _ = _segment(name)
            img, lab = rooms.render(name)
            obj = cv2.resize((lab == cls).astype(np.uint8), truth.shape[::-1], interpolation=cv2.INTER_NEAREST).astype(bool)
            self.assertLess((res.wall_mask & obj).sum() / obj.sum(), 0.02, f"{name} class {cls}")

    def test_three_named_walls_whatever_the_framing(self):
        from src.colour_studio.segmentation import WallSegmenter

        for name in rooms.ROOMS:
            res, _, _ = _segment(name)
            self.assertEqual([r.name for r in res.regions], ["Left wall", "Back wall", "Right wall"], name)
        img, lab = rooms.render("master_bedroom.jpg")
        im, lb = np.ascontiguousarray(img[:, 380:]), np.ascontiguousarray(lab[:, 380:])  # back wall left of centre
        res = WallSegmenter(FakeSegFormer(im, lb), LABELS).segment(im)
        self.assertEqual([r.name for r in res.regions], ["Back wall", "Right wall"])

    def test_model_sees_photo_shape_and_falls_back_to_square(self):
        res, _, fake = _segment()
        self.assertTrue(all(h != w for h, w in fake.calls))          # keeps the 3:2 shape
        self.assertEqual(len(fake.calls), 2)                         # normal + mirrored pass
        res2, truth, fake2 = _segment(square_only=True)               # older model files
        self.assertTrue(any(h == w for h, w in fake2.calls))
        self.assertGreater(iou(res2.wall_mask, truth), 0.9)

    def test_tap_mode_separates_walls(self):
        from src.colour_studio.segmentation import tap_result

        for room in rooms.ROOMS:
            res = tap_result(_sample_room(room), [(0.38, 0.45), (0.05, 0.12), (0.95, 0.12)])  # on bare wall
            _, lab = rooms.render(room)
            truth = cv2.resize((lab == 0).astype(np.uint8), res.wall_mask.shape[::-1],
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            self.assertGreaterEqual(len(res.regions), 2, room)
            for r in res.regions:   # never paints the ceiling, furniture or windows
                self.assertGreater((r.mask & truth).sum() / r.mask.sum(), 0.85, f"{room} {r.name}")
            self.assertGreater(iou(res.wall_mask, truth), 0.8, room)

    def test_manual_fixes_add_and_remove_spots(self):
        from src.colour_studio.segmentation import apply_fixes

        res, _, _ = _segment()
        back = [r.id for r in res.regions if r.name == "Back wall"]
        size = lambda r: [x for x in r.regions if x.name == "Back wall"][0].mask.sum()
        added = apply_fixes(res, [("add", 0.5, 0.56)], back)            # a strip of the headboard
        self.assertGreater(size(added), size(res))
        self.assertLess(size(added) - size(res), 0.04 * res.wall_mask.size)   # a spot, not a wall
        undone = apply_fixes(res, [("add", 0.5, 0.56), ("remove", 0.5, 0.56)], back)
        self.assertLess(abs(int(size(undone)) - int(size(res))), 200)

    def test_recolour_keeps_light_and_leaves_rest_alone(self):
        from src.colour_studio.recolour import recolour, srgb_to_linear

        res, _, _ = _segment(mistakes=False)
        back = [r for r in res.regions if r.name == "Back wall"][0]
        alpha = res.selection_alpha([back.id])
        out = recolour(res.image, alpha, "#1A365D")
        untouched = alpha == 0
        self.assertTrue(np.array_equal(out[untouched], res.image[untouched]))
        core = back.mask & (alpha > 0.99)
        w = np.array([0.2126, 0.7152, 0.0722])
        y_in = srgb_to_linear(res.image[core] / 255.0) @ w
        y_out = srgb_to_linear(out[core] / 255.0) @ w
        self.assertGreater(np.corrcoef(y_in, y_out)[0, 1], 0.9)    # same light pattern
        median = np.median(out[core], axis=0)
        self.assertLess(np.abs(median - np.array([0x1A, 0x36, 0x5D])).max(), 14)  # true paint colour where typically lit

    def test_recommendations_are_varied_and_explained(self):
        from src.colour_studio.recommend import delta_e, hex_to_rgb, recommend_shades, rgb_to_lab, room_palette

        res, _, _ = _segment(mistakes=False)
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

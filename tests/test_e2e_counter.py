"""
End to end, without a browser: scripted scenes go through the real camera code
(people detection, tracking, uniform check, visits) and the real billing code;
the numbers the owner sees must match what happened in the scene.
"""
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from src import billing, clock
from src import config as cfg
from src import database as db
from tests.people_scene import background, scene

ORANGE = (0, 120, 245)        # the shop uniform (BGR)
UNIFORM_HEX = "#f57800"
RED, GREEN, NAVY = (40, 40, 200), (60, 160, 60), (90, 40, 30)
BLUE, YELLOW, PURPLE = (200, 110, 30), (40, 200, 230), (150, 50, 140)
FPS = 2.0                       # what the browser camera sends on Streamlit Cloud

# --- 1. A counter camera (people in view): a buyer, a staff member, someone just asking, a passer-by
COUNTER = {
    "mode": "presence", "end": 60,
    "cast": [
        ("customer_a", RED,    [(2, .30, .93, .62), (4, .38, .93, .64), (26, .38, .93, .64), (28, .25, .93, .62)]),
        ("staff",      ORANGE, [(8, .72, .92, .60), (10, .62, .92, .62), (40, .62, .92, .62), (42, .75, .92, .60)]),
        ("customer_b", GREEN,  [(32, .20, .93, .62), (34, .30, .93, .63), (44, .30, .93, .63), (46, .20, .93, .62)]),
        ("passer_by",  NAVY,   [(50, .05, .93, .60), (50.8, .25, .93, .60)]),      # walks past in under a second
    ],
    "bills": [(20, 2)],                          # (second, cans) — customer A pays
    "no_sales": [(40, "just_asking")],           # customer B
}

# --- 2. Busy counter: a couple pays while a stranger who came in later stands at the counter too
BUSY = {
    "mode": "presence", "end": 110,
    "cast": [
        ("couple_1", BLUE,   [(2, .12, .93, .62), (4, .20, .93, .63), (58, .20, .93, .63), (60, .08, .93, .62)]),
        ("couple_2", YELLOW, [(3, .25, .93, .60), (5, .36, .93, .61), (58, .36, .93, .61), (60, .22, .93, .60)]),
        ("stranger", PURPLE, [(40, .85, .93, .62), (42, .70, .93, .63), (98, .70, .93, .63), (100, .85, .93, .62)]),
    ],
    "bills": [(52, 3)],
    "no_sales": [(95, "price")],
}

# --- 3. A doorway camera (door line): walk in, pay at the counter, walk out; staff walk through; a browser
DOOR = {
    "mode": "door", "end": 42,
    "cast": [
        ("customer_a", RED,    [(0, .55, .50, .40), (3, .55, .82, .56), (5, .22, .86, .58), (15, .22, .86, .58),
                                (17, .55, .82, .56), (20, .55, .50, .40)]),
        ("staff",      ORANGE, [(6, .82, .50, .40), (9, .82, .85, .58), (14, .82, .85, .58), (17, .82, .50, .40)]),
        ("customer_b", GREEN,  [(22, .60, .50, .40), (25, .62, .82, .56), (30, .62, .82, .56), (33, .60, .50, .40)]),
    ],
    "bills": [(10, 1)],
    "no_sales": [(28, "just_asking")],
}


class FakeTime:
    def __init__(self, t0):
        self.t = t0

    def time(self):
        return self.t

    def sleep(self, s):
        self.t += s


def run(tmp, spec, uniform_on=True, frames=None):
    """Play a scene through the camera at the browser's frame rate, with bills and didn't-buys at their seconds."""
    from src import camera_stream as cs
    from src import shop_insights as si

    start = datetime.combine(clock.today(), datetime.min.time()).replace(hour=11)
    ft = FakeTime(1_000_000.0)
    at = lambda: start + timedelta(seconds=ft.t - 1_000_000.0)
    bg = background()
    bills = []
    with mock.patch.object(cs, "time", ft), mock.patch.object(clock, "now", side_effect=at), \
            mock.patch.object(clock, "now_naive", side_effect=at):
        eng = cs.StoreVisionEngine("webcam", log_to_db=True)
        lay = dict(eng.layout, count_mode=spec["mode"], shelves=[])
        if uniform_on:
            lay["uniform"] = {"on": True, "hex": UNIFORM_HEX}
        eng.set_layout(lay)
        pid = int(sqlite3.connect(tmp).execute(
            "SELECT product_id FROM inventory GROUP BY product_id HAVING SUM(current_stock) > 10 LIMIT 1").fetchone()[0])
        todo_b, todo_n = list(spec["bills"]), list(spec["no_sales"])
        t = 0.0
        while t <= spec["end"] + 6:                   # a few seconds more: the last person is seen to leave
            frame, _ = scene(t, spec["cast"], bg=bg)
            out, _ = eng.process_frame(frame)
            if frames is not None:
                frames.append(out)
            while todo_b and t >= todo_b[0][0]:
                _, qty = todo_b.pop(0)
                bills.append(billing.create_bill([{"product_id": pid, "qty": qty}], payment_mode="UPI", at=at()))
            while todo_n and t >= todo_n[0][0]:
                billing.record_no_sale(todo_n.pop(0)[1], at=at())
            t += 1.0 / FPS
            ft.t += 1.0 / FPS
        eng.close_stream()
        b = si.buyers("camera")
    counts = db.get_door_counts_today(source="camera")
    with sqlite3.connect(tmp) as c:
        visits = c.execute("SELECT entered_at, left_at, bill_no, no_sale_reason FROM visits WHERE source='camera' "
                           "ORDER BY entered_at").fetchall()
    return {"eng": eng, "buyers": b, "counts": counts, "bills": bills, "visits": visits}


class E2E(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "shop.db"
        shutil.copy(cfg.SAMPLE_DB_PATH, self.tmp)
        import src.colour_assistant as ca
        import src.predictor as predictor
        import src.sales_forecast as sf
        import src.shop_insights as si
        import src.smart_alerts as sa

        self.ps = [mock.patch.object(m, "DB_PATH", self.tmp) for m in (cfg, db, billing, ca, predictor, sf, si, sa)]
        [p.start() for p in self.ps]
        db.ensure_app_tables()
        with sqlite3.connect(self.tmp) as c:
            for tbl in ("visits", "footfall_logs"):
                c.execute(f"DELETE FROM {tbl} WHERE source = 'camera';")

    def tearDown(self):
        [p.stop() for p in self.ps]

    def outcome(self, r):
        return [(bill is not None, reason) for _, _, bill, reason in r["visits"]]


class TestCounterCamera(E2E):
    def test_buyer_staff_enquiry_passer_by(self):
        r = run(self.tmp, COUNTER)
        self.assertEqual((r["counts"]["in"], r["counts"]["out"]), (2, 2), "staff and the passer-by aren't visitors")
        self.assertEqual(len(r["eng"].staff_ids), 1, "the person in uniform is recognised once")
        self.assertEqual(self.outcome(r), [(True, None), (False, "just_asking")])
        self.assertEqual(r["visits"][0][2], r["bills"][0]["invoice_no"])
        b = r["buyers"]
        self.assertEqual((b["bought_visits"], b["didnt_buy"], b["reasons"].get("just_asking")), (1, 1, 1), b)

    def test_visit_times_are_when_they_were_there(self):
        r = run(self.tmp, COUNTER)
        (a_in, a_out, _, _), (b_in, b_out, _, _) = r["visits"]
        secs = lambda s: (datetime.strptime(s, "%Y-%m-%d %H:%M:%S") - datetime.strptime(a_in[:11] + "11:00:00",
                                                                                       "%Y-%m-%d %H:%M:%S")).seconds
        self.assertLessEqual(abs(secs(a_in) - 2), 2)
        self.assertLessEqual(abs(secs(a_out) - 28), 3)
        self.assertLessEqual(abs(secs(b_in) - 32), 2)
        self.assertLessEqual(abs(secs(b_out) - 46), 3)

    def test_without_the_uniform_setting_staff_count_as_visitors(self):
        r = run(self.tmp, COUNTER, uniform_on=False)
        self.assertEqual(r["counts"]["in"], 3)
        self.assertEqual(len(r["eng"].staff_ids), 0)


class TestUniformCapture(unittest.TestCase):
    def test_capture_reads_the_staff_shirt(self):
        import cv2
        import numpy as np
        from src import camera_stream as cs

        eng = cs.StoreVisionEngine("webcam", log_to_db=False)
        frame, _ = scene(20, [("staff", ORANGE, [(0, .5, .92, .62), (40, .5, .92, .62)])])
        eng.latest_raw = frame
        got = eng.capture_uniform()
        self.assertIsNotNone(got)
        h_got = cv2.cvtColor(np.uint8([[cs.hex_to_bgr(got)]]), cv2.COLOR_BGR2HSV)[0, 0, 0]
        h_ref = cv2.cvtColor(np.uint8([[ORANGE]]), cv2.COLOR_BGR2HSV)[0, 0, 0]
        self.assertLessEqual(abs(int(h_got) - int(h_ref)), 3, got)


class TestBusyCounter(E2E):
    def test_couple_shares_the_bill_stranger_did_not_buy(self):
        r = run(self.tmp, BUSY)
        self.assertEqual(r["counts"]["in"], 3)
        self.assertEqual(self.outcome(r), [(True, None), (True, None), (False, "price")])
        b = r["buyers"]
        self.assertEqual((b["bought_visits"], b["didnt_buy"]), (2, 1), b)


class TestDoorCamera(E2E):
    def test_walk_in_pay_walk_out(self):
        r = run(self.tmp, DOOR)
        self.assertEqual((r["counts"]["in"], r["counts"]["out"]), (2, 2), r["counts"])
        self.assertEqual(len(r["eng"].staff_ids), 1)
        self.assertEqual(self.outcome(r), [(True, None), (False, "just_asking")])
        self.assertEqual(r["buyers"]["bought_visits"], 1)


if __name__ == "__main__":
    unittest.main()

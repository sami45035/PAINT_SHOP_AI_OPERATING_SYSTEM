"""
AI sales predictor tests.

The sample sales were generated with known festival effects (data/seed_data.py,
REAL): waterproofing season ~2.3x for waterproofing, monsoon ~0.4x for exterior
paint, Diwali ~1.9x for emulsions (the calendar guesses 2.2x). The model must
recover them from the sales alone, and must beat the simple 30-day average on
periods it never saw.
"""
import math
import shutil
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

from src import config as cfg
from src import database as db

_PATCHES = []


def setUpModule():
    tmp = Path(tempfile.mkdtemp()) / "paint_shop.db"
    shutil.copy(cfg.SAMPLE_DB_PATH, tmp)
    import src.predictor as predictor
    import src.sales_forecast as sf

    import src.colour_assistant as ca
    import src.shop_insights as si
    import src.smart_alerts as sa

    import src.billing as billing

    for mod in (cfg, db, sf, predictor, ca, si, sa, billing):
        p = mock.patch.object(mod, "DB_PATH", tmp)
        p.start()
        _PATCHES.append(p)
    db.ensure_app_tables()
    db.ensure_demo_dates_current()


def tearDownModule():
    for p in _PATCHES:
        p.stop()


def _learned(o, festival, group):
    return next(x for x in o["learned"] if x["festival"] == festival and x["group"] == group)


class TestLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src import sales_forecast as sf

        cls.o = sf.outlook(30)

    def test_learns_real_festival_effects(self):
        wp = _learned(self.o, "Post-Monsoon Waterproofing", "Waterproofing")
        self.assertGreater(wp["days_seen"], 20)
        # truth ~2.3; a small category, so the estimate moves with the sample's daily noise (1.6-2.5 across build dates)
        self.assertTrue(1.5 <= wp["learned"] <= 2.9, wp)
        lull = _learned(self.o, "Monsoon Exterior Lull", "Exterior Emulsion")
        self.assertTrue(0.25 <= lull["learned"] <= 0.6, lull)           # truth ~0.4 (0.27-0.52 across build dates)
        diwali = _learned(self.o, "Diwali Pre-Festival Surge", "Interior Emulsion")
        if diwali["days_seen"] >= 14:
            self.assertTrue(1.3 <= diwali["learned"] < diwali["calendar"], diwali)   # truth ~1.9, calendar 2.2

    def test_unseen_festival_counts_at_half_strength(self):
        for x in self.o["learned"]:
            if x["days_seen"] == 0:
                self.assertAlmostEqual(x["learned"], round(math.sqrt(x["calendar"]), 2), places=2)

    def test_weekend_is_busier(self):
        dow = self.o["shop_dow"]
        self.assertGreater(dow[5], 1.1)                                # Saturday
        self.assertAlmostEqual(sum(dow) / 7, 1.0, places=6)

    def test_demand_drops_after_a_festival_ends(self):
        from src.sales_forecast import _event_factor

        model = self.o["model"]
        g = model.groups["Interior Emulsion"]
        today = self.o["today"]
        for e in g.events:
            ends_soon = today <= e["end"] < today + timedelta(days=25)
            if g.seen.get(e["name"], 0) and ends_soon and g.event_effect[e["name"]] > 1.1:
                before = model.group_daily("Interior Emulsion", [e["end"] - timedelta(days=i) for i in range(7)])
                after = model.group_daily("Interior Emulsion", [e["end"] + timedelta(days=i + 1) for i in range(7)])
                self.assertLess(after.sum(), before.sum() * 0.85)
                self.assertLess(_event_factor(g, e["end"] + timedelta(days=1)), g.event_effect[e["name"]])


class TestBacktest(unittest.TestCase):
    def test_beats_the_simple_average_on_unseen_periods(self):
        from src import sales_forecast as sf

        r = sf.rolling_backtest(14, 10)
        self.assertEqual(r["windows"], 10)
        # Checked on sample data built for 10 different dates (15-24 Nov):
        # per product group the AI's error is 0.48-0.80 of the simple average's, closer in 6-10 of 10 periods;
        # on the shop's total cans it is closer or level (both are near on flat weeks).
        self.assertLess(r["ai_group_err_pct"], 0.85 * r["simple_group_err_pct"])
        self.assertGreaterEqual(r["ai_group_wins"], 6)
        self.assertLess(r["ai_err_pct"], r["simple_err_pct"] + 1.5)
        for row in r["periods"].to_dict("records"):
            self.assertGreater(row["actual"], 0)

    def test_backtest_never_sees_the_future(self):
        from src import sales_forecast as sf

        bt = sf.backtest(28)
        self.assertEqual(len(bt["dates"]), 28)
        self.assertEqual(bt["dates"][0], bt["cutoff"])
        data = sf.load_history(bt["until"])
        model = sf.Forecaster.fit(data, bt["cutoff"])
        # refitting on a copy that also contains the test weeks must not change a model fitted at the cut-off
        data2 = sf.load_history(bt["until"] + timedelta(days=1))
        model2 = sf.Forecaster.fit(data2, bt["cutoff"])
        for g in model.groups:
            self.assertAlmostEqual(model.groups[g].level, model2.groups[g].level, places=6)


class TestReorderUsesTheModel(unittest.TestCase):
    def test_forecast_30d_matches_the_model(self):
        from src import clock
        from src import sales_forecast as sf
        from src.predictor import compute_sales_forecast

        fc = compute_sales_forecast(30)
        model = sf.Forecaster.fit(sf.load_history(clock.today()), clock.today())
        pred = model.forecast(30, clock.today() + timedelta(days=1))
        top = fc.sort_values("forecast_30d", ascending=False).head(5)
        for r in top.itertuples():
            expected = pred[pred["product_id"] == r.product_id]["units"].sum()
            self.assertEqual(r.forecast_30d, math.ceil(expected))


if __name__ == "__main__":
    unittest.main()

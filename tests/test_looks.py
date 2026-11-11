"""Designer looks (colour prediction): reading the room, scoring looks, matching paints."""
import unittest

import numpy as np

from src.colour_studio import palettes as pl
from src.colour_studio.recommend import hex_to_rgb, lab_to_lch, rgb_to_lab


def _item(name, hex_code, big=False):
    lab = rgb_to_lab(hex_to_rgb(hex_code))[0]
    return {"name": name, "rgb": hex_to_rgb(hex_code), "lab": lab, "lch": lab_to_lch(lab)[0], "share": 0.1,
            "is_big_furniture": big}


def _ctx(**kw):
    base = {"room": "bedroom", "undertone": "warm", "wood": True, "light": "normal", "big_furniture": [],
            "n_walls": 3, "palette": []}
    base.update(kw)
    return base


class TestLooks(unittest.TestCase):
    def test_every_look_is_a_full_scheme(self):
        ids = set()
        for lk in pl.looks():
            self.assertEqual(set(lk["paints"]), {"walls", "feature", "trim"})
            self.assertEqual(len(lk["accents"]), 2)
            self.assertTrue(set(lk["styles"]) <= set(pl.STYLES), lk["name"])
            # 60-30-10: the main walls are lighter than the feature wall
            self.assertGreater(pl._lrv(lk["paints"]["walls"]["hex"]), pl._lrv(lk["paints"]["feature"]["hex"]))
            ids.add(lk["id"])
        self.assertEqual(len(ids), len(pl.looks()))
        self.assertGreaterEqual(len(ids), 20)

    def test_style_chip_wins(self):
        for style in pl.STYLES:
            top = pl.recommend_looks(_ctx(), [style], "", limit=3)
            self.assertIn(style, top[0]["styles"], style)

    def test_dim_room_prefers_light_walls(self):
        top = pl.recommend_looks(_ctx(light="dim"), [], "", limit=4)
        self.assertTrue(all(lk["walls_lrv"] >= 70 for lk in top), [(lk["name"], lk["walls_lrv"]) for lk in top])

    def test_feature_wall_that_matches_the_sofa_is_marked_down(self):
        look = next(lk for lk in pl.looks() if lk["name"] == "Coastal Calm")
        sofa = _item("blue sofa", look["paints"]["feature"]["hex"], big=True)
        plain = pl.score_look(look, _ctx(), [], [])
        clash = pl.score_look(look, _ctx(big_furniture=[sofa]), [], [])
        self.assertLess(clash["score"], plain["score"])
        self.assertIn("disappear", clash["reason"])

    def test_varied_feature_walls(self):
        from src.colour_studio.recommend import delta_e

        top = pl.recommend_looks(_ctx(), [], "", limit=6)
        labs = [rgb_to_lab(hex_to_rgb(lk["paints"]["feature"]["hex"]))[0] for lk in top]
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                self.assertGreater(delta_e(labs[i], labs[j]), 12)

    def test_mood_words(self):
        self.assertEqual(pl.mood_words_from("cosy, relaxing festival vibes"), ["cozy", "calm", "festive"])

    def test_paint_matching_prefers_stock(self):
        stocked = [{"hex_code": "#9CAF88", "shade_name": "Sage Whisper", "brand": "Berger", "total_stock": 12}]
        self.assertEqual(pl.match_paint("Sage Leaf", "#9DAE89", stocked)["shade_name"], "Sage Whisper")
        out_of_stock = [{**stocked[0], "total_stock": 0}]
        house = pl.match_paint("Sage Leaf", "#9DAE89", out_of_stock)
        self.assertEqual(house["brand"], pl.HOUSE_BRAND)
        self.assertTrue(house["tinted"])
        self.assertTrue(house["shade_code"].startswith("APX-"))
        self.assertEqual(house["shade_code"], pl.house_shade("Sage Leaf", "#9DAE89")["shade_code"])   # stable

    def test_wall_plan(self):
        class R:
            def __init__(self, i, name, share):
                self.id, self.name, self.area_share = i, name, share

        look = pl.looks()[0]
        plan = pl.wall_plan([R(1, "Left wall", 0.2), R(2, "Back wall", 0.15), R(3, "Right wall", 0.3)], look)
        self.assertEqual(plan[2], look["paints"]["feature"]["hex"])
        self.assertEqual(plan[1], look["paints"]["walls"]["hex"])
        self.assertEqual(pl.wall_plan([R(1, "Wall", 0.4)], look), {1: look["paints"]["walls"]["hex"]})

    def test_room_reading(self):
        id2label = {0: "wall", 7: "bed", 23: "sofa"}
        lm = np.zeros((100, 100), np.int64)
        lm[60:90, 20:80] = 7
        self.assertEqual(pl.detect_room(lm, id2label), "bedroom")
        ctx = pl.room_context(np.zeros((10, 10, 3), np.uint8), [_item("wooden floor", "#A0704A")], None, lm, id2label,
                              3, "")
        self.assertEqual(ctx["undertone"], "warm")
        ctx2 = pl.room_context(np.zeros((10, 10, 3), np.uint8), [], None, lm, id2label, 3, "kids room")
        self.assertEqual(ctx2["room"], "kids")


if __name__ == "__main__":
    unittest.main()

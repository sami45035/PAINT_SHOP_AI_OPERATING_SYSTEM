"""Shade Match: colour maths, reading a sample from a photo, and matching it to the shop's shades."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import colour_match as cm
from src import config as cfg

TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp()) / "paint_shop.db"
    shutil.copy(cfg.SAMPLE_DB_PATH, TMP)


def tearDownModule():
    shutil.rmtree(TMP.parent, ignore_errors=True)


def scene(hex_code, light=(1.15, 1.0, 0.72), exposure=0.8, seed=1):
    """A counter photo: the sample in the middle, white paper on the right, warm shop light, noise and a glare spot."""
    rng = np.random.default_rng(seed)
    h, w = 600, 800
    img = np.full((h, w, 3), 0.25)
    img[130:470, 230:570] = cm.srgb_to_linear(np.array(cm.hex_to_rgb(hex_code), float))
    img[170:430, 600:770] = cm.PAPER_WHITE
    img = img * np.array(light) * exposure
    img *= 1 + rng.normal(0, 0.03, (h, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    img[(yy - 290) ** 2 + (xx - 380) ** 2 < 8 ** 2] = 1.0
    return np.clip(cm.linear_to_srgb(img), 0, 255).astype(np.uint8)


WHITE = (685 / 799, 300 / 599)


class TestColourMaths(unittest.TestCase):
    def test_ciede2000_matches_published_test_data(self):
        # Sharma, Wu & Dalal (2005), pairs 1, 2, 7, 17, 25, 33
        pairs = [((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
                 ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
                 ((50, 2.5, 0), (73, 25, -18), 27.1492),
                 ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
                 ((22.7233, 20.0904, -46.694), (23.0331, 14.973, -42.5619), 2.0373),
                 ((90.9257, -0.5406, -0.9208), (88.6381, -0.8985, -0.7239), 1.5381)]
        for a, b, want in pairs:
            self.assertAlmostEqual(float(cm.delta_e2000(np.array(a), np.array(b))), want, places=3)

    def test_round_trip_and_lab(self):
        for hx in ("#C85A32", "#1A365D", "#FFFFFF", "#000000"):
            rgb = cm.hex_to_rgb(hx)
            self.assertEqual(cm.rgb_to_hex(cm.linear_to_srgb(cm.srgb_to_linear(np.array(rgb, float)))), hx)
        L, a, b = cm.rgb_to_lab((255, 255, 255))
        self.assertAlmostEqual(L, 100, places=2)
        self.assertLess(abs(a) + abs(b), 0.05)

    def test_plain_names(self):
        self.assertEqual(cm.describe(cm.hex_to_rgb("#FFFFFF")), "white")
        self.assertEqual(cm.describe(cm.hex_to_rgb("#C85A32")), "terracotta")
        self.assertEqual(cm.describe(cm.hex_to_rgb("#D9B48C")), "beige")
        self.assertIn("navy", cm.describe(cm.hex_to_rgb("#1A365D")))
        self.assertIn("red", cm.describe(cm.hex_to_rgb("#C41E3A")))
        self.assertEqual(cm.nearest_name((0, 128, 128)), ("Teal", 0.0))
        self.assertEqual(cm.nearest_name(cm.hex_to_rgb("#483D8B"))[0], "Dark Slate Blue")

    def test_base_and_lrv(self):
        self.assertEqual(cm.base_number((255, 255, 255)), 1)
        self.assertEqual(cm.base_number(cm.hex_to_rgb("#1A365D")), 4)
        self.assertAlmostEqual(cm.lrv((255, 255, 255)), 100, places=1)
        self.assertLess(cm.lrv(cm.hex_to_rgb("#1A365D")), 10)


class TestReadingAPhoto(unittest.TestCase):
    def test_white_paper_corrects_the_shop_light(self):
        for hx in ("#C85A32", "#1A365D", "#D9B48C", "#9CAF88"):
            img = scene(hx)
            true = cm.rgb_to_lab(cm.hex_to_rgb(hx))
            raw = cm.sample_colour(img, 0.5, 0.5)
            fixed = cm.sample_colour(img, 0.5, 0.5, white=WHITE)
            self.assertTrue(fixed.corrected)
            self.assertGreater(float(cm.delta_e2000(cm.rgb_to_lab(raw.rgb), true)), 3, hx)    # the light shifts it
            self.assertLess(float(cm.delta_e2000(cm.rgb_to_lab(fixed.rgb), true)), 1.0, hx)  # paper fixes it
            self.assertEqual(fixed.warnings, [], hx)

    def test_glare_spot_is_ignored(self):
        img = scene("#1A365D")
        s = cm.sample_colour(img, 380 / 799, 290 / 599, white=WHITE)   # right on the glare
        self.assertLess(float(cm.delta_e2000(cm.rgb_to_lab(s.rgb), cm.rgb_to_lab(cm.hex_to_rgb("#1A365D")))), 2.5)

    def test_pattern_and_bad_white_are_flagged(self):
        img = np.zeros((300, 300, 3), np.uint8)
        img[:, ::4] = (220, 40, 40)
        img[:, 1::4] = (30, 160, 60)
        img[:, 2::4] = (240, 240, 240)
        self.assertTrue(any("more than one colour" in w for w in cm.sample_colour(img, 0.5, 0.5).warnings))
        dark = np.full((300, 300, 3), 20, np.uint8)
        s = cm.sample_colour(dark, 0.5, 0.5, white=(0.2, 0.2))
        self.assertFalse(s.corrected)
        self.assertTrue(any("very dark" in w for w in s.warnings))


class TestMatching(unittest.TestCase):
    def test_a_catalogue_shade_matches_itself(self):
        shades = cm.shop_shades(TMP)
        self.assertGreaterEqual(len(shades), 15)
        best = cm.match(cm.hex_to_rgb("#C85A32"), shades=shades)[0]
        self.assertEqual(best["name"], "Terracotta Sunset")
        self.assertEqual(best["verdict_key"], "exact")
        self.assertTrue(best["base_code"].endswith("3"))      # tinted on a Base 3
        self.assertGreater(best["stock"], 0)

    def test_verdicts(self):
        self.assertEqual(cm.verdict(0.5)[0], "exact")
        self.assertEqual(cm.verdict(4)[0], "close")
        self.assertEqual(cm.verdict(8)[0], "family")
        self.assertEqual(cm.verdict(20)[0], "none")
        far = cm.match((120, 0, 200), shades=cm.shop_shades(TMP))[0]   # a vivid purple nobody sells here
        self.assertEqual(far["verdict_key"], "none")

    def test_bases_in_stock(self):
        bases = cm.bases_in_stock(3, TMP)
        self.assertTrue(bases)
        self.assertTrue(all(b["code"].endswith("3") for b in bases))

    def test_fan_deck_upload(self):
        raw = ("Brand,Shade Code,Shade Name,Hex Code,Range\n"
               "Asian Paints,8173,Purple Dusk,#7800C8,Royale\n"
               "Asian Paints,9999,Broken,#XYZ,Royale\n").encode()
        rows, problems = cm.parse_fan_deck(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["hex"], "#7800C8")
        self.assertEqual(len(problems), 1)
        cm.save_fan_deck(rows, TMP)
        cm.save_fan_deck(rows, TMP)                              # loading twice doesn't duplicate
        self.assertEqual(cm.library_summary(TMP), [("Asian Paints", 1)])
        best = cm.match((120, 0, 200), shades=cm.shop_shades(TMP))[0]
        self.assertEqual((best["name"], best["source"], best["verdict_key"]), ("Purple Dusk", "fan deck", "exact"))
        cm.clear_fan_decks(TMP)
        self.assertEqual(cm.library_summary(TMP), [])
        self.assertEqual(cm.parse_fan_deck(b"a,b\n1,2\n")[0], [])


if __name__ == "__main__":
    unittest.main()

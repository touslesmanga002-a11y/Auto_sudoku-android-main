"""Tests sur des captures synthetiques (aucune image reelle necessaire)."""
import os
import sys
import unittest

# Bootstrap : voir test_solver.py (fonctionne sans PYTHONPATH manuel).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sudokubot import config, vision

try:
    from test_solver import P6, P9
except ModuleNotFoundError:  # ex. `pytest` avec rootdir differente
    from tests.test_solver import P6, P9


def font(size):
    for p in config.section("vision").get("system_fonts", []):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return None


def make(puzzle, br, bc, W=1080, H=2400, top=520, highlight_row=None):
    n = len(puzzle)
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    x0, side = 30, W - 60
    step = side / n
    if highlight_row is not None:
        d.rectangle([x0, top + step * highlight_row, x0 + side, top + step * (highlight_row + 1)], fill=(220, 232, 255))
    f = font(int(step * 0.62))
    for i in range(n + 1):
        thick_h = i % br == 0
        thick_v = i % bc == 0
        y = top + step * i
        x = x0 + step * i
        d.line([(x0, y), (x0 + side, y)], fill=(40, 40, 60) if thick_h else (195, 200, 210), width=6 if thick_h else 2)
        d.line([(x, top), (x, top + side)], fill=(40, 40, 60) if thick_v else (195, 200, 210), width=6 if thick_v else 2)
    if f:
        for r in range(n):
            for c in range(n):
                v = puzzle[r][c]
                if v:
                    cx, cy = x0 + step * (c + 0.5), top + step * (r + 0.5)
                    d.text((cx, cy), str(v), fill=(20, 20, 20), font=f, anchor="mm")
        ky = top + side + 300
        kstep = side / n
        for i in range(n):
            d.text((x0 + kstep * (i + 0.5), ky), str(i + 1), fill=(40, 90, 230), font=font(int(step * 0.7)), anchor="mm")
    return np.asarray(im, dtype=np.uint8)


class VisionTest(unittest.TestCase):
    def test_grid_9(self):
        rgb = make(P9, 3, 3)
        g = vision.detect_grid(rgb)
        self.assertIsNotNone(g)
        self.assertEqual(g["n"], 9)
        self.assertEqual((g["br"], g["bc"]), (3, 3))

    def test_grid_6_with_highlight(self):
        rgb = make(P6, 2, 3, highlight_row=2)
        g = vision.detect_grid(rgb)
        self.assertIsNotNone(g)
        self.assertEqual(g["n"], 6)
        self.assertEqual((g["br"], g["bc"]), (2, 3))

    def test_small_screen(self):
        rgb = make(P9, 3, 3, W=720, H=1600, top=350)
        g = vision.detect_grid(rgb)
        self.assertIsNotNone(g)
        self.assertEqual(g["n"], 9)

    def test_keypad_and_read(self):
        if font(40) is None:
            self.skipTest("aucune police disponible")
        rgb = make(P9, 3, 3)
        g = vision.detect_grid(rgb)
        t = vision.Templates()
        t.clear_src("learned")
        keys = vision.detect_keypad(rgb, g, t)
        self.assertIsNotNone(keys)
        self.assertEqual(len(keys), 9)
        vision.keypad_to_templates(keys, t, vision.symbols_for(9, config.section("vision")))
        cells = vision.read_cells(rgb, g, t)
        read = vision.cells_to_grid(cells)
        wrong = sum(1 for r in range(9) for c in range(9) if read[r][c] != P9[r][c])
        self.assertLessEqual(wrong, 2)


class InkColorTest(unittest.TestCase):
    """Ne confond pas l'encre rouge et le fond/cercle rouge (cf. screenshot
    grille_avec_chiffres_rouges : 4 rouge, 4 noir sur fond rose, 4/3 rose pale)."""

    def test_classify_measured_values(self):
        cl = vision.classify_color
        self.assertEqual(cl((190, 56, 96)), "error")    # 4 rouge vif mesure
        self.assertEqual(cl((232, 174, 188)), "error")  # 4/3 rose pale mesure
        self.assertEqual(cl((49, 50, 56)), "given")      # 4 noir sur rose mesure
        self.assertEqual(cl((45, 50, 56)), "given")      # 9 noir mesure
        self.assertEqual(cl((40, 89, 190)), "user")      # 9 bleu mesure

    def _patch(self, digit="4", ink=(0, 0, 0), bg=(255, 255, 255),
               ring=None, size=120):
        f = font(90)
        if f is None:
            self.skipTest("aucune police disponible")
        im = Image.new("RGB", (size, size), bg)
        d = ImageDraw.Draw(im)
        if ring:
            d.ellipse([8, 8, size - 8, size - 8], outline=ring, width=4)
        d.text((size // 2, size // 2), digit, fill=ink, font=f, anchor="mm")
        return np.asarray(im, dtype=np.uint8)

    def _color(self, patch):
        cfg = config.section("vision")
        a = vision.analyse_patch(patch, cfg, 100)
        self.assertIsNotNone(a)
        return a["color"]

    def test_black_with_red_ring_stays_given(self):
        # Chiffre noir "entoure en rouge" : l'encre reste noire -> given.
        self.assertEqual(self._color(self._patch(ring=(255, 0, 0))), "given")

    def test_black_on_pink_stays_given(self):
        self.assertEqual(self._color(self._patch(bg=(240, 136, 161))), "given")

    def test_red_ink_is_error(self):
        self.assertEqual(self._color(self._patch(ink=(220, 30, 40))), "error")

    def test_pale_red_ink_is_error(self):
        self.assertEqual(self._color(self._patch(ink=(232, 174, 188))), "error")


class KeypadOrderTest(unittest.TestCase):
    """L'etiquetage gauche->droite = 1..N est refuse si les gabarits disent
    autre chose (rangee partielle + blob parasite qui decalerait tout)."""

    def _strip(self, digits, size_w=60, size_h=80):
        f = font(50)
        if f is None:
            self.skipTest("aucune police disponible")
        im = Image.new("RGB", (size_w * len(digits), size_h), (255, 255, 255))
        d = ImageDraw.Draw(im)
        blobs = []
        for i, ch in enumerate(digits):
            x0 = i * size_w + 12
            d.text((i * size_w + size_w // 2, size_h // 2), ch,
                   fill=(40, 90, 230), font=f, anchor="mm")
            blobs.append({"x0": x0, "x1": x0 + 36, "y0": 14, "y1": 66,
                          "x": x0 + 18, "y": 40, "w": 36, "h": 52})
        return np.asarray(im, dtype=np.uint8), blobs

    def test_correct_order_accepted(self):
        rgb, blobs = self._strip("123456789")
        t = vision.Templates()
        t.clear_src("learned")
        cfg = config.section("vision")
        self.assertTrue(vision._keypad_verify_order(blobs, 9, t, cfg, rgb))

    def test_shifted_order_rejected(self):
        # '2'..'9','1' : l'ordre 1..9 est faux, les gabarits le voient.
        rgb, blobs = self._strip("234567891")
        t = vision.Templates()
        t.clear_src("learned")
        cfg = config.section("vision")
        self.assertFalse(vision._keypad_verify_order(blobs, 9, t, cfg, rgb))

    def test_single_row(self):
        ok = [{"y": 100.0, "h": 50.0} for _ in range(8)]
        self.assertTrue(vision._single_row(ok))
        mixed = ok + [{"y": 250.0, "h": 50.0}]
        self.assertFalse(vision._single_row(mixed))

    def test_roughly_regular_multi_gaps(self):
        # 6 touches, 3 trous (simples et doubles) : accepte.
        xs = (0, 116, 348, 464, 580, 812)
        blobs = [{"x": float(x), "y": 0.0, "h": 50.0} for x in xs]
        self.assertTrue(vision._roughly_regular(blobs))

    def test_roughly_regular_random_rejected(self):
        xs = (0, 80, 250, 330, 700)
        blobs = [{"x": float(x), "y": 0.0, "h": 50.0} for x in xs]
        self.assertFalse(vision._roughly_regular(blobs))


class QuickToggleTest(unittest.TestCase):
    """Interrupteur 'placement rapide' : bleu = actif (a eteindre)."""

    def _shot(self, on, W=200, H=200):
        img = np.full((H, W, 3), 255, dtype=np.uint8)
        # pilule en bas-droite, comme sous la grille (x 0.93..0.99)
        x0, x1, y0, y1 = 186, 197, 60, 80
        img[y0:y1, x0:x1] = (30, 144, 255) if on else (200, 200, 200)
        return img

    def _grid(self):
        return {"x1": 196.0, "y1": 40.0, "ch": 60.0, "n": 9}

    def test_detect_on(self):
        st = vision.detect_quick_toggle(self._shot(True), self._grid())
        self.assertIsNotNone(st)
        self.assertTrue(st["active"])
        self.assertGreater(st["frac"], 0.03)
        self.assertTrue(150.0 < st["x"] < 200.0 and 40.0 < st["y"] < 94.0)

    def test_detect_off(self):
        st = vision.detect_quick_toggle(self._shot(False), self._grid())
        self.assertIsNotNone(st)
        self.assertFalse(st["active"])

    def test_no_grid_no_crash(self):
        self.assertIsNone(vision.detect_quick_toggle(
            np.zeros((10, 10, 3), dtype=np.uint8), self._grid()))


if __name__ == "__main__":
    unittest.main()

"""Tests des emplacements memorises (aucun telephone requis)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

# Bootstrap : voir test_solver.py.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sudokubot import config, spots, vision

try:
    from test_popups import FakeLink, make_ctrl
except ModuleNotFoundError:
    from tests.test_popups import FakeLink, make_ctrl
try:
    from test_solver import P9
except ModuleNotFoundError:
    from tests.test_solver import P9


def font(size):
    for p in config.section("vision").get("system_fonts", []):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return None


def make_game(W=540, H=900, top=200):
    """Mini capture de reference : grille 9x9 + pave complet + toggle gris."""
    n = 9
    im = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    x0, side = 15, W - 30
    step = side / n
    f = font(int(step * 0.62))
    for i in range(n + 1):
        w = 4 if i % 3 == 0 else 1
        y, x = top + step * i, x0 + step * i
        d.line([(x0, y), (x0 + side, y)], fill=(40, 40, 60), width=w)
        d.line([(x, top), (x, top + side)], fill=(40, 40, 60), width=w)
    if f:
        for i in range(n):
            d.text((x0 + step * (i + 0.5), top + side + 120), str(i + 1),
                   fill=(40, 90, 230), font=f, anchor="mm")
    d.rounded_rectangle([W - 70, top + side + 20, W - 10, top + side + 55],
                        radius=15, fill=(200, 200, 200))
    return np.asarray(im, dtype=np.uint8)


class SpotsPureTest(unittest.TestCase):
    def test_box_roundtrip(self):
        box = {"x": 0.5, "y": 0.6, "w": 0.2, "h": 0.1}
        px = spots.box_frac_to_px(box, 1000, 2000)
        self.assertEqual([round(v) for v in px], [400, 1100, 600, 1300])

    def test_grid_reconstruct(self):
        g = spots.grid_frac_to_px({"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0,
                                   "n": 9, "br": 3, "bc": 3}, 900, 900)
        self.assertEqual(len(g["xs"]), 10)
        self.assertAlmostEqual(g["cw"], 100.0)
        self.assertEqual(g["box_src"], "spots")

    def test_slot_ink(self):
        full = np.full((40, 40, 3), 255, dtype=np.uint8)
        full[10:30, 15:25] = (43, 88, 179)
        empty = np.full((40, 40, 3), 255, dtype=np.uint8)
        self.assertGreater(spots.slot_ink_frac(full, [0, 0, 40, 40]), 0.05)
        self.assertLess(spots.slot_ink_frac(empty, [0, 0, 40, 40]), 0.01)

    def test_present_slots(self):
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        img[20:60, 20:60] = (43, 88, 179)  # slot 1 occupe
        slots = [{"v": 1, "x": 40 / 300, "y": 40 / 100, "w": 40 / 300, "h": 40 / 100},
                 {"v": 2, "x": 150 / 300, "y": 40 / 100, "w": 40 / 300, "h": 40 / 100}]
        out = spots.present_slots(img, slots, 300, 100)
        self.assertEqual([k["v"] for k in out], [1])

    def test_lattice_present(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8) + 255
        for x in range(0, 200, 20):
            img[:, x - 1:x + 2] = 30
        for y in range(0, 200, 20):
            img[y - 1:y + 2, :] = 30
        g = {"xs": [float(x) for x in range(0, 200, 20)],
             "ys": [float(y) for y in range(0, 200, 20)],
             "x0": 0, "y0": 0, "x1": 199, "y1": 199}
        self.assertTrue(spots.lattice_present(img, g))
        blank = np.full((200, 200, 3), 255, dtype=np.uint8)
        self.assertFalse(spots.lattice_present(blank, g))

    def test_pick_hit_accents(self):
        items = [{"text": "Débutant Lv1", "b": [100, 500, 400, 560]},
                 {"text": "Facile", "b": [100, 600, 400, 660]}]
        hit = spots._pick_hit(items, ["debutant"], 1080, 2400)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], 250 / 1080)

    def test_pick_hit_split_blocks(self):
        # L'OCR coupe « Astuce » / « intelligente » : boite union.
        items = [{"text": "Astuce", "b": [700, 1640, 800, 1680]},
                 {"text": "intelligente", "b": [810, 1640, 980, 1680]},
                 {"text": "Crayon", "b": [500, 1640, 600, 1680]}]
        hit = spots._pick_hit(items, ["astuce intelligente"], 1080, 2400)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], (700 + 980) / 2 / 1080)
        self.assertAlmostEqual(hit["w"], (980 - 700) / 1080)

    def test_pick_hit_other_line_ignored(self):
        # Meme mots mais pas sur la meme ligne : pas de fusion abusive.
        items = [{"text": "Astuce", "b": [700, 1640, 800, 1680]},
                 {"text": "intelligente", "b": [100, 2000, 300, 2040]}]
        hit = spots._pick_hit(items, ["astuce intelligente"], 1080, 2400)
        # repli mot distinctif : la boite « Astuce » seule
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], 750 / 1080)

    def test_pick_hit_single_word_fallback(self):
        items = [{"text": "Astuce", "b": [700, 1640, 800, 1680]}]
        hit = spots._pick_hit(items, ["astuce intelligente"], 1080, 2400)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit["x"], 750 / 1080)

    def test_pick_hit_none(self):
        items = [{"text": "Crayon", "b": [100, 100, 200, 140]}]
        self.assertIsNone(spots._pick_hit(items, ["astuce intelligente"], 1080, 2400))

    def test_sanitize(self):
        raw = {"game": {"grid": {"x0": 0, "y0": 0, "x1": 2, "y1": 1, "n": 9},
                        "keys": [{"v": 1, "x": 0.1, "y": 0.8, "w": 0.05, "h": 0.05},
                                 {"v": "x"}],
                        "toggle": {"x": 0.9, "y": 0.7, "w": 0.05, "h": 0.03},
                        "nope": {"x": 0.5}},
                 "zzz": {"a": 1}}
        clean = spots.sanitize(raw)
        self.assertEqual(clean["game"]["grid"]["x1"], 1.0)  # borne
        self.assertEqual(len(clean["game"]["keys"]), 1)
        self.assertIn("toggle", clean["game"])
        self.assertNotIn("nope", clean["game"])
        self.assertNotIn("zzz", clean)

    def test_sanitize_vieux_toggle_point(self):
        # Ancien format (point seul) : complete avec les dimensions par
        # defaut au lieu de jeter l'emplacement.
        clean = spots.sanitize({"game": {"toggle": {"x": 0.9, "y": 0.7}}})
        self.assertIn("toggle", clean["game"])
        self.assertTrue(all(k in clean["game"]["toggle"] for k in ("x", "y", "w", "h")))

    def test_needs_upgrade(self):
        self.assertTrue(spots.needs_upgrade({"game": {"toggle": {"x": 0.9, "y": 0.7}}}))
        self.assertFalse(spots.needs_upgrade(
            {"game": {"toggle": {"x": 0.9, "y": 0.7, "w": 0.05, "h": 0.03}}}))
        self.assertFalse(spots.needs_upgrade({}))

    def test_needs_calibration(self):
        # Regression : un spots.json ne contenant que des boutons APPRIS
        # rafraichit son mtime -> is_stale dit faux alors que les hitboxes
        # manquent. needs_calibration regarde le CONTENU.
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "game.jpg")
            Image.new("RGB", (10, 10)).save(rp)
            refs = {"game": rp}
            self.assertTrue(spots.needs_calibration({}, refs))
            self.assertTrue(spots.needs_calibration(
                {"learned": {"win:ok": {"x": .1, "y": .1, "w": .1, "h": .1}}}, refs))
            ok = {"game": {"grid": {"x0": 0, "y0": 0, "x1": 1, "y1": 1,
                                     "n": 9, "br": 3, "bc": 3}},
                  "_meta": {"game": {"file": "game.jpg",
                                      "mtime": os.path.getmtime(rp)}}}
            self.assertFalse(spots.needs_calibration(ok, refs))
            # reference plus neuve que la calibration : recalibrer
            os.utime(rp, (9999999999, 9999999999))
            self.assertTrue(spots.needs_calibration(ok, refs))
            # sans reference : rien a calibrer
            self.assertFalse(spots.needs_calibration({}, {}))

    def test_is_stale_vieux_format(self):
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "game.jpg")
            op = os.path.join(d, "spots.json")
            Image.new("RGB", (10, 10)).save(rp)
            spots.save({"game": {"toggle": {"x": 0.9, "y": 0.7}}}, op)
            os.utime(op, (9999999999, 9999999999))  # json plus neuf, mais vieux format
            self.assertTrue(spots.is_stale({"game": rp}, op))

    def test_box_blue_frac(self):
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        img[20:50, 30:70] = (30, 144, 255)
        self.assertGreater(spots.box_blue_frac(img, [30, 20, 70, 50]), 0.5)
        gray = np.full((100, 100, 3), 200, dtype=np.uint8)
        self.assertLess(spots.box_blue_frac(gray, [30, 20, 70, 50]), 0.15)

    def test_is_stale_and_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "game.jpg")
            op = os.path.join(d, "spots.json")
            Image.new("RGB", (10, 10)).save(rp)
            self.assertTrue(spots.is_stale({"game": rp}, op))  # pas de sortie
            spots.save({"game": {}}, op)
            self.assertFalse(spots.is_stale({"game": rp}, op))
            os.utime(rp, (9999999999, 9999999999))
            self.assertTrue(spots.is_stale({"game": rp}, op))
            self.assertEqual(spots.load(op), {"game": {}})

    def test_find_refs_exts(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "game.png"), "w").close()
            open(os.path.join(d, "niveaux.jpeg"), "w").close()
            open(os.path.join(d, "notes.txt"), "w").close()
            refs = spots.find_refs(d)
            self.assertIn("game", refs)
            self.assertIn("niveaux", refs)
            self.assertNotIn("fin_partie", refs)


class CalibrateGameTest(unittest.TestCase):
    def test_game_reference(self):
        if font(90) is None:
            self.skipTest("aucune police disponible")
        rgb = make_game()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "game.jpg")
            Image.fromarray(rgb).save(p)
            out, (_r, W, H) = spots.calibrate_game(p, lambda m: None)
        self.assertEqual(out["grid"]["n"], 9)
        self.assertEqual(len(out["keys"]), 9)
        self.assertEqual(sorted(s["v"] for s in out["keys"]), list(range(1, 10)))
        self.assertIn("toggle", out)
        tg = out["toggle"]
        for k in ("x", "y", "w", "h"):
            self.assertIn(k, tg)
            self.assertTrue(0.0 < tg[k] < 1.0)
        # slots utilisables : tous pleins ici
        got = spots.present_slots(rgb, out["keys"], W, H)
        self.assertEqual(len(got), 9)


class ControllerSpotsTest(unittest.TestCase):
    def setUp(self):
        from sudokubot import controller as _c
        self._old_link = _c.LINK
        self.controller = _c

    def tearDown(self):
        self.controller.LINK = self._old_link

    def _ctrl(self, spot_data):
        self.controller.LINK = FakeLink()
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.spots = spot_data
        return c

    def _box(self, x=0.5, y=0.8, w=0.3, h=0.06):
        return {"x": x, "y": y, "w": w, "h": h}

    def test_start_new_game_spots_sans_ocr(self):
        c = self._ctrl({"fin_partie": {"nouvelle_partie": self._box()},
                        "niveaux": {"debutant": self._box(y=0.6)}})
        c._home_done = True
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c.shot = lambda stable=False: np.zeros((2400, 1080, 3), dtype=np.uint8)
        # verification grille : on simule une grille presente
        import sudokubot.vision as _v
        _orig = _v.detect_grid
        _v.detect_grid = lambda rgb, dead=0, cfg=None: {"n": 9}
        try:
            self.assertTrue(c._start_new_game())
        finally:
            # controller a importé le module, pas la fonction : ok
            _v.detect_grid = _orig
        link = self.controller.LINK
        self.assertEqual(link.find_calls, 0)  # aucun OCR, que des hitboxes
        mem = [l for l in list(c.logs) if "memorise" in l.get("msg", "")]
        self.assertTrue(mem)

    def test_start_new_game_premier_ocr(self):
        # Sans spots et 1er lancement : OCR (ancien code intact).
        self.controller.LINK = FakeLink(found_text="Nouvelle partie")
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c.spots = {}
        c.shot = lambda stable=False: np.zeros((2400, 1080, 3), dtype=np.uint8)
        import sudokubot.vision as _v
        _orig = _v.detect_grid
        _v.detect_grid = lambda rgb, dead=0, cfg=None: {"n": 9}
        try:
            self.assertTrue(c._start_new_game())
        finally:
            _v.detect_grid = _orig
        self.assertTrue(c._home_done)

    def test_spot_box_none_sans_spots(self):
        c = self._ctrl({})
        c.last_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        self.assertIsNone(c.spot_box("fin_partie", "nouvelle_partie"))

    def test_spots_grid_mauvaise_taille(self):
        c = self._ctrl({"game": {"grid": {"x0": 0, "y0": 0, "x1": 1, "y1": 1,
                                           "n": 4, "br": 2, "bc": 2}}})
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        self.assertIsNone(c.spots_grid(rgb, {"n": 9}))

    def test_auto_ignore_spots(self):
        c = self.controller.Controller({"auto": True})
        self.assertTrue(c.auto_spots)
        self.assertEqual(c.spots, {})


class SeqLink(FakeLink):
    """FakeLink qui repond a chaque find_any avec la reponse suivante
    (les ecrans changent entre deux recherches)."""

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def call(self, ctype, timeout=25.0, **params):
        if ctype == "find_any":
            self.find_calls += 1
            if self.responses:
                return self.responses.pop(0)
            return {"ok": True, "found": False, "hits": []}
        return FakeLink.call(self, ctype, timeout=timeout, **params)


class ExpressTest(unittest.TestCase):
    """Fin de partie express : verification puis enchainement DIRECT, replis."""

    def setUp(self):
        from sudokubot import controller as _c
        self._old_link = _c.LINK
        self.controller = _c

    def tearDown(self):
        self.controller.LINK = self._old_link

    def _box(self, x=0.5, y=0.8, w=0.3, h=0.06):
        return {"x": x, "y": y, "w": w, "h": h}

    def _ctrl(self):
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.countdown_sleep = lambda ms, what="Attend": None
        c._post_game = True
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        return c

    def _spots9(self):
        keys = [{"v": v + 1, "x": 0.06 + v * 0.11, "y": 0.84, "w": 0.07, "h": 0.05}
                for v in range(9)]
        return {"game": {"keys": keys},
                "fin_partie": {"nouvelle_partie": self._box()},
                "niveaux": {"debutant": self._box(y=0.6)}}

    def test_gate_avec_chiffre(self):
        c = self._ctrl()
        c.spots = self._spots9()
        rgb = np.full((2400, 1080, 3), 255, dtype=np.uint8)
        rgb[1980:2050, 40:110] = (43, 88, 179)  # encre bleue dans slot 1
        grid = {"n": 9}
        self.assertTrue(c._keypad_has_digit(rgb, grid))

    def test_gate_sans_chiffre(self):
        c = self._ctrl()
        c.spots = self._spots9()
        rgb = np.full((2400, 1080, 3), 255, dtype=np.uint8)
        self.assertFalse(c._keypad_has_digit(rgb, {"n": 9}))

    def test_gate_sans_slots(self):
        c = self._ctrl()
        c.spots = {}
        rgb = np.full((100, 100, 3), 255, dtype=np.uint8)
        self.assertFalse(c._keypad_has_digit(rgb, {"n": 9}))

    def test_blinds_sans_pubs(self):
        c = self._ctrl()
        c.spots = {"learned": {"win:ok": self._box(y=0.7),
                               "bonus:claim": self._box(y=0.66),
                               "hint:watch": self._box(y=0.8)}}
        pts = c._express_blinds(1080, 2400)
        # win:ok + 3 croix (rate/bonus/win-ok) ; claim/watch exclus, dedup ok
        self.assertTrue(any(abs(x - 0.5 * 1080) < 60 for (x, y) in pts))
        self.assertGreaterEqual(len(pts), 3)
        self.assertLess(len(pts), 8)

    def _tap_pos(self, msg):
        import re as _re
        m = _re.search(r"appui (\d+),(\d+)", msg)
        return int(m.group(1)), int(m.group(2))

    def test_express_direct_deux_appuis_sans_scan(self):
        """Verification puis « Nouvelle partie » / « Débutant » DIRECTS par
        hitboxes : aucune attente ni scan entre la verification et le 1er
        appui (budget 1 s — structurel, pas un delai reglable)."""
        c = self._ctrl()
        c.spots = self._spots9()
        seq = []
        c.shot = lambda stable=False: seq.append("shot") or np.full(
            (2400, 1080, 3), 255, dtype=np.uint8)
        c.tap_box = lambda b: seq.append(("tap", b))
        c.sleep = lambda ms: seq.append(("sleep", ms))
        c.countdown_sleep = lambda ms, what="Attend": seq.append(("sleep", ms))
        c.wait_for_grid = lambda max_ms=None: True
        self.controller.LINK = FakeLink()  # muet : aucun scan ne doit servir
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        self.assertTrue(c._postgame_express(grid))
        taps = [s[1] for s in seq if isinstance(s, tuple) and s[0] == "tap"]
        self.assertEqual(len(taps), 2)  # nouvelle puis debutant
        i_first = next(i for i, s in enumerate(seq)
                       if isinstance(s, tuple) and s[0] == "tap")
        # Rien (attente, scan, autre appui) entre la capture de verification
        # et le 1er appui : le budget 1 s est verrouille par la structure.
        self.assertEqual(seq[:i_first], ["shot"])
        self.assertEqual(self.controller.LINK.find_calls, 0)  # aucun scan
        # Les deux appuis visent bien les hitboxes (centres).
        def center(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        (x1, y1), (x2, y2) = [center(b) for b in taps]
        self.assertAlmostEqual(x1, 0.5 * 1080, delta=1)
        self.assertAlmostEqual(y1, 0.8 * 2400, delta=1)
        self.assertAlmostEqual(x2, 0.5 * 1080, delta=1)
        self.assertAlmostEqual(y2, 0.6 * 2400, delta=1)

    def test_express_repli_serie_quotidienne(self):
        """Menu « Série quotidienne » interpose : ses boutons ferment l'ecran,
        puis « Nouvelle partie » est RECLIQUE via le texte (pas l'emplacement
        predetermine) et « Débutant » par son hitbox."""
        c = self._ctrl()
        c.spots = self._spots9()
        c.shot = lambda stable=False: np.full((2400, 1080, 3), 255, dtype=np.uint8)
        started = []

        def wait_for_grid(max_ms=None):
            started.append(1)
            return len(started) >= 2  # 1er essai (aveugle) echoue, repli ok

        c.wait_for_grid = wait_for_grid
        self.controller.LINK = SeqLink([
            # 1) scan de repli : le menu serie est la (bouton « retour »).
            {"ok": True, "found": True, "group": 8, "text": "retour",
             "via": "accessibilite", "b": [10, 10, 100, 40],
             "hits": [{"group": 8, "text": "retour", "via": "accessibilite",
                       "b": [10, 10, 100, 40]}]},
            # 2) « Nouvelle partie » retrouve par le texte (OCR).
            {"ok": True, "found": True, "group": 0, "text": "Nouvelle partie",
             "via": "ocr", "b": [20, 20, 200, 60],
             "hits": [{"group": 0, "text": "Nouvelle partie", "via": "ocr",
                       "b": [20, 20, 200, 60]}]},
        ])
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        self.assertTrue(c._postgame_express(grid))
        self.assertEqual(len(started), 2)  # chaine aveugle puis repli
        taps = [l.get("msg", "") for l in list(c.logs)
                if "(test) appui" in l.get("msg", "")]
        self.assertEqual(len(taps), 5)  # 2 aveugles + retour + nouvelle + debutant
        # 1er appui : hitbox « Nouvelle partie » ; 2e : hitbox « Débutant ».
        x, y = self._tap_pos(taps[0])
        self.assertTrue(378 <= x <= 702 and 1848 <= y <= 1992, (x, y))
        x, y = self._tap_pos(taps[1])
        self.assertTrue(378 <= x <= 702 and 1368 <= y <= 1512, (x, y))
        # 3e : le bouton « retour » du menu serie (sa boite).
        x, y = self._tap_pos(taps[2])
        self.assertTrue(10 <= x <= 100 and 10 <= y <= 40, (x, y))
        # 4e : « Nouvelle partie » retrouvee PAR LE TEXTE (sa boite).
        x, y = self._tap_pos(taps[3])
        self.assertTrue(20 <= x <= 200 and 20 <= y <= 60, (x, y))
        # 5e : « Débutant » par son hitbox, a nouveau.
        x, y = self._tap_pos(taps[4])
        self.assertTrue(378 <= x <= 702 and 1368 <= y <= 1512, (x, y))

    def test_express_repli_popup_puis_ocr(self):
        """Popup sur l'ecran de fin : fermee au repli (bouton/croix), puis
        « Nouvelle partie » par le texte et « Débutant » par hitbox."""
        c = self._ctrl()
        c.spots = self._spots9()
        c.shot = lambda stable=False: np.full((2400, 1080, 3), 255, dtype=np.uint8)
        started = []

        def wait_for_grid(max_ms=None):
            started.append(1)
            return len(started) >= 2

        c.wait_for_grid = wait_for_grid
        self.controller.LINK = SeqLink([
            # 1) scan de repli : popup d'avis par-dessus l'ecran de fin.
            {"ok": True, "found": True, "group": 0, "text": "Aimez-vous",
             "via": "accessibilite", "b": [400, 1400, 680, 1500],
             "hits": [{"group": 0, "text": "Aimez-vous", "via": "accessibilite",
                       "b": [400, 1400, 680, 1500]}]},
            # 2) « Nouvelle partie » retrouve par le texte (OCR).
            {"ok": True, "found": True, "group": 0, "text": "Nouvelle partie",
             "via": "ocr", "b": [20, 20, 200, 60],
             "hits": [{"group": 0, "text": "Nouvelle partie", "via": "ocr",
                       "b": [20, 20, 200, 60]}]},
        ])
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        # rate_dismiss_texts fige : pas de recherche supplementaire au repli.
        with mock.patch.dict(config.get(), {"menus": {"rate_dismiss_texts": []}}):
            self.assertTrue(c._postgame_express(grid))
        taps = [l.get("msg", "") for l in list(c.logs)
                if "(test) appui" in l.get("msg", "")]
        self.assertEqual(len(taps), 5)  # 2 aveugles + croix + nouvelle + debutant
        # 3e : la croix de la popup d'avis (autour de [0.5, 0.625] ecran).
        x, y = self._tap_pos(taps[2])
        self.assertTrue(464 <= x <= 616 and 1440 <= y <= 1560, (x, y))
        # 4e : « Nouvelle partie » par le texte ; 5e : « Débutant » hitbox.
        x, y = self._tap_pos(taps[3])
        self.assertTrue(20 <= x <= 200 and 20 <= y <= 60, (x, y))
        x, y = self._tap_pos(taps[4])
        self.assertTrue(378 <= x <= 702 and 1368 <= y <= 1512, (x, y))

    def test_express_repli_rafale_si_ecran_sans_texte(self):
        """Repli sur une vue sans texte (perso/pub) : rafale aveugle (boutons
        connus uniquement, jamais de pub) puis nouvelle tentative."""
        c = self._ctrl()
        c.spots = self._spots9()
        c.spots["learned"] = {"win:ok": self._box(y=0.7)}
        c.shot = lambda stable=False: np.full((2400, 1080, 3), 255, dtype=np.uint8)
        started = []

        def wait_for_grid(max_ms=None):
            started.append(1)
            return len(started) >= 2

        c.wait_for_grid = wait_for_grid
        self.controller.LINK = SeqLink([
            {"ok": True, "found": False, "hits": []},  # ecran muet -> rafale
            {"ok": True, "found": False, "hits": []},  # nouvelle introuvable
            {"ok": True, "found": False, "hits": []},  # (2e passe du texte)
        ])
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        self.assertTrue(c._postgame_express(grid))
        taps = [l.get("msg", "") for l in list(c.logs)
                if "(test) appui" in l.get("msg", "")]
        # 1re chaine (2) + rafale aveugle (>= 3 points connus) + 2e chaine (2).
        self.assertGreaterEqual(len(taps), 7)

    def test_express_reprend_si_pave(self):
        c = self._ctrl()
        c.spots = self._spots9()
        rgb = np.full((2400, 1080, 3), 255, dtype=np.uint8)
        rgb[1980:2050, 40:110] = (43, 88, 179)
        c.shot = lambda stable=False: rgb
        self.controller.LINK = FakeLink()
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        self.assertFalse(c._postgame_express(grid))  # encore en jeu : normal
        taps = [l for l in list(c.logs) if "(test) appui" in l.get("msg", "")]
        self.assertEqual(taps, [])

    def test_express_sans_spots(self):
        c = self._ctrl()
        c.spots = {}
        grid = {"n": 9, "x0": 0, "y1": 0, "ch": 100}
        self.assertFalse(c._postgame_express(grid))

    def test_express_porte_perdu(self):
        """Pave encore rempli + ecran « Perdu » par-dessus : le bouton du
        popup part via le texte (pas l'emplacement habituel)."""
        c = self._ctrl()
        c.spots = self._spots9()
        c.handle_streak = lambda: None
        c.shot = lambda stable=False: rgb
        rgb = np.full((2400, 1080, 3), 255, dtype=np.uint8)
        rgb[1980:2050, 40:110] = (43, 88, 179)  # chiffre restant au pave
        c.wait_for_grid = lambda max_ms=None: True
        self.controller.LINK = FakeLink(find_answers=["found", "found"])
        with mock.patch.object(self.controller.spotsmod, "save",
                               lambda *a, **k: None):
            self.assertTrue(c._postgame_express({"n": 9, "x0": 0, "y1": 0, "ch": 100}))
        self.assertIn("loss:new_game", c.spots.get("learned") or {})
        taps = [l.get("msg", "") for l in list(c.logs)
                if "(test) appui" in l.get("msg", "")]
        self.assertGreaterEqual(len(taps), 2)  # popup + debutant
        import re as _re
        m = _re.search(r"appui (\d+),(\d+)", taps[0])
        x, y = int(m.group(1)), int(m.group(2))
        # 1er appui = bouton du popup (b=[10,10,100,40]), pas la rafale.
        self.assertGreaterEqual(x, 10)
        self.assertLessEqual(x, 100)
        self.assertGreaterEqual(y, 10)
        self.assertLessEqual(y, 40)


class FastGateTest(unittest.TestCase):
    """Porte rapide de fin de partie : vide/rouge -> analyse, pleine -> direct."""

    def _grid(self, n=2, step=40):
        xs = [float(i * step) for i in range(n + 1)]
        return {"n": n, "xs": xs, "ys": xs}

    def _rgb(self, fills):
        # fills : {(r,c): couleur} sur fond blanc, cases 40px.
        img = np.full((80, 80, 3), 255, dtype=np.uint8)
        for (r, c), col in fills.items():
            img[r * 40 + 12:r * 40 + 32, c * 40 + 10:c * 40 + 20] = col
        return img

    def test_pleine(self):
        c = make_ctrl(None)
        rgb = self._rgb({(r, c): (30, 30, 30) for r in range(2) for c in range(2)})
        self.assertFalse(c._grid_needs_work(rgb, self._grid()))

    def test_vide(self):
        c = make_ctrl(None)
        fills = {(r, c): (30, 30, 30) for r in range(2) for c in range(2)}
        del fills[(1, 1)]
        rgb = self._rgb(fills)
        self.assertTrue(c._grid_needs_work(rgb, self._grid()))

    def test_rouge(self):
        c = make_ctrl(None)
        fills = {(r, c): (30, 30, 30) for r in range(2) for c in range(2)}
        fills[(0, 1)] = (220, 30, 40)
        rgb = self._rgb(fills)
        self.assertTrue(c._grid_needs_work(rgb, self._grid()))


class CountdownTest(unittest.TestCase):
    """Le bandeau affiche 'Attend Ns...' pendant les attentes."""

    def test_compte_a_rebours(self):
        c = make_ctrl(None)
        c.countdown_sleep(1250)
        self.assertEqual(c.msg, "Attend 1s...")
        c.countdown_sleep(0)
        self.assertEqual(c.msg, "Attend 1s...")  # inchange, pas de crash


class HumanEdgeTest(unittest.TestCase):
    def test_bornes(self):
        import random
        from sudokubot import human
        random.seed(7)
        worst = 0.0
        for _ in range(5000):
            x, y = human.point_in(100.0, 100.0, 50.0, 50.0, 0.13, 0.28,
                                  edge_prob=1.0, edge_max=0.46)
            worst = max(worst, abs(x - 100.0) / 50.0, abs(y - 100.0) / 50.0)
        self.assertLessEqual(worst, 0.46 + 0.03)  # borne + biais droitier
        random.seed(7)
        worst0 = 0.0
        for _ in range(5000):
            x, y = human.point_in(100.0, 100.0, 50.0, 50.0, 0.13, 0.28)
            worst0 = max(worst0, abs(x - 100.0) / 50.0, abs(y - 100.0) / 50.0)
        self.assertLessEqual(worst0, 0.28 + 0.03)


class SpotsConfigTest(unittest.TestCase):
    def test_keys(self):
        m = config.section("menus")
        for key in ("spots_enabled", "spots_slot_ink", "postgame_flame_wait_ms",
                    "express_rounds", "express_gap_ms", "win_ok_pos",
                    "debutant_appear_ms", "appear_poll_ms"):
            self.assertIn(key, m, "menus.%s manquant" % key)
        self.assertIn("finish_extra_ms", config.section("vision"),
                      "vision.finish_extra_ms manquant")
        self.assertIn("grid_found_wait_ms", config.section("game"),
                      "game.grid_found_wait_ms manquant")
        rules = {r.get("name"): r for r in m.get("rules", [])}
        self.assertEqual(rules["debutant"]["wait_after_ms"], [600, 900])
        self.assertEqual(m.get("loading_max_ms"), 5000)
        h = config.section("human")
        for key in ("edge_prob", "edge_max"):
            self.assertIn(key, h, "human.%s manquant" % key)
        self.assertEqual(config.section("game").get("detect_delay_ms"), 206)


class DebugOverlayTest(unittest.TestCase):
    """debug.overlay est respecte : false = aucun encodage JPEG (plus rapide)."""

    def test_overlay_desactive(self):
        c = make_ctrl(None)
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        with mock.patch.dict(config.get(), {"debug": {"overlay": False}}):
            c._debug_overlay(rgb)
        self.assertIsNone(c.overlay)

    def test_overlay_actif(self):
        c = make_ctrl(None)
        rgb = np.zeros((40, 40, 3), dtype=np.uint8)
        with mock.patch.dict(config.get(), {"debug": {"overlay": True}}):
            c._debug_overlay(rgb)
        self.assertIsNotNone(c.overlay)


if __name__ == "__main__":
    unittest.main()

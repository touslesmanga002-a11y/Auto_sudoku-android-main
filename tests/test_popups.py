"""Tests des popups/didacticiels + Perdu (aucun telephone requis, LINK simule)."""
import os
import sys
import time
import unittest

# Bootstrap : voir test_solver.py (fonctionne sans PYTHONPATH manuel).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from sudokubot import config, controller, vision
from sudokubot.controller import halve_batch, pick_popup, POPUP_ORDER
from sudokubot.controller import banned_from_cells, verify_missing

try:
    from test_solver import P9, S4, G4
except ModuleNotFoundError:  # ex. `pytest` avec rootdir differente
    from tests.test_solver import P9, S4, G4


class FakeLink:
    """Simule LINK.call pour find_any / tap / tap_batch."""

    def __init__(self, hits=None, found_text=None, find_answers=None, ui_items=None):
        # hits : liste de {"group": i, "text":..., "via":..., "b":[...]}
        self.hits = hits if hits is not None else []
        self.found_text = found_text
        # find_answers : file de reponses "found"/"miss" consommee dans
        # l'ordre a chaque find_any (simule des ecrans qui changent).
        self.find_answers = list(find_answers or [])
        # ui_items : reponse fixe pour ui_scan [{"text":..., "b":[...]}].
        self.ui_items = ui_items if ui_items is not None else []
        self.taps = []
        self.find_calls = 0

    def call(self, ctype, timeout=25.0, **params):
        if ctype == "find_any":
            self.find_calls += 1
            if self.find_answers:
                ans = self.find_answers.pop(0)
                if ans == "found":
                    txt = self.found_text or "suivant"
                    return {"ok": True, "found": True, "group": 0, "text": txt,
                            "via": "accessibilite", "b": [10, 10, 100, 40],
                            "hits": [{"group": 0, "text": txt,
                                      "via": "accessibilite", "b": [10, 10, 100, 40]}]}
                return {"ok": True, "found": False, "hits": []}
            if self.hits:
                ordered = sorted(self.hits, key=lambda h: h["group"])
                h0 = ordered[0]
                return {"ok": True, "found": True, "group": h0["group"],
                        "text": h0["text"], "via": h0.get("via", "accessibilite"),
                        "b": h0.get("b", [10, 10, 100, 40]), "hits": ordered}
            if self.found_text:
                return {"ok": True, "found": True, "group": 0, "text": self.found_text,
                        "via": "accessibilite", "b": [10, 10, 100, 40],
                        "hits": [{"group": 0, "text": self.found_text,
                                  "via": "accessibilite", "b": [10, 10, 100, 40]}]}
            return {"ok": True, "found": False, "hits": []}
        if ctype in ("tap", "tap_batch"):
            self.taps.append((ctype, params))
            return {"ok": True, "done": 2, "total": 2, "failed": 0, "lag_max": 0}
        if ctype == "ui_scan":
            return {"ok": True, "items": list(self.ui_items)}
        self.taps.append((ctype, params))
        return {"ok": True}


def make_ctrl(link):
    c = controller.Controller({})
    c.test = True  # aucun appui reel : tap() ne fait que journaliser
    c.spots = {}  # isolation : pas d'emplacements en tests (sauf set explicite)
    return c


# Persistance spots neutralisee en tests (enregistree en memoire seulement).
# Portee MODULE (setUpModule) pour ne pas fuiter vers les autres fichiers
# de tests qui tournent dans le meme processus.
from unittest import mock as _mock
SPOTS_SAVED = []


def _fake_spots_save(spots, path=None):
    SPOTS_SAVED.append(spots)
    return spots


_save_patcher = None


def setUpModule():
    global _save_patcher
    _save_patcher = _mock.patch.object(controller.spotsmod, "save", _fake_spots_save)
    _save_patcher.start()


def tearDownModule():
    global _save_patcher
    if _save_patcher is not None:
        _save_patcher.stop()
        _save_patcher = None


class BatchTest(unittest.TestCase):
    def test_halve(self):
        self.assertEqual(halve_batch(27, 1), 27)
        self.assertEqual(halve_batch(27, 2), 13)
        self.assertEqual(halve_batch(27, 4), 6)
        self.assertEqual(halve_batch(27, 64), 1)   # plancher : 1 coup par salve
        self.assertEqual(halve_batch(1, 8), 1)
        self.assertEqual(halve_batch(8, 0), 8)      # diviseur invalide ignore
        self.assertEqual(halve_batch(8, None), 8)

    def test_loss_then_recovery(self):
        c = make_ctrl(None)
        self.assertEqual(c.batch_div, 1)
        c.batch_div = min(64, c.batch_div * 2)
        c.batch_div = min(64, c.batch_div * 2)
        self.assertEqual(halve_batch(18, c.batch_div), 4)
        c.batch_div = max(1, c.batch_div // 2)     # partie gagnee : remonte d'un cran
        self.assertEqual(halve_batch(18, c.batch_div), 9)


class OneGoTest(unittest.TestCase):
    """speed.one_go : toute la partie dans une seule salve, aucune pause en cours."""

    MOVES = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1)]

    def _prep(self, one_go):
        c = make_ctrl(None)
        c.dead_top = 0
        c.speed.set_choice("normal")       # palier a batch=1 : le pire cas
        xs = [float(i * 100) for i in range(10)]
        self.grid = {"n": 9, "xs": xs, "ys": list(xs), "cw": 100.0, "ch": 100.0,
                     "x0": xs[0], "y0": xs[0], "x1": xs[-1], "y1": xs[-1]}
        self.keys = [{"v": v, "x": 50.0, "y": 2000.0, "w": 90.0, "h": 90.0}
                     for v in range(1, 10)]
        c.sleep = lambda *_: None
        c.shot = lambda stable=False: np.zeros((2340, 1080, 3), np.uint8)
        c.spots_grid = lambda rgb, grid: grid
        c._ensure_quick_off = lambda rgb, g: rgb
        c._grid_needs_work = lambda rgb, g: False
        c._debug_overlay = lambda *a, **k: None
        self.calls = []

        def fake_batch(taps):
            self.calls.append([tuple(t) for t in taps])
            return {"ok": True, "done": len(taps), "total": len(taps),
                    "failed": 0, "lag_max": 0}

        c.tap_batch = fake_batch
        patcher = _mock.patch.dict(config.get()["speed"], {"one_go": one_go})
        patcher.start()
        self.addCleanup(patcher.stop)
        return c

    def test_une_seule_salve_pour_toute_la_partie(self):
        c = self._prep(True)
        self.assertTrue(c.speed.one_go())
        self.assertEqual(c.execute(self.MOVES, self.grid, self.keys), "complete")
        self.assertEqual(len(self.calls), 1)                       # une seule salve
        self.assertEqual(len(self.calls[0]), 2 * len(self.MOVES))  # tous les coups d'un bloc

    def test_sans_one_go_les_salves_restent_decoupees(self):
        c = self._prep(False)
        self.assertFalse(c.speed.one_go())
        cells = [[{"v": 1, "color": "ink"} for _ in range(9)] for _ in range(9)]
        with _mock.patch.object(controller.vision, "read_cells",
                                lambda rgb, g, tpl, cfg=None: cells):
            self.assertEqual(c.execute(self.MOVES, self.grid, self.keys), "complete")
        # batch=1 : une salve par coup, avec verification entre chaque (comportement historique)
        self.assertEqual(len(self.calls), len(self.MOVES))
        self.assertEqual([len(t) for t in self.calls], [2] * len(self.MOVES))

    def test_one_go_lit_la_config(self):
        c = make_ctrl(None)
        with _mock.patch.dict(config.get()["speed"], {"one_go": True}):
            self.assertTrue(c.speed.one_go())
        with _mock.patch.dict(config.get()["speed"], {"one_go": False}):
            self.assertFalse(c.speed.one_go())


class PriorityTest(unittest.TestCase):
    def test_order_loss_first(self):
        by = {"win": {}, "bonus": {}, "loss": {}, "hint": {}}
        self.assertEqual(pick_popup(by), "loss")

    def test_hint_before_bonus(self):
        # Ecran "Indices gratuits" : contient aussi "Bonus !" -> hint gagne.
        self.assertEqual(pick_popup({"bonus": {}, "hint": {}}), "hint")

    def test_bonus_alone(self):
        self.assertEqual(pick_popup({"win": {}, "bonus": {}}), "bonus")

    def test_none(self):
        self.assertIsNone(pick_popup({}))

    def test_order_constant(self):
        self.assertEqual(list(POPUP_ORDER),
                         ["loss", "tutorial", "tip", "hint", "friends", "bonus", "win"])


class KeypadPartialTest(unittest.TestCase):
    def _blobs(self, n):
        return [{"x": float(i), "y": 0.0, "w": 10.0, "h": 20.0} for i in range(n)]

    def test_missing_last(self):
        # 8 touches, le 9 est complet dans la grille -> [1..8].
        counts = {v: 9 for v in range(1, 10)}
        counts[9] = 9
        for v in range(1, 9):
            counts[v] = 3
        labs = vision._keypad_infer_from_counts(self._blobs(8), 9, counts)
        self.assertEqual(labs, [1, 2, 3, 4, 5, 6, 7, 8])

    def test_missing_middle(self):
        # Trou au milieu (le 2 est complet, cf. screenshot 10-01-50).
        counts = {v: 2 for v in range(1, 10)}
        counts[2] = 9
        labs = vision._keypad_infer_from_counts(self._blobs(8), 9, counts)
        self.assertEqual(labs, [1, 3, 4, 5, 6, 7, 8, 9])

    def test_mismatch_none(self):
        counts = {v: 1 for v in range(1, 10)}
        self.assertIsNone(vision._keypad_infer_from_counts(self._blobs(8), 9, counts))


class ScanTest(unittest.TestCase):
    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def test_scan_priority_with_hits(self):
        # L'agent renvoie 2 groupes : hint (3) + bonus (5) -> hint gagne.
        controller.LINK = FakeLink(hits=[
            {"group": 5, "text": "Bonus !", "via": "ocr", "b": [1, 2, 3, 4]},
            {"group": 3, "text": "Indices gratuits", "via": "accessibilite", "b": [1, 2, 3, 4]},
        ])
        c = make_ctrl(None)
        # Groupes reels du controleur : l'index 3 = hint, 5 = bonus ?
        names = [n for (n, _t) in [(n, t) for (n, on, t) in c._blocking_defs() if on]]
        self.assertEqual(names[3], "hint")
        self.assertEqual(names[5], "bonus")
        self.assertEqual(c.scan_blocking(use_ocr=True), "hint")

    def test_scan_none(self):
        controller.LINK = FakeLink(hits=[])
        c = make_ctrl(None)
        self.assertIsNone(c.scan_blocking(use_ocr=False))

    def test_tutorial_jamais_croix_que_suivant_valider(self):
        # 1er appel (bouton) trouve, re-scan : plus de didacticiel -> fini.
        # La croix ne doit JAMAIS etre visee (aucun log 'croix').
        controller.LINK = FakeLink(find_answers=["found", "miss"],
                                   found_text="suivant")
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self.assertTrue(c.handle_tutorial())
        self.assertEqual((c._pop_last, c._pop_n), (None, 0))
        croix = [l for l in list(c.logs) if "croix" in l.get("msg", "")]
        self.assertEqual(croix, [])

    def test_tutorial_plusieurs_etapes_enchainees(self):
        # suivant -> etape suivante (valider) -> termine : 4 scans au total.
        controller.LINK = FakeLink(find_answers=["found", "found",
                                                 "found", "miss"],
                                   found_text="suivant")
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self.assertTrue(c.handle_tutorial())
        self.assertEqual(controller.LINK.find_calls, 4)

    def test_hint_close_priority(self):
        # "Passer" visible -> tape sans chercher "Regarder".
        controller.LINK = FakeLink(found_text="Passer")
        c = make_ctrl(None)
        self.assertTrue(c.handle_hint())

    def test_menu_target_regression(self):
        pick = controller.Controller.pick_menu_target
        self.assertEqual(pick(True, False, True), "debutant")    # liste difficultes
        self.assertEqual(pick(False, True, False), "nouvelle_partie")  # victoire
        self.assertEqual(pick(False, False, True), "debutant")
        self.assertIsNone(pick(False, False, False))


class FusedPriorityTest(unittest.TestCase):
    """Apres une victoire : 'Nouvelle partie' visible -> on skippe la serie ;
    sinon flamme, sinon retour/continuer."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self, hits):
        controller.LINK = FakeLink(hits=hits)
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        return c

    def _hit(self, group, text):
        return {"group": group, "text": text, "via": "accessibilite",
                "b": [10, 10, 200, 60]}

    def test_nouvelle_avant_flamme(self):
        # Victoire + flamme visibles ensemble : nouvelle partie gagne.
        c = self._ctrl([self._hit(3, "Nouvelle partie"),
                        self._hit(1, "Touchez la flamme")])
        self.assertEqual(c.handle_menus_fused(), "nouvelle_partie")

    def test_nouvelle_avant_retour(self):
        c = self._ctrl([self._hit(3, "Nouvelle partie"),
                        self._hit(2, "retour")])
        self.assertEqual(c.handle_menus_fused(), "nouvelle_partie")

    def test_flamme_seule(self):
        import numpy as np
        c = self._ctrl([self._hit(1, "Touchez la flamme")])
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self.assertEqual(c.handle_menus_fused(), "flamme")

    def test_retour_seul(self):
        controller.LINK = FakeLink(hits=[self._hit(2, "Série quotidienne")],
                                   found_text="retour")
        # Note : FakeLink priorise `hits` pour le 1er appel (titres) ; le 2e
        # appel (bouton retour, accessibilite seule) utilise found_text.
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        self.assertEqual(c.handle_menus_fused(), "retour")

    def test_continuer_direct(self):
        # Le hit EST le bouton « continuer » : appui direct sur sa boite.
        c = self._ctrl([])
        self.assertEqual(c._tap_streak_retour(self._hit(2, "continuer")), "retour")

    def test_rate_reste_premier(self):
        c = self._ctrl([self._hit(0, "Aimez-vous le Sudoku?"),
                        self._hit(3, "Nouvelle partie")])
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self.assertEqual(c.handle_menus_fused(), "rate")


class StuckTest(unittest.TestCase):
    """Anti-blocage : meme action repetee -> escalade (retour hors partie)."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self, post_game=False, grid_recent=True):
        controller.LINK = FakeLink()
        c = make_ctrl(None)
        c.sleep = lambda ms: None  # pas d'attente reelle dans les tests
        c.test = False  # autorise le back (LINK simule, aucun appui reel)
        c._post_game = post_game
        c.last_grid_t = 9999999999.0 if grid_recent else 0.0
        return c

    def test_flamme_repetee_postgame_back(self):
        # Flamme tapee 4x sans effet apres une victoire -> 1 appui retour.
        c = self._ctrl(post_game=True)
        for _ in range(3):
            self.assertFalse(c._note_action("flamme"))
        self.assertTrue(c._note_action("flamme"))
        backs = [t for t in controller.LINK.taps if t[0] == "back"]
        self.assertEqual(len(backs), 1)

    def test_pas_de_back_en_pleine_partie(self):
        # Tutoriel persistant en pleine partie (grille vue a l'instant) :
        # escalade SANS retour (on reessaie juste l'autre branche).
        c = self._ctrl(post_game=False, grid_recent=True)
        for _ in range(3):
            self.assertFalse(c._note_action("popup:tutorial"))
        self.assertTrue(c._note_action("popup:tutorial"))
        backs = [t for t in controller.LINK.taps if t[0] == "back"]
        self.assertEqual(backs, [])
        self.assertIsNone(c._pop_last)  # memoire croix-d'abord reinitialisee

    def test_back_si_grille_perdue_de_vue(self):
        # Meme en partie, si la grille n'est plus vue depuis longtemps,
        # le retour est autorise (l'appui n'interrompra pas un jeu en cours).
        c = self._ctrl(post_game=False, grid_recent=False)
        for _ in range(4):
            c._note_action("popup:bonus")
        backs = [t for t in controller.LINK.taps if t[0] == "back"]
        self.assertEqual(len(backs), 1)

    def test_changement_action_rearme(self):
        c = self._ctrl(post_game=True)
        c._note_action("flamme")
        c._note_action("flamme")
        c._note_action("retour")   # autre ecran -> compteur reparti
        self.assertEqual((c._last_action, c._action_n), ("retour", 1))
        self.assertFalse(c._note_action("retour"))
        backs = [t for t in controller.LINK.taps if t[0] == "back"]
        self.assertEqual(backs, [])


class MissingKeysTest(unittest.TestCase):
    """Pave partiel au demarrage (ex. 1 3 4 5 6 7 8 9 sans le 2) : le chiffre
    manquant est verifie en grille, et on ne bloque jamais."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _cells(self, counts, N=9):
        cells = []
        pool = []
        for v, n in counts.items():
            pool += [{"v": v, "score": 0.9}] * n
        pool += [{"v": 0, "score": 1.0}] * (N * N - len(pool))
        it = iter(pool)
        for r in range(N):
            cells.append([dict(next(it), r=r, c=c) for c in range(N)])
        return cells

    def _keys(self, values):
        return [{"v": v, "x": 0.0, "y": 0.0, "w": 10.0, "h": 10.0} for v in values]

    def _ctrl(self):
        controller.LINK = FakeLink()
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        return c

    def test_verify_confirme_par_grille(self):
        from sudokubot.controller import verify_missing
        ok, ko = verify_missing([2], {2: 9}, set(), 9)
        self.assertEqual((ok, ko), ([2], []))

    def test_verify_confirme_par_jeu(self):
        # Le jeu masque le 2 (termine) meme si on ne le compte que 8 fois
        # (une case mal lue) : normal, pas de relecture inutile.
        from sudokubot.controller import verify_missing
        ok, ko = verify_missing([2], {2: 8}, {2}, 9)
        self.assertEqual((ok, ko), ([2], []))

    def test_verify_douteux(self):
        from sudokubot.controller import verify_missing
        ok, ko = verify_missing([2], {2: 3}, set(), 9)
        self.assertEqual((ok, ko), ([], [2]))

    def test_manquant_complet_on_joue_le_reste(self):
        # Cas reel : pave 1 3 4 5 6 7 8 9, le 2 est 9x en grille.
        c = self._ctrl()
        cells = self._cells({2: 9, 1: 3, 5: 2})
        moves = [(0, 0, 2), (0, 1, 5), (1, 1, 3)]
        out = c._handle_missing_keys(moves, self._keys([1, 3, 4, 5, 6, 7, 8, 9]),
                                     {2}, cells, 9)
        self.assertEqual(out, [(0, 1, 5), (1, 1, 3)])  # le coup en 2 ecarte, le reste joue
        self.assertEqual(controller.LINK.taps, [])  # aucune relecture, aucun blocage

    def test_douteux_relecture_bornee_puis_on_joue(self):
        # Touche absente ET chiffre incomplet : 3 relectures max, puis on joue.
        c = self._ctrl()
        cells = self._cells({2: 3})
        moves = [(0, 0, 2), (0, 1, 5)]
        keys = self._keys([1, 3, 4, 5, 6, 7, 8, 9])
        for _ in range(5):
            out = c._handle_missing_keys(moves, keys, set(), cells, 9)
        self.assertEqual(out, [(0, 1, 5)])
        self.assertEqual(c._keypad_retries, 0)  # compteur retombe, pas de blocage

    def test_rien_de_jouable_pas_de_blocage(self):
        # Tous les coups exigent la touche absente : relecture, sans bloquer.
        c = self._ctrl()
        cells = self._cells({2: 3})
        out = c._handle_missing_keys([(0, 0, 2)], self._keys([1, 3, 4, 5, 6, 7, 8, 9]),
                                     set(), cells, 9)
        self.assertEqual(out, [])
        self.assertEqual((c._last_action, c._action_n), ("attente_touches", 1))


class WatchdogTest(unittest.TestCase):
    """Watchdog : 7 s sans progres -> 'suivant' en bas -> terminer ;
    11 s -> OCR de localisation + relance nouvelle partie (sans grille)."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self, idle_s, **kw):
        controller.LINK = FakeLink(**{k: v for k, v in kw.items()
                                       if k in ("hits", "found_text",
                                                "find_answers", "ui_items")})
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c._last_progress_t = time.time() - idle_s
        c._loading_until = 0.0
        return c

    def _bottom_suivant(self):
        return {"text": "suivant", "b": [100, 2000, 400, 2100]}

    def test_suivant_en_bas_termine(self):
        # 'suivant' en bas -> didacticiel confirme -> 'terminer' tape.
        c = self._ctrl(8, ui_items=[self._bottom_suivant()],
                       find_answers=["found", "miss"], found_text="terminer")
        self.assertTrue(c._tutorial_bottom_check())

    def test_suivant_pas_en_bas_ignore(self):
        haut = {"text": "suivant", "b": [100, 200, 400, 300]}
        c = self._ctrl(8, ui_items=[haut], find_answers=["found"])
        self.assertFalse(c._tutorial_bottom_check())
        self.assertEqual(controller.LINK.find_calls, 0)

    def test_watchdog_stage1_7s(self):
        c = self._ctrl(8, ui_items=[self._bottom_suivant()],
                       find_answers=["found", "miss"], found_text="terminer")
        self.assertTrue(c._watchdog(None, {"n": 9}))
        self.assertEqual(c._last_action, "tuto_watchdog")

    def test_watchdog_calme_avant_7s(self):
        c = self._ctrl(3, ui_items=[self._bottom_suivant()])
        self.assertFalse(c._watchdog(None, {"n": 9}))
        self.assertEqual(controller.LINK.find_calls, 0)

    def test_watchdog_respecte_chargement(self):
        c = self._ctrl(30, ui_items=[self._bottom_suivant()])
        c._loading_until = time.time() + 60
        self.assertFalse(c._watchdog(None, None))

    def test_watchdog_stage2_retrouve_didacticiel(self):
        # 12 s sans grille : le sweep OCR voit le didacticiel (groupe 2).
        hit = {"group": 2, "text": "Dernier chiffre possible",
               "via": "ocr", "b": [10, 10, 200, 60]}
        c = self._ctrl(12, hits=[hit])
        c._post_game = False
        c.last_grid_t = 0.0  # grille perdue de vue depuis longtemps
        self.assertTrue(c._watchdog(None, None))
        self.assertEqual(c._last_action, "tuto_relance")

    def test_watchdog_stage2_back_borne(self):
        # Rien a l'ecran, post-partie : 1 seul 'retour' (cooldown).
        c = self._ctrl(12)
        c.test = False
        c._post_game = True
        self.assertTrue(c._lost_sweep())
        self.assertFalse(c._lost_sweep())  # cooldown : pas de 2e retour
        backs = [t for t in controller.LINK.taps if t[0] == "back"]
        self.assertEqual(len(backs), 1)

    def test_watchdog_en_niveau_jamais_de_relance(self):
        # 12 s avec grille : balayage seul, aucun back, aucune relance.
        c = self._ctrl(12)
        c.test = False
        self.assertFalse(c._watchdog(None, {"n": 9}))
        self.assertEqual(controller.LINK.taps, [])


class RedTest(unittest.TestCase):
    """Chiffres rouges : ne comptent pas, et le solveur ne les replace pas."""

    def _cells(self, grid, reds=None, score=0.9):
        reds = reds or {}
        cells = []
        for r, row in enumerate(grid):
            crow = []
            for c, v in enumerate(row):
                if (r, c) in reds:
                    crow.append({"r": r, "c": c, "v": reds[(r, c)],
                                 "score": score, "alts": [], "color": "error"})
                elif v:
                    crow.append({"r": r, "c": c, "v": v, "score": score,
                                 "alts": [], "color": "given"})
                else:
                    crow.append({"r": r, "c": c, "v": 0, "score": 1.0,
                                 "alts": [], "color": None})
            cells.append(crow)
        return cells

    def test_banned_from_cells(self):
        cells = [[{"r": 0, "c": 0, "v": 5, "color": "error"},
                  {"r": 0, "c": 1, "v": 5, "color": "given"},
                  {"r": 0, "c": 2, "v": 0, "color": None}]]
        self.assertEqual(banned_from_cells(cells), {(0, 0): {5}})

    def test_grid_counts_ignore_red(self):
        c = make_ctrl(None)
        cells = [[{"r": 0, "c": 0, "v": 5, "score": 0.9, "color": "error"},
                  {"r": 0, "c": 1, "v": 5, "score": 0.9, "color": "given"},
                  {"r": 0, "c": 2, "v": 5, "score": 0.2, "color": "given"}]]
        cnt = c._grid_counts(cells, 9)
        self.assertEqual(cnt[5], 1)  # seul le noir a bon score compte

    def test_red_wrong_value_corrected(self):
        # P9 + rouge errone au trou (0,2) : la solution reste l'unique S,
        # le rouge n'est jamais replace (le coup corrige la case).
        from sudokubot import solver
        cnt0, S = solver.solve(P9, 3, 3)
        self.assertEqual(cnt0, 1)
        w = S[0][2] % 9 + 1  # valeur differente de la vraie
        self.assertNotEqual(w, S[0][2])
        c = make_ctrl(None)
        cells = self._cells(P9, reds={(0, 2): w})
        res = c.solve_cells(cells, 3, 3)
        self.assertIsNotNone(res)
        _, sol, _ = res
        self.assertEqual(sol, S)
        self.assertNotEqual(sol[0][2], w)

    def test_red_true_value_never_replayed(self):
        # 4x4 a trou unique : si le rouge est justement la seule valeur
        # coherente, soit on trouve une alternative sans ce chiffre,
        # soit on declare incoherent — JAMAIS on ne le replace.
        c = make_ctrl(None)
        cells = self._cells(G4, reds={(3, 3): 1})
        res = c.solve_cells(cells, 2, 2)
        if res is not None:
            _, sol, _ = res
            self.assertNotEqual(sol[3][3], 1)
        # controle sans rouge : la solution unique est retrouvee.
        res2 = c.solve_cells(self._cells(G4), 2, 2)
        self.assertIsNotNone(res2)
        self.assertEqual(res2[1], S4)


class ToolbarGuardTest(unittest.TestCase):
    """Le pave accessibilite ignore la rangee d'outils (badge, libelles) :
    aucun appui ne doit atterrir sur 'Astuce intelligente' par erreur."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _grid(self):
        return {"n": 9, "y1": 1370.0}

    def _key(self, v, cy=1900.0, h=80.0, cx=None):
        return {"text": str(v), "b": [(cx or (100.0 + v * 90.0)) - 35, cy - h / 2,
                                      (cx or (100.0 + v * 90.0)) + 35, cy + h / 2]}

    def test_badge_ignore_et_vraie_touche_gardee(self):
        # Touches 1..9 (h=80) + badge '3' (h=36) sur la rangee d'outils :
        # le 3 garde la position du pave, pas celle du badge.
        items = [self._key(v) for v in range(1, 10)]
        items.append({"text": "Astuce intelligente",
                      "b": [700.0, 1640.0, 980.0, 1700.0]})
        items.append({"text": "3", "b": [900.0, 1560.0, 936.0, 1596.0]})
        controller.LINK = FakeLink(ui_items=items)
        c = make_ctrl(None)
        keys = c.keypad_from_accessibility(self._grid())
        self.assertIsNotNone(keys)
        by_v = {k["v"]: k for k in keys}
        self.assertEqual(by_v[3]["y"], 1900.0)  # pas 1578 (badge)

    def test_chiffre_cache_badge_seul_rejete(self):
        # Le 5 est masque (complet) : seul son badge reste -> rejete
        # (bande d'outils), le 5 est declare absent (masque par le jeu).
        items = [self._key(v) for v in (1, 2, 3, 4, 6, 7, 8, 9)]
        items.append({"text": "Astuce intelligente",
                      "b": [700.0, 1640.0, 980.0, 1700.0]})
        items.append({"text": "5", "b": [900.0, 1560.0, 936.0, 1596.0]})
        controller.LINK = FakeLink(ui_items=items)
        c = make_ctrl(None)
        keys = c.keypad_from_accessibility(self._grid())
        self.assertIsNotNone(keys)
        self.assertNotIn(5, [k["v"] for k in keys])

    def test_sans_barre_d_outils_micro_noeud_rejete(self):
        # Sans libelles d'outils visibles : le micro-noeud est rejete par
        # sa hauteur (< 60% de la mediane).
        items = [self._key(v) for v in (1, 2, 4, 5, 6, 7, 8, 9)]
        items.append({"text": "3", "b": [900.0, 1560.0, 936.0, 1596.0]})
        controller.LINK = FakeLink(ui_items=items)
        c = make_ctrl(None)
        keys = c.keypad_from_accessibility(self._grid())
        self.assertIsNotNone(keys)
        self.assertNotIn(3, [k["v"] for k in keys])

    def test_tap_astuce_volontaire(self):
        controller.LINK = FakeLink(found_text="Astuce intelligente")
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        self.assertTrue(c._tap_astuce_button())

    def test_validate_keys_zone_et_encre(self):
        # Touche toolbar (y~1.5ch), toggle (y~0.3ch) et touche grise :
        # rejetees. Seule la vraie touche bleue du pave est gardee.
        H, W = 2200, 1000
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)
        rgb[1890:1970, 100:180] = (43, 88, 179)  # vraie touche bleue
        grid = {"n": 9, "y1": 1370.0, "ch": 116.0}
        keys = [
            {"v": 2, "x": 140.0, "y": 1930.0, "w": 70.0, "h": 70.0},
            {"v": 3, "x": 500.0, "y": 1545.0, "w": 70.0, "h": 70.0},
            {"v": 4, "x": 900.0, "y": 1410.0, "w": 60.0, "h": 40.0},
            {"v": 5, "x": 300.0, "y": 1930.0, "w": 70.0, "h": 70.0},
        ]
        c = make_ctrl(None)
        c.last_rgb = rgb
        out = c._validate_keys(keys, grid)
        self.assertEqual([k["v"] for k in out], [2])  # 3=toolbar, 4=toggle, 5=grise

    def test_validate_tout_rejete_pas_d_appuis(self):
        c = make_ctrl(None)
        c.last_rgb = np.full((2200, 1000, 3), 255, dtype=np.uint8)
        grid = {"n": 9, "y1": 1370.0, "ch": 116.0}
        keys = [{"v": 3, "x": 500.0, "y": 1545.0, "w": 70.0, "h": 70.0}]
        self.assertEqual(c._validate_keys(keys, grid), [])

    def test_quick_cooldown_pas_de_martelage(self):
        from unittest import mock
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.shot = lambda stable=False: np.zeros((10, 10, 3), dtype=np.uint8)
        grid = {"n": 9, "y1": 1370.0, "ch": 116.0}
        rgb = np.zeros((2200, 1000, 3), dtype=np.uint8)
        st = {"active": True, "x": 900.0, "y": 1600.0, "frac": 0.1}
        with mock.patch.object(controller.vision, "detect_quick_toggle",
                               return_value=dict(st)):
            c._ensure_quick_off(rgb, grid)
            n1 = len([l for l in list(c.logs) if "(test) appui" in l.get("msg", "")])
            c._ensure_quick_off(rgb, grid)  # aussitot : cooldown, aucun appui
            n2 = len([l for l in list(c.logs) if "(test) appui" in l.get("msg", "")])
        self.assertEqual(n1, 2)  # toggle jamais eteint (mock) -> les 2 essais
        self.assertEqual(n2, 2)  # puis cooldown : plus rien

    def test_plan_astuce_une_fois_par_partie(self):
        from unittest import mock
        c = make_ctrl(None)
        moves = [(0, 0, v) for v in range(1, 7)]
        with mock.patch("random.random", return_value=0.0):
            target, detour = c._plan_astuce(moves)
            self.assertTrue(detour)
            self.assertEqual(target, moves[:3])
        c._astuce_used = True  # tentative effectuee : plus de tirage
        with mock.patch("random.random", return_value=0.0):
            _, detour2 = c._plan_astuce(moves)
            self.assertFalse(detour2)
        self.assertFalse(c._plan_astuce([(0, 0, 1)])[1])  # trop peu de coups

    def test_compteurs_gris_et_ligne_rejetes(self):
        # Meme taille que les touches mais GRIS (compteurs) ou BLEU hors
        # rangee : rejetes. Seule la rangee bleue principale est gardee.
        H, W = 2200, 1000
        rgb = np.full((H, W, 3), 255, dtype=np.uint8)
        items = []
        for i, v in enumerate(range(1, 10)):
            cx, cy = 120.0 + i * 90.0, 1900.0
            rgb[int(cy) - 25:int(cy) + 25, int(cx) - 20:int(cx) + 20] = (43, 88, 179)
            items.append({"text": str(v), "b": [cx - 35, cy - 40, cx + 35, cy + 40]})
            # compteur gris juste en dessous (meme taille !)
            items.append({"text": str(v), "b": [cx - 35, cy + 68, cx + 35, cy + 148]})
            rgb[int(cy) + 68:int(cy) + 148, int(cx) - 35:int(cx) + 35] = (79, 82, 86)
        controller.LINK = FakeLink(ui_items=items)
        c = make_ctrl(None)
        c.last_rgb = rgb
        keys = c.keypad_from_accessibility({"n": 9, "y1": 1370.0})
        self.assertIsNotNone(keys)
        self.assertEqual(sorted(k["v"] for k in keys), list(range(1, 10)))
        for k in keys:
            self.assertAlmostEqual(k["y"], 1900.0, delta=1.0)


class QuickOffTest(unittest.TestCase):
    """_ensure_quick_off : pilule bleue -> appui + screenshot frais."""

    def _shot(self, on):
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        img[60:80, 186:197] = (30, 144, 255) if on else (200, 200, 200)
        return img

    def _grid(self):
        return {"x1": 196.0, "y1": 40.0, "ch": 60.0, "n": 9}

    def _taps(self, c):
        return [l for l in list(c.logs) if "(test) appui" in l.get("msg", "")]

    def test_eteint_puis_frais(self):
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        rgb_off = self._shot(False)
        c.shot = lambda stable=False: rgb_off
        out = c._ensure_quick_off(self._shot(True), self._grid())
        self.assertIs(out, rgb_off)
        self.assertEqual(len(self._taps(c)), 1)

    def test_deja_eteint_rien_a_faire(self):
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.shot = lambda stable=False: (_ for _ in ()).throw(AssertionError("pas de capture"))
        out = c._ensure_quick_off(self._shot(False), self._grid())
        self.assertEqual(len(self._taps(c)), 0)
        self.assertEqual(out.shape, (200, 200, 3))

    def test_appui_sur_emplacement_memorise(self):
        # Pilule detectee ailleurs (centroide) mais emplacement calibre :
        # l'appui part sur l'emplacement memorise.
        import re
        from unittest import mock
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2000, 1000, 3), dtype=np.uint8)
        c.shot = lambda stable=False: np.zeros((2000, 1000, 3), dtype=np.uint8)
        c.spots = {"game": {"toggle": {"x": 0.2, "y": 0.3, "w": 0.05, "h": 0.02}}}
        grid = {"n": 9, "y1": 1370.0, "ch": 116.0}
        rgb = np.zeros((2000, 1000, 3), dtype=np.uint8)
        st = {"active": True, "x": 900.0, "y": 1600.0, "frac": 0.1}
        off = {"active": False, "x": None, "y": None, "frac": 0.0}
        with mock.patch.object(controller.vision, "detect_quick_toggle",
                               side_effect=[dict(st), dict(off), dict(off)]):
            c._ensure_quick_off(rgb, grid)
        taps = self._taps(c)
        self.assertEqual(len(taps), 1)
        m = re.search(r"appui (\d+),(\d+)", taps[0].get("msg", ""))
        self.assertIsNotNone(m)
        # centre memorise (200, 600) a la dispersion pres, pas le centroide
        self.assertLess(abs(int(m.group(1)) - 200), 60)
        self.assertLess(abs(int(m.group(2)) - 600), 60)

    def test_filet_bleu_emplacement(self):
        # Detection par amas muette mais pilule bleue sur l'emplacement
        # memorise : on eteint quand meme.
        import re
        from unittest import mock
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        rgb = np.zeros((2000, 1000, 3), dtype=np.uint8)
        rgb[680:720, 900:970] = (30, 144, 255)  # pilule bleue
        c.last_rgb = rgb
        c.shot = lambda stable=False: np.zeros((2000, 1000, 3), dtype=np.uint8)
        c.spots = {"game": {"toggle": {"x": 0.935, "y": 0.35, "w": 0.07, "h": 0.02}}}
        grid = {"n": 9, "y1": 1370.0, "ch": 116.0}
        off = {"active": False, "x": None, "y": None, "frac": 0.0}
        with mock.patch.object(controller.vision, "detect_quick_toggle",
                               return_value=dict(off)):
            c._ensure_quick_off(rgb, grid)
        taps = self._taps(c)
        self.assertEqual(len(taps), 1)
        m = re.search(r"appui (\d+),(\d+)", taps[0].get("msg", ""))
        self.assertIsNotNone(m)
        self.assertLess(abs(int(m.group(1)) - 935), 60)
        self.assertLess(abs(int(m.group(2)) - 700), 60)


class LearnedButtonsTest(unittest.TestCase):
    """Bouton deja vu (OCR) : retape direct, sans scan ; oublie si blocage."""

    def setUp(self):
        self.old = controller.LINK
        del SPOTS_SAVED[:]

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self):
        controller.LINK = FakeLink()
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        return c

    def test_tap_appris_sans_scan(self):
        c = self._ctrl()
        c.spots = {"learned": {"win:ok": {"x": 0.5, "y": 0.7, "w": 0.6, "h": 0.08}}}
        self.assertTrue(c.handle_win())
        link = controller.LINK
        self.assertEqual(link.find_calls, 0)  # aucun find_any : direct
        taps = [l for l in list(c.logs) if "(test) appui" in l.get("msg", "")]
        self.assertEqual(len(taps), 1)

    def test_ocr_apprend_pour_la_suite(self):
        c = self._ctrl()
        controller.LINK = FakeLink(found_text="OK")
        self.assertTrue(c.handle_win())
        learned = (c.spots.get("learned") or {}).get("win:ok")
        self.assertIsNotNone(learned)
        self.assertTrue(all(0.0 <= learned[k] <= 1.0 for k in ("x", "y", "w", "h")))
        self.assertTrue(SPOTS_SAVED)  # persiste (mocke en tests)

    def test_oubli_si_blocage(self):
        c = self._ctrl()
        c._post_game = True
        c.spots = {"learned": {"win:ok": {"x": 0.5, "y": 0.7, "w": 0.6, "h": 0.08},
                               "bonus:claim": {"x": 0.5, "y": 0.8, "w": 0.5, "h": 0.08}}}
        for _ in range(4):
            c._note_action("popup:win")
        learned = c.spots.get("learned") or {}
        self.assertNotIn("win:ok", learned)  # oublie : re-apprendra par OCR
        self.assertIn("bonus:claim", learned)  # les autres restent


class DrainTest(unittest.TestCase):
    """Apres-match : 1 scan accessibilite, hitboxes, OCR seulement en repli."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _hit(self, group, text):
        return {"group": group, "text": text, "via": "accessibilite",
                "b": [10, 10, 200, 60]}

    def _ctrl(self, hits):
        controller.LINK = FakeLink(hits=hits)
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c._post_game = True
        return c

    def test_nouvelle_direct_hitbox(self):
        c = self._ctrl([self._hit(9, "Nouvelle partie"), self._hit(7, "Touchez la flamme")])
        c._home_done = True
        c.spots = {"fin_partie": {"nouvelle_partie": {"x": 0.5, "y": 0.9, "w": 0.6, "h": 0.07}}}
        self.assertEqual(c._postgame_drain(), "nouvelle_partie")
        link = controller.LINK
        self.assertEqual(link.find_calls, 1)  # que le scan titres, aucun OCR
        logs = [l.get("msg", "") for l in list(c.logs)]
        self.assertTrue(any("nouvelle_partie" in m for m in logs))

    def test_nouvelle_sans_hitbox_repli(self):
        # Sans hitbox : le bouton part par le texte (1 seul find_any), puis
        # l'attente adaptative (ui_scan, wait=0 force) avant le cycle suivant.
        c = self._ctrl([self._hit(9, "Nouvelle partie")])
        c._home_done = True
        c.spots = {}
        self.assertEqual(c._postgame_drain(), "nouvelle_partie")
        self.assertEqual(controller.LINK.find_calls, 1)

    def test_nouvelle_puis_debutant_enchaines(self):
        """Les DEUX hitboxes connues : enchainement direct nouvelle ->
        debutant dans le meme souffle (1 seul scan, 2 appuis)."""
        c = self._ctrl([self._hit(9, "Nouvelle partie")])
        c.countdown_sleep = lambda ms, what="Attend": None
        c._home_done = True
        c.spots = {"fin_partie": {"nouvelle_partie": {"x": 0.5, "y": 0.9,
                                                       "w": 0.6, "h": 0.07}},
                   "niveaux": {"debutant": {"x": 0.5, "y": 0.6,
                                             "w": 0.4, "h": 0.06}}}
        self.assertEqual(c._postgame_drain(), "debutant")
        self.assertEqual(controller.LINK.find_calls, 1)  # aucun scan en plus
        taps = [l for l in list(c.logs) if "(test) appui" in l.get("msg", "")]
        self.assertEqual(len(taps), 2)  # nouvelle + debutant, rien d'autre

    def test_flamme_si_pas_de_nouvelle(self):
        c = self._ctrl([self._hit(7, "Touchez la flamme")])
        self.assertEqual(c._postgame_drain(), "flamme")

    def test_rien_vu_repli_ocr(self):
        controller.LINK = FakeLink(hits=[])
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c._post_game = True
        self.assertFalse(c._postgame_drain())

    def test_avis_avant_tout(self):
        c = self._ctrl([self._hit(0, "Aimez-vous le Sudoku?"),
                        self._hit(9, "Nouvelle partie")])
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        self.assertEqual(c._postgame_drain(), "rate")


class LossFlowTest(unittest.TestCase):
    """Perdu : « Nouvelle partie » du POPUP (position differente de
    l'emplacement habituel) -> via le texte (accessibilite/OCR) puis appris,
    JAMAIS l'hitbox fin_partie."""

    FIN = {"nouvelle_partie": {"x": 0.5, "y": 0.9, "w": 0.6, "h": 0.06}}
    NIV = {"debutant": {"x": 0.5, "y": 0.6, "w": 0.4, "h": 0.06}}

    def setUp(self):
        self.old = controller.LINK
        del SPOTS_SAVED[:]

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self, link):
        controller.LINK = link
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        c.countdown_sleep = lambda ms, what="Attend": None
        c.shot = lambda stable=False: np.zeros((2400, 1080, 3), dtype=np.uint8)
        c.handle_streak = lambda: None
        c.wait_for_grid = lambda max_ms=None: False
        c.last_rgb = np.zeros((2400, 1080, 3), dtype=np.uint8)
        c._home_done = True
        return c

    def _taps(self, c):
        import re
        return [l.get("msg", "") for l in list(c.logs)
                if "(test) appui" in l.get("msg", "")]

    @staticmethod
    def _find_xy(msg):
        import re
        m = re.search(r"appui (\d+),(\d+)", msg)
        return (int(m.group(1)), int(m.group(2))) if m else (None, None)

    def test_ocr_pas_hitbox(self):
        c = self._ctrl(FakeLink(find_answers=["found"]))
        c.spots = {"fin_partie": self.FIN, "niveaux": self.NIV}
        self.assertTrue(c.handle_loss())
        self.assertEqual(controller.LINK.find_calls, 1)  # recherche du texte
        taps = self._taps(c)
        self.assertGreaterEqual(len(taps), 2)  # popup + debutant
        x, y = self._find_xy(taps[0])
        # 1er appui = le bouton trouve par le texte (b=[10,10,100,40])...
        self.assertGreaterEqual(x, 10)
        self.assertLessEqual(x, 100)
        self.assertGreaterEqual(y, 10)
        self.assertLessEqual(y, 40)
        # ... et PAS l'hitbox fin_partie (0.5, 0.9) = (540, 2160).
        self.assertGreater(abs(x - 540) + abs(y - 2160), 1000)
        self.assertIn("loss:new_game", c.spots.get("learned") or {})
        self.assertTrue(SPOTS_SAVED)  # persiste (mocke en tests)

    def test_appris_direct_sans_scan(self):
        c = self._ctrl(FakeLink())
        c.spots = {"learned": {"loss:new_game": {"x": 0.5, "y": 0.55, "w": 0.4, "h": 0.05}},
                   "niveaux": self.NIV}
        self.assertTrue(c.handle_loss())
        self.assertEqual(controller.LINK.find_calls, 0)  # aucun scan
        taps = self._taps(c)
        x, y = self._find_xy(taps[0])
        # centre appris (540, 1320) a la dispersion pres
        self.assertLess(abs(x - 540), 220)
        self.assertLess(abs(y - 1320), 65)
        self.assertLess(abs(x - 540) + abs(y - 2160), 1000)  # jamais l'hitbox


class WaitSeenTest(unittest.TestCase):
    """Attente adaptative « Nouvelle partie » -> « Debutant » : ui_scan seul
    (jamais de find_any : les contrats 'bouton appris = zero scan' restent
    vrais), sortie totale des que le texte est visible."""

    def setUp(self):
        self.old = controller.LINK

    def tearDown(self):
        controller.LINK = self.old

    def _ctrl(self, link):
        controller.LINK = link
        c = make_ctrl(None)
        c.sleep = lambda ms: None
        return c

    def test_sort_tot_si_visible(self):
        c = self._ctrl(FakeLink(ui_items=[{"text": "Débutant Lv1", "b": [1, 2, 3, 4]}]))
        self.assertTrue(c._wait_seen(["Débutant"], [0, 1200], "débutant"))
        self.assertEqual(controller.LINK.find_calls, 0)  # ui_scan, pas find_any

    def test_plafond_si_invisible(self):
        c = self._ctrl(FakeLink())
        self.assertFalse(c._wait_seen(["Débutant"], [0, 300], "débutant"))
        self.assertEqual(controller.LINK.find_calls, 0)

    def test_texte_normalise(self):
        # Accents/casse : « Débutant » attendu trouve meme ecrit "debutant".
        c = self._ctrl(FakeLink(ui_items=[{"text": "debutant", "b": [1, 2, 3, 4]}]))
        self.assertTrue(c._wait_seen(["Débutant"], [0, 600], "débutant"))


class ConfigTest(unittest.TestCase):
    def test_popup_keys_present(self):
        m = config.section("menus")
        for key in ("popups_enabled", "loss_title_texts", "tutorial_title_texts",
                    "tip_title_texts", "hint_title_texts", "friends_title_texts",
                    "bonus_title_texts", "win_title_texts", "dismiss_wait_after_ms",
                    "unknown_back_grace_s", "rate_dismiss_texts",
                    "tutorial_close_pos", "bonus_close_pos",
                    "stuck_repeats", "stuck_back",
                    "tutorial_max_steps", "tutorial_end_texts",
                    "tutorial_bottom_frac", "watchdog_enabled",
                    "watchdog_stuck_s", "watchdog_lost_s",
                    "watchdog_ocr_cooldown_s", "watchdog_back_cooldown_s",
                    "astuce_texts", "astuce_pos", "toolbar_texts",
                    "quick_toggle_enabled", "quick_toggle_max_taps"):
            self.assertIn(key, m, "menus.%s manquant" % key)
        v = config.section("vision")
        self.assertIn("keypad_missing_retries", v)
        self.assertIn("keypad_acc_min_h_ratio", v)
        self.assertIn("keypad_acc_blue", v)
        h = config.section("human")
        self.assertIn("astuce_prob", h)
        self.assertAlmostEqual(h["astuce_prob"], 0.065)
        # positions valides : fractions d'ecran [x, y]
        for key in ("tutorial_close_pos", "bonus_close_pos"):
            pos = m.get(key)
            if pos is not None:
                self.assertEqual(len(pos), 2)
                self.assertTrue(all(0.0 <= v <= 1.0 for v in pos))


if __name__ == "__main__":
    unittest.main()

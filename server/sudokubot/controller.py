"""Controleur : le "cerveau". Decide tout, n'agit qu'a travers LINK.call(...) vers l'agent AutoJs6."""
import base64
import collections
import random
import threading
import time
import traceback

import numpy as np

from . import config, human, solver, vision
from .bus import LINK, AgentOffline
from .speed import Speed
from . import spots as spotsmod


class Stopped(Exception):
    pass


# ------------------------------------------------------------------ popups
# Ordre de priorite interne des popups/didacticiels (testes APRES les cas
# deja geres : menus, avis, serie). Les popups recouvrent la grille : elles
# sont donc balayees AVANT de jouer, sinon on appuierait a travers l'overlay.
POPUP_ORDER = ("loss", "tutorial", "tip", "hint", "friends", "bonus", "win")


def pick_popup(by_group):
    """Rend le 1er type de popup present (dict {nom: hit}) selon POPUP_ORDER."""
    for name in POPUP_ORDER:
        if name in by_group:
            return name
    return None


def halve_batch(batch, div):
    """Taille de salve effective apres un Perdu : divise par `div`, min 1."""
    try:
        div = max(1, int(div))
    except Exception:
        div = 1
    try:
        return max(1, int(batch) // div)
    except Exception:
        return 1


def banned_from_cells(cells):
    """Cases rouges : {(r, c): {v}} — le jeu dit que ce chiffre ne va pas a
    cet emplacement, le solveur a interdiction de l'y replacer."""
    out = {}
    try:
        for row in cells or []:
            for x in row:
                if x.get("color") == "error" and int(x.get("v") or 0) > 0:
                    out.setdefault((int(x["r"]), int(x["c"])), set()).add(int(x["v"]))
    except Exception:
        pass
    return out


def verify_missing(missing, counts, hidden, N):
    """Verifie les chiffres absents du pave dans la grille actuelle.

    missing : chiffres requis par la solution mais sans touche.
    counts : {v: occurrences lues en grille} (meme base que le solveur).
    hidden : chiffres masques par le jeu (accessibilite : boutons natifs
             absents) -> le jeu les considere comme termines.
    Renvoie (confirmes, douteux) : un chiffre est confirme si on le compte
    N fois en grille (totalement reparti : normal qu'il soit masque) ou si
    le jeu lui-meme le masque. Les douteux viennent d'une detection ratee.
    """
    ok, ko = [], []
    for v in missing or []:
        try:
            full = int(counts.get(v, 0)) >= int(N)
        except Exception:
            full = False
        if full or (hidden and v in hidden):
            ok.append(v)
        else:
            ko.append(v)
    return ok, ko


class Controller(threading.Thread):
    def __init__(self, opts):
        super().__init__(daemon=True)
        self.opts = opts
        self.state = "running" if not opts.get("beta") else "idle"
        self.test = bool(opts.get("test"))
        self.msg = "Démarrage"
        self.games = 0
        self.moves = 0
        self.errors = 0
        self.unknown = 0
        self.just_finished = False
        self.last_finish = 0.0
        self.dead_top = 0
        self.shot_ms = 0
        self.last_rgb = None
        self.overlay = None
        self.thumb = None
        self.rev = 0
        self.keypad_cache = {}
        self.templates = vision.Templates()
        self.speed = Speed()
        self.speed.load_perf()
        self.logs = collections.deque(maxlen=600)
        self.seq = 0
        self.lock = threading.Lock()
        self.last_info = {}
        self.launched = False
        self.batch_div = 1      # penalite temporaire apres un Perdu (salves / 2, / 4... ; min 1 coup)
        self.last_grid_t = 0.0  # derniere grille vue (evite "back" en pleine partie)
        self._pop_last = None   # memoire croix-d'abord (tutoriel, bonus) : 1er passage = croix
        self._pop_n = 0
        self._last_action = None  # anti-blocage : derniere action effectuee
        self._action_n = 0        # ... et son nombre de repetitions consecutives
        self._post_game = False   # True entre une partie terminee et la grille suivante
        self._keypad_retries = 0  # essais de relecture quand une touche manque vraiment
        self._last_progress_t = time.time()  # watchdog : derniere activite utile
        self._loading_until = 0.0  # fin du chargement attendu (pas de watchdog avant)
        self._watchdog_ocr_t = 0.0
        self._watchdog_back_t = 0.0
        self._quick_tap_t = 0.0  # dernier appui sur l'interrupteur (cooldown)
        self._astuce_used = False  # clin d'oeil humain : une fois par partie max
        self._last_found = None  # dernier hit (pour memoriser les boutons)
        # Emplacements memorises (emplacements/*.jpg -> data/spots.json).
        # Vide en mode --auto (tout a l'OCR) ou sans references.
        self.auto_spots = bool(opts.get("auto"))
        self.spots = {} if self.auto_spots else spotsmod.load()
        self._home_done = False  # 1er « Nouvelle partie » (accueil) : toujours OCR
        self._spots_keys_ok = True  # pave spots verifie une fois par partie

    # ------------------------------------------------------------ journal / etat
    def log(self, msg, level="info"):
        with self.lock:
            self.seq += 1
            self.logs.append({"seq": self.seq, "t": time.strftime("%H:%M:%S"), "lvl": level, "msg": msg})
        print("[%s] %s" % (level, msg), flush=True)

    def say(self, msg):
        self.msg = msg
        self.log(msg)

    def control(self, action, **kw):
        if action in ("start", "resume", "play"):
            self.state = "running"
            self.say("Lecture")
        elif action == "pause":
            self.state = "paused"
            self.say("Pause")
        elif action == "toggle":
            return self.control("pause" if self.state == "running" else "start")
        elif action == "stop":
            self.state = "stopped"
            self.say("Arrêté")
        elif action == "speed":
            if self.speed.set_choice(kw.get("level", "auto")):
                self.log("Vitesse : " + self.speed.label())
        elif action == "test":
            self.test = bool(kw.get("value", not self.test))
            self.log("Mode test : %s" % self.test)
        elif action == "forget_templates":
            self.templates.clear_src("learned")
            self.templates.save()
            self.log("Gabarits appris effacés")
        elif action == "reset_keypad":
            self.keypad_cache.clear()
            self.log("Cache du pavé effacé")
        return self.status()

    def status(self):
        return {
            "state": self.state, "msg": self.msg, "games": self.games, "moves": self.moves,
            "errors": self.errors, "speed_choice": self.speed.choice, "speed_effective": self.speed.effective(),
            "speed_label": self.speed.label(), "choices": self.speed.choices(), "agent": LINK.online(),
            "agent_info": LINK.info, "rev": self.rev, "test": self.test, "shot_ms": self.shot_ms,
            "templates": {"learned": self.templates.count("learned"), "keypad": self.templates.count("keypad"),
                          "font": self.templates.count("font")},
            "perf": self.speed.history[-12:],
            "batch_div": self.batch_div,
            "post_game": self._post_game,
            "spots": {"auto": self.auto_spots,
                      "enabled": config.section("menus").get("spots_enabled", True),
                      "refs": sorted(spotsmod.find_refs().keys()),
                      "have": sorted(k for k in (self.spots or {}) if not k.startswith("_")),
                      "stale": spotsmod.is_stale() or spotsmod.needs_calibration(self.spots)},
        }

    def checkpoint(self):
        while self.state in ("paused", "idle"):
            time.sleep(0.15)
        if self.state == "stopped":
            raise Stopped()

    def sleep(self, ms):
        end = time.time() + ms / 1000.0
        while True:
            self.checkpoint()
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(left, 0.1))

    def rand_ms(self, v):
        if isinstance(v, (list, tuple)):
            return random.uniform(v[0], v[1])
        return float(v or 0)

    def countdown_sleep(self, ms, what="Attend"):
        """Attente VISIBLE : affiche 'Attend Ns...' (N...1) sur le bandeau et
        la page web, par pas d'1 s. Insensible a pause/reprise (checkpoint)."""
        import math
        try:
            total = max(0, int(ms))
        except Exception:
            total = 0
        if total <= 0:
            return
        n = max(1, int(math.ceil(total / 1000.0)))
        self.log("%s %ds..." % (what, n))
        end = time.time() + total / 1000.0
        last_shown = None
        while True:
            self.checkpoint()
            left = end - time.time()
            if left <= 0:
                break
            cur = max(1, int(math.ceil(left)))
            if cur != last_shown:
                last_shown = cur
                self.msg = "%s %ds..." % (what, cur)
            # Via self.sleep (mockable en tests, et respecte pause/reprise).
            self.sleep(min(left, 0.1) * 1000.0)

    def _debug_overlay(self, *args, **kw):
        """Image de debug (JPEG) : sautee si debug.overlay=false (gain reel :
        pas d'encodage JPEG a chaque cycle/salve sur Termux)."""
        if not config.section("debug").get("overlay", True):
            return
        self.overlay = vision.render(*args, **kw)

    # ------------------------------------------------------------ primitives agent
    def shot(self, stable=False):
        vcfg = config.section("vision")
        prev = None
        tries = vcfg.get("stable_max_tries", 4) if stable else 1
        rgb = None
        for _ in range(max(1, tries)):
            t0 = time.time()
            res = LINK.call("screenshot", quality=88, timeout=config.section("server").get("call_timeout_s", 25))
            if not res.get("ok"):
                raise AgentOffline(res.get("error", "capture impossible"))
            rgb = vision.to_rgb_array(base64.b64decode(res["image"]))
            self.shot_ms = int((time.time() - t0) * 1000)
            self.dead_top = int(res.get("dead", 0) or 0)
            self.last_info = res
            if prev is None or vision.frame_diff(prev, rgb) > vcfg.get("stable_diff", 2.5):
                prev = rgb
                if tries > 1:
                    time.sleep(0.09)
                    continue
            break
        self.last_rgb = rgb
        try:
            self.thumb = vision.thumb(rgb)
        except Exception:
            pass
        self.rev += 1
        return rgb

    def tap(self, x, y, dur=60):
        if self.test:
            self.log("(test) appui %d,%d" % (x, y))
            return {"ok": True}
        return LINK.call("tap", x=int(x), y=int(y), duration=int(dur))

    def tap_box(self, b):
        """Appui humain a l'interieur d'un rectangle [l,t,r,b] (rarement au bord)."""
        hcfg = config.section("human")
        try:
            edge_prob = float(hcfg.get("edge_prob", 0.05))
        except Exception:
            edge_prob = 0.05
        try:
            edge_max = float(hcfg.get("edge_max", 0.46))
        except Exception:
            edge_max = 0.46
        l, t, r, bb = b
        x, y = human.point_in((l + r) / 2.0, (t + bb) / 2.0, (r - l), (bb - t), 0.12, 0.28,
                              edge_prob, edge_max)
        lv = self.speed.level()
        return self.tap(x, y, human.tap_duration(lv))

    def tap_batch(self, taps):
        if self.test:
            self.log("(test) salve de %d appuis" % len(taps))
            return {"ok": True, "done": len(taps), "total": len(taps), "failed": 0, "lag_max": 0}
        span = (taps[-1][2] + taps[-1][3]) / 1000.0 if taps else 0
        return LINK.call("tap_batch", taps=taps, timeout=span + 15)

    # ------------------------------------------------------------ boucle principale
    def run(self):
        self.log("Contrôleur prêt (test=%s)" % self.test)
        while True:
            try:
                if self.state != "running":
                    time.sleep(0.2)
                    continue
                if not LINK.online():
                    self.msg = "En attente de l'agent AutoJs6…"
                    time.sleep(0.5)
                    continue
                self.cycle()
            except Stopped:
                time.sleep(0.2)
            except AgentOffline as e:
                self.say("Agent injoignable : %s" % e)
                time.sleep(2)
            except Exception as e:
                self.errors += 1
                self.log("Erreur : %s\n%s" % (e, traceback.format_exc()), "error")
                time.sleep(1.5)

    def ensure_game(self):
        g = config.section("game")
        pkg = g.get("package")
        cur = (LINK.info or {}).get("pkg")
        if g.get("auto_launch") and pkg and (not self.launched or (cur and cur != pkg and self.unknown >= 3)):
            self.launched = True
            if cur != pkg:
                self.say("Lancement du jeu " + pkg)
                if not self.test:
                    LINK.call("launch", package=pkg)
                self._mark_progress()
                self.sleep(g.get("launch_wait_ms", 4000))

    def cycle(self):
        self.checkpoint()
        self.ensure_game()
        # Menus : 1 seule capture non stabilisee (rapide). L'ancien
        # shot(stable=True) faisait jusqu'a 4 captures par cycle (~2-3 s)
        # pour rien sur des ecrans statiques. La stabilite reste verifiee
        # dans execute() (settle + relecture) et wait_for_grid().
        rgb = self.shot(stable=False)
        if vision.is_blank(rgb):
            self.say("Écran noir : relance du jeu")
            self.launched = False
            self.sleep(1500)
            return
        grid = None
        try:
            # A. Popups/didacticiels bloquants (accessibilite seule, ~0.5 s, sans
            # OCR) : ils recouvrent la grille (Perdu, tutoriels, astuces...), il
            # faut les fermer AVANT de jouer, sinon on appuie a travers l'overlay.
            # Ne s'applique que si aucun cas deja gere n'est present (le scan ne
            # couvre que les nouveaux titres), et ne coute qu'un appel rapide.
            blocked = self.scan_blocking(use_ocr=False)
            if blocked and self.handle_blocking(blocked):
                self.unknown = 0
                self._note_action("popup:" + blocked)
                return
        except AgentOffline:
            blocked = None
        except Stopped:
            raise
        except Exception as e:
            self.log("Scan popups : %s" % e, "warn")
        grid = vision.detect_grid(rgb, self.dead_top)
        try:
            # Watchdog : bloque sans progres depuis un moment (didacticiel
            # coince ?) -> suivant/terminer a 7 s, OCR + relance a 11 s.
            if self._watchdog(rgb, grid):
                self.unknown = 0
                return
        except AgentOffline:
            pass
        except Stopped:
            raise
        except Exception as e:
            self.log("Watchdog : %s" % e, "warn")
        if grid:
            self.unknown = 0
            self.last_grid_t = time.time()
            self._pop_reset()
            self._last_action, self._action_n = None, 0
            self._post_game = False
            # « Placement rapide » : si l'option est activee (bleu), on la
            # coupe d'un appui et on repart sur un screenshot frais.
            rgb = self._ensure_quick_off(rgb, grid)
            if self.play(rgb, grid) == "blocked":
                self.unknown = 0
                self._note_action("blocked")
            return
        self._debug_overlay(rgb)
        # Apres-match : drain rapide (1 scan accessibilite, hitboxes) avant
        # le chemin generique (plusieurs scans + OCR).
        if self._post_game:
            try:
                drained = self._postgame_drain()
            except AgentOffline:
                drained = None
            except Stopped:
                raise
            except Exception as e:
                self.log("Drain : %s" % e, "warn")
                drained = None
            if drained:
                self.unknown = 0
                self._note_action(drained)
                if drained == "debutant":
                    self.wait_for_grid()
                elif drained == "flamme":
                    self.sleep(400)
                return
        # Chemin rapide fusionne : 1 seul appel agent (1 seul OCR max) pour
        # tous les menus, au lieu de 3-4 appels separes (10-15 s par bouton).
        # Desactivable via menus.fused_scan=false (retombe sur l'ancien chemin).
        # handle_menus_fused() renvoie :
        #   - une action ('rate', 'flamme', ...) si tape,
        #   - False si scan neuf termine sans rien (on saute l'ancien chemin),
        #   - None si vieil agent / erreur (on garde l'ancien chemin).
        fused_done_legacy = False
        if config.section("menus").get("fused_scan", True):
            done = self.handle_menus_fused()
            if done:
                self.unknown = 0
                self._note_action(done)
                if done == "nouvelle_partie":
                    self._home_done = True  # l'accueil est passe (OCR)
                if done == "debutant":
                    self.wait_for_grid()
                elif done == "flamme":
                    # Apres la flamme vient (parfois !) l'ecran « Série
                    # quotidienne » + « retour ». s'il ne vient pas (serie deja
                    # allumee : 2e partie du jour et suivantes), le cycle
                    # suivant trouve directement « Nouvelle partie ».
                    self.sleep(400)
                return
            fused_done_legacy = (done is False)
        # La popup "Aimez-vous le Sudoku ? / Évaluez-nous" recouvre l'écran de
        # victoire : elle passe avant les menus, sinon on tape sur
        # « Nouvelle partie » à travers l'overlay (sans effet).
        if not fused_done_legacy:
            rate = self.handle_rate()
            if rate:
                self.unknown = 0
                self._note_action("rate")
                return
            name = self.handle_menu()
            if name:
                self.unknown = 0
                self._note_action(name)
                if name == "nouvelle_partie":
                    self._home_done = True  # l'accueil est passe (OCR)
                if name == "debutant":
                    self.wait_for_grid()
                return
            streak = self.handle_streak()
            if streak:
                self.unknown = 0
                self._note_action(streak)
                if streak == "flamme":
                    # après avoir touché la flamme, le jeu affiche l'écran "Série quotidienne"
                    self.sleep(400)
                return
        # Repli OCR (vues personnalisees) : si les textes des popups echappent a
        # l'accessibilite, on retente UNE fois avec OCR avant de declarer l'ecran
        # inconnu. Borne a partir de 2 echecs pour ne pas payer un OCR a chaque cycle.
        if self.unknown + 1 >= 2:
            try:
                blocked_ocr = self.scan_blocking(use_ocr=True)
                if blocked_ocr and self.handle_blocking(blocked_ocr):
                    self.unknown = 0
                    self._note_action("popup:" + blocked_ocr)
                    return
            except AgentOffline:
                pass
            except Stopped:
                raise
            except Exception as e:
                self.log("Scan popups OCR : %s" % e, "warn")
        # Repli aveugle post-victoire : si rien n'est reconnu peu après une partie
        # terminée, c'est (presque) toujours la popup d'avis dont les textes ont
        # échappé à l'accessibilité et à l'OCR → on tente la croix quand même.
        if self.unknown + 1 >= 2 and (time.time() - self.last_finish) < 180:
            self.say("Écran inconnu post-victoire → essai de la croix (popup d'avis ?)")
            self.tap_rate_close(None)
            mcfg0 = config.section("menus")
            self.sleep(self.rand_ms(mcfg0.get("rate_wait_after_ms", [1100, 1700])))
            self.unknown = 0
            self._note_action("rate_blind")
            return
        self.unknown += 1
        mcfg = config.section("menus")
        self.msg = "Écran inconnu (%d)" % self.unknown
        if self.unknown >= mcfg.get("unknown_max_tries", 10):
            self.unknown = 0
            # En pleine partie (grille vue recemment), un "back" ejecterait du
            # niveau pour rien (ex. tutoriel en vue personnalisee) : on attend.
            try:
                grace = float(mcfg.get("unknown_back_grace_s", 20))
            except Exception:
                grace = 20.0
            if time.time() - self.last_grid_t < grace:
                self.log("Écran inconnu en partie → simple attente (grâce %.0f s)" % grace)
                self.sleep(mcfg.get("unknown_wait_ms", 1200))
                return
            act = mcfg.get("unknown_action", "back")
            self.log("Écran inconnu persistant → %s" % act, "warn")
            if not self.test:
                if act == "back":
                    LINK.call("back")
                elif act == "launch":
                    self.launched = False
        self.sleep(mcfg.get("unknown_wait_ms", 1200))

    # ------------------------------------------------------------ menus
    @staticmethod
    def pick_menu_target(level_here, nouvelle_found, debutant_found):
        """Désambiguïsation victoire vs liste des difficultés.

        - Écran victoire : « Nouvelle partie » (bouton) + « Débutant » (libellé
          de difficulté), SANS « Facile »/« Moyen » → viser « Nouvelle partie ».
        - Liste des difficultés : titre « Nouvelle partie » + lignes « Débutant »,
          « Facile », « Moyen » → viser la ligne « Débutant ».
        """
        if level_here:
            if debutant_found:
                return "debutant"
            return "nouvelle_partie" if nouvelle_found else None
        if nouvelle_found:
            return "nouvelle_partie"
        return "debutant" if debutant_found else None

    def _menu_timeout(self):
        """Timeout unique pour la recherche de texte dans les menus (s).

        Avant : 20 s en dur a chaque appel (x3-4 par cycle). Maintenant :
        configurable via menus.find_timeout_s (defaut 10 s). L'accessibilite
        repond en < 1 s quand le texte existe ; le timeout ne penalise que
        les cas manques (OCR ~2-4 s).
        """
        try:
            return float(config.section("menus").get("find_timeout_s", 10))
        except Exception:
            return 10.0

    def _call_find_any(self, groups, exact, blacklist, skip_ocr=False, timeout=None):
        """Enveloppe unique pour tous les find_any (evite les doublons lents).

        Attention : LINK.call(ctype, timeout=...) consomme `timeout` comme
        delai d'attente cote serveur, il n'est PAS transmis a l'agent.
        On transmet donc le delai agent via `agent_timeout`, et on donne
        a LINK.call une marge (+12 s) pour laisser l'OCR repondre sans
        faux "agent injoignable".
        """
        if timeout is None:
            timeout = self._menu_timeout()
        try:
            timeout = float(timeout)
        except Exception:
            timeout = 10.0
        call_timeout = timeout + 12.0
        try:
            return LINK.call("find_any", timeout=call_timeout, groups=groups,
                             exact=exact, blacklist=blacklist,
                             agent_timeout=timeout,
                             skip_ocr=bool(skip_ocr))
        except AgentOffline:
            return None

    def _find_rule(self, rule, blacklist, timeout=None):
        res = self._call_find_any([rule.get("texts", [])],
                                  [bool(rule.get("exact"))],
                                  blacklist, timeout=timeout)
        if res is None:
            return None
        return res if res.get("found") else None

    def _tap_rule(self, rule, res, wait=None):
        """`wait` : attente apres-appui forcee (0 = aucune, l'appelant gere)."""
        self.say("Menu « %s » (%s via %s)" % (rule["name"], res.get("text"), res.get("via")))
        self.sleep(self.rand_ms([250, 600]))
        self.tap_box(res["b"])
        self.countdown_sleep(self.rand_ms(rule.get("wait_after_ms", 1200) if wait is None else wait))
        return rule["name"]

    def handle_menus_fused(self):
        """Cycle menus rapide : UN seul appel agent pour tout (1 OCR max).

        Groupes (ordre du scan) : avis > flamme > retour > nouvelle >
        debutant > niveaux. DECISION en revanche : avis, puis « Nouvelle
        partie » / « Débutant » (si « Nouvelle partie » est visible, la serie
        quotidienne est absente : on l'enchaine direct et on skippe la
        serie), puis seulement flamme > retour (serie optionnelle : 1re
        partie du jour ; les suivantes n'ont pas cet ecran).
        L'agent renvoie `hits` (1 hit par groupe trouve) avec une seule
        capture OCR partagee, au lieu de 3-4 captures+OCR separes
        (~10-15 s par bouton avant).
        Renvoie l'action ('rate', 'flamme', 'retour', 'nouvelle_partie',
        'debutant'), False si scan neuf sans rien (sauter l'ancien chemin),
        ou None si vieil agent / erreur (garder l'ancien chemin).
        """
        mcfg = config.section("menus")
        blacklist = mcfg.get("blacklist", [])
        by_name = {r.get("name"): r for r in mcfg.get("rules", [])}
        r_deb = by_name.get("debutant")
        r_new = by_name.get("nouvelle_partie")
        rate_texts = mcfg.get("rate_texts") or ["Aimez-vous", "Evaluez-nous", "Noter"]
        flame_texts = mcfg.get("streak_flame_texts") or ["Touchez la flamme", "enflammez"]
        retour_texts = mcfg.get("streak_retour_texts") or ["retour"]
        level_texts = mcfg.get("level_texts") or ["Facile", "Moyen", "Easy", "Medium"]
        new_texts = (r_new.get("texts", []) if r_new else ["Nouvelle partie", "New game"])
        deb_texts = (r_deb.get("texts", []) if r_deb else ["Debutant", "Beginner"])
        groups = [rate_texts, flame_texts, retour_texts, new_texts, deb_texts, level_texts]
        exact = [False] * len(groups)
        res = self._call_find_any(groups, exact, blacklist)
        if res is None:
            return None
        if not res.get("found"):
            # Scan neuf termine (hits=[] ou absent mais found=False) : rien
            # a l'ecran. On tente quand meme les secours manuels, puis on
            # signale "scan fait" (False) pour eviter de refaire 3 OCR.
            tapped = self._menus_fallback_taps()
            return tapped if tapped else False
        hits = res.get("hits")
        if not isinstance(hits, list):
            # Vieil agent (sans `hits`) : on ne sait pas quels autres groupes
            # sont presents -> retomber sur l'ancien chemin multi-appels.
            # On tape quand meme le groupe trouve si c'est un menu simple.
            try:
                g0 = int(res.get("group", -1))
                if g0 == 0:
                    h0 = res
                    self.say("Popup d'avis (%s via %s) → fermeture (croix)" % (h0.get("text"), h0.get("via")))
                    self.sleep(self.rand_ms([250, 600]))
                    self.tap_rate_close(h0.get("b"))
                    self.sleep(self.rand_ms(mcfg.get("rate_wait_after_ms", [1100, 1700])))
                    return "rate"
            except Exception:
                pass
            return None
        by_group = {}
        for h in hits:
            try:
                by_group[int(h.get("group"))] = h
            except Exception:
                pass
        # 1) Popup d'avis (prioritaire : recouvre l'ecran de victoire)
        if 0 in by_group:
            h = by_group[0]
            self.say("Popup d'avis (%s via %s) → fermeture (croix)" % (h.get("text"), h.get("via")))
            self.sleep(self.rand_ms([250, 600]))
            self.tap_rate_close(h.get("b"))
            self.sleep(self.rand_ms(mcfg.get("rate_wait_after_ms", [1100, 1700])))
            return "rate"
        # 2) « Nouvelle partie » / « Débutant » EN PREMIER : si « Nouvelle
        # partie » est visible, il n'y a PAS de menu serie (2e partie du jour
        # et suivantes) : on l'enchaine directement et on skippe la serie.
        # (Desambiguise victoire vs liste des difficultes via niveaux.)
        level_here = 5 in by_group
        nouvelle_found = 3 in by_group
        debutant_found = 4 in by_group
        target = self.pick_menu_target(level_here, nouvelle_found, debutant_found)
        if target == "nouvelle_partie" and r_new is not None and 3 in by_group:
            return self._tap_rule(r_new, by_group[3])
        if target == "debutant" and r_deb is not None and 4 in by_group:
            return self._tap_rule(r_deb, by_group[4])
        # 3) Serie quotidienne (optionnelle : 1re partie du jour seulement) :
        # flamme a allumer, puis bouton retour/continuer.
        if mcfg.get("streak_enabled", True):
            if 1 in by_group:
                h = by_group[1]
                return self._tap_streak_flame(h)
            if 2 in by_group:
                return self._tap_streak_retour(by_group[2])
        if not level_here and not nouvelle_found and not debutant_found:
            return False
        tapped = self._menus_fallback_taps()
        return tapped if tapped else False

    def _menus_fallback_taps(self):
        """Secours : coordonnees manuelles si le texte est introuvable."""
        mcfg = config.section("menus")
        fb = mcfg.get("fallback_taps", {}) or {}
        if self.last_rgb is not None:
            H, W = self.last_rgb.shape[:2]
            for rule in mcfg.get("rules", []):
                xy = fb.get(rule["name"])
                if isinstance(xy, (list, tuple)) and len(xy) == 2:
                    self.say("Menu « %s » (coordonnées manuelles)" % rule["name"])
                    x, y = xy[0] * W, xy[1] * H
                    self.tap_box([x - 30, y - 15, x + 30, y + 15])
                    self.sleep(self.rand_ms(rule.get("wait_after_ms", 1200)))
                    return rule["name"]
        return None

    def _tap_streak_flame(self, h):
        mcfg = config.section("menus")
        if self.last_rgb is None:
            return None
        H, W = self.last_rgb.shape[:2]
        self.say("Série : rallumage de la flamme (%s via %s)" % (h.get("text"), h.get("via")))
        self.sleep(self.rand_ms([250, 600]))
        pos = mcfg.get("streak_flame_pos") or [0.5, 0.32]
        try:
            fx, fy = float(pos[0]), float(pos[1])
        except Exception:
            fx, fy = 0.5, 0.32
        x, y = fx * W, fy * H
        dx, dy = W * 0.12, H * 0.08
        self.tap_box([x - dx, y - dy, x + dx, y + dy])
        self.sleep(self.rand_ms(mcfg.get("streak_flame_wait_after_ms", [1500, 2200])))
        return "flamme"

    def _tap_streak_retour(self, h):
        mcfg = config.section("menus")
        txt = str(h.get("text") or "")
        self.say("Série quotidienne (%s via %s)" % (txt, h.get("via")))
        self.sleep(self.rand_ms([250, 600]))
        low = txt.lower()
        if "retour" in low or "continu" in low:
            # Le hit EST le bouton (« retour » ou « continuer ») : on l'appuie.
            self.tap_box(h["b"])
        else:
            # 2e recherche SANS OCR (accessibilite seule, < 1 s) : les boutons
            # "retour" / "continuer" sont des vrais boutons Android, l'OCR est
            # inutile ici. Avant : un 2e find_any avec OCR (~2-4 s) a chaque
            # ecran serie.
            tapped = False
            r2 = self._call_find_any([["retour", "Retour", "RETOUR",
                                       "continuer", "Continuer", "Continue"]],
                                     [False], mcfg.get("blacklist", []),
                                     skip_ocr=True)
            if r2 is None:
                return None
            if r2.get("found"):
                self.tap_box(r2["b"])
                tapped = True
            if not tapped and self.last_rgb is not None:
                H, W = self.last_rgb.shape[:2]
                pos = mcfg.get("streak_retour_pos") or [0.5, 0.92]
                try:
                    rx, ry = float(pos[0]), float(pos[1])
                except Exception:
                    rx, ry = 0.5, 0.92
                x, y = rx * W, ry * H
                self.tap_box([x - W * 0.35, y - 40, x + W * 0.35, y + 40])
        self.sleep(self.rand_ms(mcfg.get("streak_retour_wait_after_ms", [1200, 1800])))
        return "retour"

    def _postgame_drain(self):
        """Apres-match : UN seul scan accessibilite (sans OCR) pour tout.

        Les titres vus, les boutons partent par hitboxes (spots/apprises) :
        aucun OCR dans le cas courant. Si l'accessibilite ne voit rien,
        renvoie False et le chemin generique (avec OCR) prend le relais.
        Renvoie l'action, False si rien vu, None si agent muet.
        """
        mcfg = config.section("menus")
        bl = mcfg.get("blacklist", [])
        by_name = {r.get("name"): r for r in mcfg.get("rules", [])}
        r_new = by_name.get("nouvelle_partie")
        r_deb = by_name.get("debutant")
        groups = [
            mcfg.get("rate_texts") or ["Aimez-vous", "Evaluez-nous"],          # 0
            mcfg.get("win_title_texts") or ["Bien joue"],                     # 1
            mcfg.get("bonus_title_texts") or ["Bonus"],                       # 2
            mcfg.get("tutorial_title_texts") or ["Dernier chiffre possible"],  # 3
            mcfg.get("tip_title_texts") or ["Astuce"],                        # 4
            mcfg.get("hint_title_texts") or ["Indices gratuits"],             # 5
            mcfg.get("friends_title_texts") or ["avec des amis"],             # 6
            mcfg.get("streak_flame_texts") or ["Touchez la flamme"],          # 7
            mcfg.get("streak_retour_texts") or ["retour"],                    # 8
            (r_new.get("texts", []) if r_new else ["Nouvelle partie"]),       # 9
            (r_deb.get("texts", []) if r_deb else ["Debutant"]),              # 10
            mcfg.get("level_texts") or ["Facile", "Moyen"],                   # 11
        ]
        res = self._call_find_any(groups, [False] * len(groups), bl, skip_ocr=True)
        if res is None:
            return None
        if not res.get("found"):
            return False
        hits = res.get("hits")
        by_group = {}
        if isinstance(hits, list):
            for h in hits:
                try:
                    by_group[int(h.get("group"))] = h
                except Exception:
                    pass
        else:
            try:
                by_group[int(res.get("group", -1))] = res
            except Exception:
                pass
        if not by_group:
            return False
        wait = mcfg.get("dismiss_wait_after_ms", [1100, 1700])
        # 1) Avis (recouvre la victoire) : bouton appris, sinon la croix.
        if 0 in by_group:
            if self._tap_learned_or_ocr("rate", "dismiss",
                                        mcfg.get("rate_dismiss_texts") or [],
                                        mcfg.get("rate_wait_after_ms", [1100, 1700]),
                                        "avis (bouton)"):
                return "rate"
            self.say("Popup d'avis → fermeture (croix)")
            self.sleep(self.rand_ms([250, 600]))
            self.tap_rate_close(None)
            self.sleep(self.rand_ms(mcfg.get("rate_wait_after_ms", [1100, 1700])))
            return "rate"
        # 2) Popups bloquantes (boutons appris ou recherche, sans OCR ici).
        if 1 in by_group and self.handle_win():
            return "win"
        if 2 in by_group and self.handle_bonus():
            return "bonus"
        if 3 in by_group and self.handle_tutorial():
            return "tutorial"
        if 4 in by_group and self.handle_tip():
            return "tip"
        if 5 in by_group and self.handle_hint():
            return "hint"
        if 6 in by_group and self.handle_friends():
            return "friends"
        # 3) Menus : « Nouvelle partie » d'abord (pas de serie), via hitboxes.
        target = self.pick_menu_target(11 in by_group, 9 in by_group, 10 in by_group)
        if target == "nouvelle_partie" and r_new is not None and 9 in by_group:
            sp = self.spot_box("fin_partie", "nouvelle_partie")
            if sp is not None:
                self.say("Menu « nouvelle_partie » (mémorisé)")
                self.sleep(self.rand_ms([250, 600]))
                self.tap_box(sp)
                self._home_done = True
                sp_deb = self.spot_box("niveaux", "debutant")
                if sp_deb is not None:
                    # Deux hitboxes connues : enchainement DIRECT nouvelle ->
                    # debutant (attente visible), sans cycle ni scan supplementaire.
                    self.countdown_sleep(self.rand_ms(mcfg.get("loss_wait_after_ms", [850, 1050])))
                    self.sleep(self.rand_ms([250, 600]))
                    self.tap_box(sp_deb)
                    self.countdown_sleep(self.rand_ms(
                        (r_deb or {}).get("wait_after_ms", [600, 900])))
                    return "debutant"
                self._wait_seen(r_deb.get("texts", []) if r_deb else [],
                                mcfg.get("debutant_appear_ms")
                                or r_new.get("wait_after_ms", [1100, 1700]),
                                "menu difficulté")
                return "nouvelle_partie"
            self._home_done = True
            done = self._tap_rule(r_new, by_group[9], wait=0)
            self._wait_seen(r_deb.get("texts", []) if r_deb else [],
                            mcfg.get("debutant_appear_ms")
                            or r_new.get("wait_after_ms", [1100, 1700]),
                            "menu difficulté")
            return done
        if target == "debutant" and r_deb is not None and 10 in by_group:
            sp = self.spot_box("niveaux", "debutant")
            if sp is not None:
                self.say("Menu « débutant » (mémorisé)")
                self.sleep(self.rand_ms([250, 600]))
                self.tap_box(sp)
                self.countdown_sleep(self.rand_ms(r_deb.get("wait_after_ms", [600, 900])))
                return "debutant"
            return self._tap_rule(r_deb, by_group[10])
        # 4) Serie optionnelle (flamme / retour).
        if mcfg.get("streak_enabled", True):
            if 7 in by_group:
                return self._tap_streak_flame(by_group[7])
            if 8 in by_group:
                return self._tap_streak_retour(by_group[8])
        return False

    def handle_menu(self):
        mcfg = config.section("menus")
        rules = mcfg.get("rules", [])
        if not rules:
            return None
        by_name = {r.get("name"): r for r in rules}
        r_deb = by_name.get("debutant")
        r_new = by_name.get("nouvelle_partie")
        blacklist = mcfg.get("blacklist", [])
        # Cas général : les deux règles « Débutant » / « Nouvelle partie » coexistent
        # sur deux écrans différents → on lève l'ambiguïté via « Facile »/« Moyen ».
        if r_deb is not None and r_new is not None:
            level_texts = mcfg.get("level_texts") or ["Facile", "Moyen", "Easy", "Medium"]
            lv = self._call_find_any([level_texts], [False], blacklist)
            if lv is None:
                return None
            level_here = bool(lv.get("found"))
            if level_here:
                hit = self._find_rule(r_deb, blacklist)
                if hit:
                    return self._tap_rule(r_deb, hit)
                hit = self._find_rule(r_new, blacklist)
                if hit:
                    return self._tap_rule(r_new, hit)
            else:
                hit = self._find_rule(r_new, blacklist)
                if hit:
                    return self._tap_rule(r_new, hit)
                hit = self._find_rule(r_deb, blacklist)
                if hit:
                    return self._tap_rule(r_deb, hit)
        else:
            res = self._call_find_any([r.get("texts", []) for r in rules],
                                      [bool(r.get("exact")) for r in rules],
                                      blacklist)
            if res is None:
                return None
            if res.get("found"):
                return self._tap_rule(rules[int(res["group"])], res)
        # secours : coordonnees manuelles
        fb = mcfg.get("fallback_taps", {}) or {}
        if self.last_rgb is not None:
            H, W = self.last_rgb.shape[:2]
            for rule in rules:
                xy = fb.get(rule["name"])
                if isinstance(xy, (list, tuple)) and len(xy) == 2:
                    self.say("Menu « %s » (coordonnées manuelles)" % rule["name"])
                    x, y = xy[0] * W, xy[1] * H
                    self.tap_box([x - 30, y - 15, x + 30, y + 15])
                    self.sleep(self.rand_ms(rule.get("wait_after_ms", 1200)))
                    return rule["name"]
        return None

    def tap_rate_close(self, ref_box=None):
        """Appui sur la croix (✕) de la popup d'avis (noter.jpg).

        La croix n'a aucun texte : on vise une position relative configurable
        (`menus.rate_close_pos`, défaut [0.5, 0.625] calibré sur 1080x2340).
        Si `ref_box` (bouton « Évaluez-nous », coordonnées capture) est fourni,
        on reprend son centre en x (la carte est centrée) pour rester robuste.
        """
        if self.last_rgb is None:
            return None
        mcfg = config.section("menus")
        H, W = self.last_rgb.shape[:2]
        pos = mcfg.get("rate_close_pos") or [0.5, 0.625]
        try:
            fx, fy = float(pos[0]), float(pos[1])
        except Exception:
            fx, fy = 0.5, 0.625
        x = (ref_box[0] + ref_box[2]) / 2.0 if ref_box else fx * W
        y = fy * H
        x = max(2, min(W - 2, x))
        y = max(2, min(H - 2, y))
        dx, dy = max(60, W * 0.07), 60
        return self.tap_box([x - dx, y - dy, x + dx, y + dy])

    def handle_rate(self):
        """Popup post-victoire « Aimez-vous le Sudoku ? / Évaluez-nous » (noter.jpg).

        Détecte les textes de la popup puis tape sur sa croix (✕) sous la carte.
        Renvoie 'rate' si traité, None sinon.
        """
        mcfg = config.section("menus")
        rate_texts = mcfg.get("rate_texts") or [
            "Aimez-vous", "Evaluez-nous", "Évaluez-nous", "Evaluez", "Évaluez", "Evaluate",
        ]
        res = self._call_find_any([rate_texts], [False],
                                  mcfg.get("blacklist", []))
        if res is None:
            return None
        if not res.get("found"):
            return None
        # Variante avec bouton textuel ("Non merci", "Plus tard"...) : prioritaire
        # sur la croix quand il est visible.
        dismiss = mcfg.get("rate_dismiss_texts") or []
        if dismiss and self._tap_learned_or_ocr("rate", "dismiss", dismiss, mcfg.get("rate_wait_after_ms", [1100, 1700]),
                                                  "avis (bouton)"):
            return "rate"
        self.say("Popup d'avis (%s via %s) → fermeture (croix)" % (res.get("text"), res.get("via")))
        self.sleep(self.rand_ms([250, 600]))
        self.tap_rate_close(res.get("b"))
        self.sleep(self.rand_ms(mcfg.get("rate_wait_after_ms", [1100, 1700])))
        return "rate"

    def handle_streak(self):
        """Écrans de fin de niveau « série » : flamme à rallumer puis bouton retour.

        - streek-1 : « Touchez la flamme, enflammez votre série ! » → toucher la flamme.
        - streek-2 : « Série quotidienne / Bien joué » + bouton « retour » → toucher retour.
        Renvoie 'flamme' / 'retour' si traité, None sinon.
        """
        mcfg = config.section("menus")
        if not mcfg.get("streak_enabled", True):
            return None
        flame_texts = mcfg.get("streak_flame_texts") or [
            "Touchez la flamme", "enflammez", "votre série", "votre serie",
        ]
        retour_texts = mcfg.get("streak_retour_texts") or [
            "Série quotidienne", "Serie quotidienne", "Bien joué", "Bien joue",
            "constance", "retour",
        ]
        res = self._call_find_any([flame_texts, retour_texts],
                                  [False, False],
                                  mcfg.get("blacklist", []))
        if res is None:
            return None
        if not res.get("found"):
            return None
        grp = int(res.get("group", -1))
        if self.last_rgb is None:
            return None
        H, W = self.last_rgb.shape[:2]
        if grp == 0:
            return self._tap_streak_flame(res)
        if grp == 1:
            return self._tap_streak_retour(res)
        return None

    # ------------------------------------------------------------ popups / didacticiels
    def _blocking_defs(self):
        """Groupes de titres balayes en 1 appel (ordre = POPUP_ORDER)."""
        mcfg = config.section("menus")
        return [
            ("loss", mcfg.get("loss_enabled", True),
             mcfg.get("loss_title_texts") or ["Perdu"]),
            ("tutorial", mcfg.get("tutorial_enabled", True),
             mcfg.get("tutorial_title_texts") or ["Dernier chiffre possible"]),
            ("tip", mcfg.get("tip_enabled", True),
             mcfg.get("tip_title_texts") or ["Astuce intelligente"]),
            ("hint", mcfg.get("hint_enabled", True),
             mcfg.get("hint_title_texts") or ["Indices gratuits"]),
            ("friends", mcfg.get("friends_enabled", True),
             mcfg.get("friends_title_texts") or ["Jouer avec des amis"]),
            ("bonus", mcfg.get("bonus_enabled", True),
             mcfg.get("bonus_title_texts") or ["Bonus"]),
            ("win", mcfg.get("win_enabled", True),
             mcfg.get("win_title_texts") or ["Bien joué"]),
        ]

    def scan_blocking(self, use_ocr=False):
        """Balaye les titres des popups en UN appel agent.

        use_ocr=False : accessibilite seule, sans capture (~0.5 s) → appele a
        chaque cycle avant de jouer. True : + OCR (vues personnalisees) →
        repli occasionnel seulement. Renvoie le nom du type ou None.
        """
        mcfg = config.section("menus")
        if not mcfg.get("popups_enabled", True):
            return None
        defs = [(n, t) for (n, on, t) in self._blocking_defs() if on and t]
        if not defs:
            return None
        res = self._call_find_any([t for (_, t) in defs], [False] * len(defs),
                                  mcfg.get("blacklist", []), skip_ocr=not use_ocr)
        if res is None or not res.get("found"):
            return None
        hits = res.get("hits")
        if isinstance(hits, list) and hits:
            by_group = {}
            for h in hits:
                try:
                    by_group[defs[int(h.get("group", -1))][0]] = h
                except Exception:
                    pass
            if by_group:
                return pick_popup(by_group)
        # Vieil agent (sans `hits`) : un seul groupe connu.
        try:
            return defs[int(res.get("group", -1))][0]
        except Exception:
            return None

    def handle_blocking(self, name):
        """Ferme la popup `name` (vue par scan_blocking). Renvoie True si traite."""
        handler = {
            "loss": self.handle_loss,
            "tutorial": self.handle_tutorial,
            "tip": self.handle_tip,
            "hint": self.handle_hint,
            "friends": self.handle_friends,
            "bonus": self.handle_bonus,
            "win": self.handle_win,
        }.get(name)
        if handler is None:
            return False
        try:
            return bool(handler())
        except AgentOffline:
            return False

    def _pop_step(self, key):
        """Memoire 'croix d'abord' (popup bonus) : 1er passage consecutif = 1,
        puis 2, 3... (les didacticiels ne s'en servent plus : jamais de croix)."""
        if self._pop_last == key:
            self._pop_n += 1
        else:
            self._pop_last, self._pop_n = key, 1
        return self._pop_n

    def _pop_reset(self):
        self._pop_last, self._pop_n = None, 0

    # ------------------------------------------------------------ emplacements memorises
    def spots_on(self):
        """Hitboxes utilisables ? (config + pas de mode --auto + calibre)."""
        try:
            on = bool(config.section("menus").get("spots_enabled", True))
        except Exception:
            on = True
        return on and not self.auto_spots and bool(self.spots)

    def spot_box(self, ref, name):
        """Hitbox memorisee en pixels capture [l,t,r,b], ou None."""
        if not self.spots_on():
            return None
        try:
            box = (self.spots.get(ref) or {}).get(name)
            if not box or self.last_rgb is None:
                return None
            H, W = self.last_rgb.shape[:2]
            return spotsmod.box_frac_to_px(box, W, H)
        except Exception:
            return None

    def spots_grid(self, rgb, grid):
        """Grille memorisee (pixels), si meme taille et treillis visible."""
        if not self.spots_on():
            return None
        try:
            gf = (self.spots.get("game") or {}).get("grid")
            if not gf or int(gf.get("n")) != int(grid["n"]):
                return None
            H, W = rgb.shape[:2]
            g2 = spotsmod.grid_frac_to_px(gf, W, H)
            if not spotsmod.lattice_present(rgb, g2):
                return None
            return g2
        except Exception:
            return None

    def keypad_from_spots(self, rgb, grid):
        """Touches aux emplacements memorises (slots 1..9, ordre fixe : pas de
        compaction sur ce telephone, les chiffres termines laissent un trou).
        Renvoie (touches, hidden_slots) ou (None, set())."""
        if not self.spots_on():
            return None, set()
        try:
            slots = (self.spots.get("game") or {}).get("keys") or []
            N = grid["n"]
            if len(slots) != N:
                return None, set()
            H, W = rgb.shape[:2]
            try:
                ink_min = float(config.section("menus").get("spots_slot_ink", 0.01))
            except Exception:
                ink_min = 0.01
            keys = spotsmod.present_slots(rgb, slots, W, H, ink_min=ink_min)
            if not keys:
                return None, set()
            present = set(k["v"] for k in keys)
            hidden = set(range(1, N + 1)) - present
            # Glyphes pour la lecture des cases (comme detect_keypad).
            vcfg = config.section("vision")
            for kd in keys:
                bl = {"x0": kd["x"] - kd["w"] / 2.0, "x1": kd["x"] + kd["w"] / 2.0,
                      "y0": kd["y"] - kd["h"] / 2.0, "y1": kd["y"] + kd["h"] / 2.0,
                      "h": kd["h"]}
                patch = vision._pad_crop(rgb, bl)
                an = vision.analyse_patch(patch, dict(vcfg, ink_min_frac=0.002,
                                                      glyph_min_h=0.3),
                                          max(1, patch.shape[0]))
                if an:
                    kd["glyph"] = an["glyph"]
                    kd["aspect"] = an["aspect"]
            # Verifie une fois par partie que l'ordre n'a pas bouge
            # (compaction ?) : sinon on retombe sur la detection classique.
            if self._spots_keys_ok:
                symbols = vision.symbols_for(N, vcfg)
                self.templates.ensure_fonts(symbols, vcfg)
                bad = 0
                for kd in keys:
                    if "glyph" not in kd:
                        continue
                    rk = self.templates.match(kd["glyph"], kd["aspect"], set(symbols))
                    if rk and rk[0][1] > 0.30 and rk[0][0] != symbols[kd["v"] - 1]:
                        bad += 1
                if bad:
                    self.log("Pave spots incoherent (%d touches) : detection classique" % bad,
                             "warn")
                    self._spots_keys_ok = False
                    return None, set()
            return keys, hidden
        except Exception:
            return None, set()

    def _mark_progress(self):
        """Signale une activite utile (coup joue, nouvel ecran traite...)."""
        self._last_progress_t = time.time()

    def _note_action(self, name):
        """Anti-blocage : note l'action traitee ; si la MEME action revient
        `menus.stuck_repeats` fois de suite (l'appui ne change pas l'ecran :
        flamme manquee, croix ratee, bouton inactif...), on escalade : on
        retente l'autre branche (memoire popups reinitialisee) puis, si on est
        hors partie (post-victoire ou grille plus vue depuis un moment), un
        appui « retour » qui referme casi tout (flamme, serie, avis, pubs).
        Renvoie True si escalade effectuee."""
        if name == self._last_action:
            self._action_n += 1
        else:
            self._last_action, self._action_n = name, 1
            self._mark_progress()  # un nouvel ecran traite = ca avance
        mcfg = config.section("menus")
        try:
            reps = max(2, int(mcfg.get("stuck_repeats", 4)))
        except Exception:
            reps = 4
        if self._action_n < reps:
            return False
        self._action_n = 0
        self._pop_reset()
        if name.startswith("popup:"):
            # Bouton appris peut-etre perime : on l'oublie, l'OCR re-apprendra.
            self._forget_learned(name.split(":", 1)[1])
        try:
            grace = float(mcfg.get("unknown_back_grace_s", 20))
        except Exception:
            grace = 20.0
        # « Retour » seulement hors partie : juste apres une victoire (la
        # flamme/serie est optionnelle : si elle ne vient pas, on revient a
        # l'ecran de victoire puis « Nouvelle partie ») ou grille perdue de
        # vue depuis longtemps. En pleine partie on reessaie juste l'autre
        # branche (croix -> bouton et inversement), jamais de retour.
        can_back = self._post_game or (time.time() - self.last_grid_t >= grace)
        self.log("Blocage : '%s' %d fois sans effet → %s"
                 % (name, reps, "retour" if can_back else "autre branche"),
                 "warn")
        if can_back and mcfg.get("stuck_back", True) and not self.test:
            try:
                LINK.call("back")
            except AgentOffline:
                pass
            self.sleep(1200)
        else:
            self.sleep(800)
        return True

    def _tap_first_text(self, texts, wait_after_ms=1200, what="bouton"):
        """Appuie sur le 1er texte visible (accessibilite d'abord, OCR sinon)."""
        mcfg = config.section("menus")
        if not texts:
            return False
        for skip in (True, False):
            try:
                res = self._call_find_any([list(texts)], [False],
                                          mcfg.get("blacklist", []), skip_ocr=skip)
            except AgentOffline:
                return False
            if res is None:
                return False
            if res.get("found") and res.get("b"):
                self.say("Popup (%s : %s via %s)" % (what, res.get("text"), res.get("via")))
                self.sleep(self.rand_ms([250, 600]))
                self.tap_box(res["b"])
                self._last_found = res
                self.countdown_sleep(self.rand_ms(wait_after_ms))
                return True
        return False

    def _tap_learned_or_ocr(self, popup, role, texts, wait, what):
        """Bouton deja vu (OCR precedent) ? On le retape direct, sans scan
        ni OCR. Sinon recherche classique + memorisation de la position
        (fractions, persistees dans spots.json) pour la prochaine fois."""
        mcfg = config.section("menus")
        try:
            learn_on = bool(mcfg.get("learn_buttons", True))
        except Exception:
            learn_on = True
        learned = None
        if learn_on:
            try:
                learned = (self.spots.get("learned") or {}).get(popup + ":" + role)
            except Exception:
                learned = None
        if learned and self.last_rgb is not None:
            try:
                H, W = self.last_rgb.shape[:2]
                box = spotsmod.box_frac_to_px(learned, W, H)
                self.say("Popup (%s : bouton memorise)" % what)
                self.sleep(self.rand_ms([250, 600]))
                self.tap_box(box)
                self.sleep(self.rand_ms(wait))
                return True
            except Exception:
                learned = None
        self._last_found = None
        if not self._tap_first_text(texts, wait, what):
            return False
        # Memorise la position trouvee (accessibilite ou OCR : stable aussi).
        try:
            hit = self._last_found
            if learn_on and hit is not None and hit.get("b") and self.last_rgb is not None:
                H, W = self.last_rgb.shape[:2]
                l, t, r, bb = hit["b"]
                box = {"x": ((l + r) / 2.0) / W, "y": ((t + bb) / 2.0) / H,
                       "w": (r - l) / W, "h": (bb - t) / H}
                if all(0.0 <= v <= 1.0 for v in box.values()):
                    learned_all = dict(self.spots.get("learned") or {})
                    learned_all[popup + ":" + role] = box
                    self.spots["learned"] = learned_all
                    spotsmod.save(self.spots)
                    self.log("Bouton memorise : %s:%s" % (popup, role))
        except Exception as e:
            self.log("Memorisation bouton : %s" % e, "warn")
        return True

    def _forget_learned(self, popup):
        """Oublie les boutons appris d'une popup (position perimee ?) :
        le prochain passage re-apprendra par OCR."""
        try:
            learned = dict(self.spots.get("learned") or {})
            dropped = [k for k in learned if k.split(":")[0] == popup]
            for k in dropped:
                del learned[k]
            if dropped:
                if learned:
                    self.spots["learned"] = learned
                else:
                    self.spots.pop("learned", None)
                spotsmod.save(self.spots)
                self.log("Boutons oublies : %s (re-apprentissage OCR)" % popup)
        except Exception:
            pass

    def _tap_frac_pos(self, pos, what="croix", dx_frac=0.07, dy=60):
        """Appui centre sur une position en fraction d'ecran [x, y]."""
        if self.last_rgb is None:
            return False
        try:
            fx, fy = float(pos[0]), float(pos[1])
        except Exception:
            return False
        H, W = self.last_rgb.shape[:2]
        x = max(2, min(W - 2, fx * W))
        y = max(2, min(H - 2, fy * H))
        dx = max(60, W * dx_frac)
        self.say("Popup (%s en [%.2f, %.2f])" % (what, fx, fy))
        self.sleep(self.rand_ms([250, 600]))
        self.tap_box([x - dx, y - dy, x + dx, y + dy])
        mcfg = config.section("menus")
        self.sleep(self.rand_ms(mcfg.get("dismiss_wait_after_ms", [1100, 1700])))
        return True

    def _tap_spot_or_text(self, ref, name, texts, wait, what, pos_key=None):
        """Appuie via l'hitbox memorisee si presente, sinon via l'OCR
        (ancien code, intact) puis position manuelle. Renvoie 'spots',
        'ocr', 'pos' ou None. L'attente qui suit est VISIBLE (compte a rebours)."""
        sp = self.spot_box(ref, name)
        if sp is not None:
            self.say("Menu « %s » (%s memorise)" % (what, ref))
            self.sleep(self.rand_ms([250, 600]))
            self.tap_box(sp)
            self.countdown_sleep(self.rand_ms(wait))
            return "spots"
        if self._tap_first_text(texts, wait, what):
            return "ocr"
        if pos_key:
            pos = config.section("menus").get(pos_key)
            if pos and self._tap_frac_pos(pos, what + " (position)"):
                return "pos"
        return None

    def _tap_loss_new_game(self, texts, wait):
        """« Nouvelle partie » de l'ecran Perdu : le bouton du popup n'est PAS
        a l'emplacement habituel (fin_partie) -> recherche du texte
        (accessibilite puis OCR) et memorisation « loss:new_game » pour les
        fois suivantes. JAMAIS l'hitbox fin_partie ici (elle vise le bouton
        de l'ecran de victoire, pas celui du popup). Position manuelle
        `loss_new_game_pos` en dernier recours."""
        if self._tap_learned_or_ocr("loss", "new_game", texts, wait,
                                    "nouvelle partie (Perdu)"):
            return "ocr"
        pos = config.section("menus").get("loss_new_game_pos")
        if pos and self._tap_frac_pos(pos, "nouvelle partie (Perdu, position)"):
            return "pos"
        return None

    def _start_new_game(self, from_loss=False):
        """Enchaine « Nouvelle partie » puis « Débutant ».

        - from_loss=True (ecran « Perdu ») : le bouton « Nouvelle partie »
          est celui du POPUP : position differente de l'emplacement
          fin_partie -> toujours via le texte (accessibilite/OCR), puis
          position apprise « loss:new_game » ; jamais l'hitbox fin_partie.
        - Le tout 1er « Nouvelle partie » de la session (accueil) passe
          TOUJOURS par l'OCR ; ensuite les hitboxes memorisees si presentes
          (sinon l'OCR, code intact).
        - Apres « Débutant » + ~0.5 s : si la serie/flamme apparait, on la
          ferme via OCR puis on re-appuie (hitboxes) + delai, et on joue.
        Renvoie True si enchaine, False si introuvable.
        """
        mcfg = config.section("menus")
        # L'attente apres « Nouvelle partie » est ADAPTATIVE (_wait_seen) :
        # « Debutant » est tape des qu'il est visible (plafond
        # debutant_appear_ms), au lieu de ~1 s fixes.
        wait = 0
        appear = mcfg.get("debutant_appear_ms") or mcfg.get("loss_wait_after_ms", [850, 1050])
        new_texts = mcfg.get("loss_new_game_texts") or ["Nouvelle partie", "New game"]
        tapped = None
        if from_loss:
            tapped = self._tap_loss_new_game(new_texts, wait)
        elif self._home_done:
            tapped = self._tap_spot_or_text("fin_partie", "nouvelle_partie",
                                            new_texts, wait, "nouvelle partie",
                                            pos_key="loss_new_game_pos")
        else:
            # Ecran d'accueil : OCR d'abord (puis hitboxes aux prochains tours).
            if self._tap_first_text(new_texts, wait, "nouvelle partie (accueil)"):
                tapped = "ocr"
            else:
                tapped = self._tap_spot_or_text("fin_partie", "nouvelle_partie",
                                                new_texts, wait, "nouvelle partie",
                                                pos_key="loss_new_game_pos")
        if not tapped:
            self.log("Bouton « Nouvelle partie » introuvable", "warn")
            return False
        self._home_done = True
        self._astuce_used = False  # nouvelle partie : nouveau clin d'oeil possible
        self._spots_keys_ok = True  # re-verifie le pave spots (nouvelle grille)
        # Débutant : hitbox si presente, sinon OCR (regle existante).
        by_name = {r.get("name"): r for r in mcfg.get("rules", [])}
        r_deb = by_name.get("debutant")
        if r_deb is None:
            r_deb = {"name": "debutant", "texts": ["Débutant", "Debutant", "Beginner"],
                     "wait_after_ms": [1800, 2600]}
        if self.spot_box("niveaux", "debutant") is not None:
            # Emplacement connu : attente VISIBLE (config) puis appui direct,
            # aucun scan (l'emplacement est deja calibre).
            self.countdown_sleep(self.rand_ms(mcfg.get("loss_wait_after_ms", [850, 1050])))
        else:
            self._wait_seen(r_deb.get("texts", []), appear, "menu difficulté")
        deb = self._tap_spot_or_text("niveaux", "debutant",
                                     r_deb.get("texts", []),
                                     r_deb.get("wait_after_ms", [1800, 2600]),
                                     "débutant")
        if not deb:
            hit = self._find_rule(r_deb, mcfg.get("blacklist", []))
            if hit:
                self._tap_rule(r_deb, hit)
        # Recheck serie/flamme ~0.5 s apres (optionnelle du jour) : si elle
        # est la, on la ferme via OCR puis on re-appuie + delai, et on joue.
        try:
            fw = mcfg.get("postgame_flame_wait_ms", [400, 600])
        except Exception:
            fw = [400, 600]
        self.sleep(self.rand_ms(fw))
        try:
            rgb = self.shot()
        except AgentOffline:
            return True
        if vision.detect_grid(rgb, self.dead_top):
            self._pop_reset()
            return True
        streak = self.handle_streak()
        if streak:
            self._note_action(streak)
            self.sleep(self.rand_ms(fw))
            self._tap_spot_or_text("fin_partie", "nouvelle_partie",
                                   new_texts, wait, "nouvelle partie (apres serie)",
                                   pos_key="loss_new_game_pos")
            if self.spot_box("niveaux", "debutant") is not None:
                self.countdown_sleep(self.rand_ms(mcfg.get("loss_wait_after_ms", [850, 1050])))
            else:
                self._wait_seen(r_deb.get("texts", []), appear, "menu difficulté")
            self._tap_spot_or_text("niveaux", "debutant",
                                   r_deb.get("texts", []),
                                   r_deb.get("wait_after_ms", [1800, 2600]),
                                   "débutant (apres serie)")
            self.sleep(self.rand_ms(fw))
            self.wait_for_grid()
            self._pop_reset()
            return True
        self.wait_for_grid()
        self._pop_reset()
        return True

    def handle_loss(self):
        """Ecran 'Perdu' (3 erreurs) : « Nouvelle partie » puis « Débutant ».

        Divise en plus les salves par 2 de maniere temporaire (x2 a chaque Perdu
        consecutif, remonte d'un cran a chaque partie gagnee) jusqu'a 1 coup
        par salve : moins d'appuis aveugles quand la lecture a echoue.
        """
        mcfg = config.section("menus")
        if mcfg.get("loss_halve_batch", True):
            self.batch_div = min(64, self.batch_div * 2)
            self.log("Perdu : salves divisees (/%d)" % self.batch_div, "warn")
        else:
            self.say("Écran Perdu → nouvelle partie")
        return self._start_new_game(from_loss=True)

    def handle_tutorial(self):
        """Didacticiels ('Dernier chiffre possible', ...) : JAMAIS la croix,
        uniquement « suivant » puis/ou « valider ».

        Les etapes s'enchainent (suivant -> valider -> ...) : apres chaque
        appui on refait un scan (rapide, accessibilite) pour voir si une
        autre etape est apparue, et on l'enchaine aussitot (borne par
        tutorial_max_steps). Quand il n'y a plus rien, le cycle suivant
        refait un scan frais (capture + relecture) avant de relancer le
        remplissage automatique.
        """
        mcfg = config.section("menus")
        self._pop_reset()
        try:
            max_steps = max(1, int(mcfg.get("tutorial_max_steps", 4)))
        except Exception:
            max_steps = 4
        texts = mcfg.get("tutorial_next_texts") or ["suivant", "valider"]
        titles = mcfg.get("tutorial_title_texts") or ["Dernier chiffre possible"]
        wait = mcfg.get("dismiss_wait_after_ms", [1100, 1700])
        for _ in range(max_steps):
            tapped = self._tap_learned_or_ocr("tutorial", "next", texts, wait, "didacticiel (suivant/valider)")
            if not tapped:
                pos = mcfg.get("tutorial_next_pos")
                tapped = bool(pos) and self._tap_frac_pos(pos, "didacticiel (bouton bas)")
            if not tapped:
                self.log("Didacticiel : bouton introuvable", "warn")
                return False
            # Re-scan : etape suivante (valider...) ou termine ?
            try:
                again = self._call_find_any([titles], [False],
                                            mcfg.get("blacklist", []), skip_ocr=True)
            except AgentOffline:
                return True
            if again is None or not again.get("found"):
                break
        # Yeux frais pour la suite : le surlignage du didacticiel a pu changer
        # les glyphes, on re-detectera pave + cases au prochain cycle.
        self.keypad_cache.clear()
        self._keypad_retries = 0
        return True

    def handle_tip(self):
        """Popup 'Astuce intelligente' (sans croix fonctionnelle) : « OK »."""
        mcfg = config.section("menus")
        self._pop_reset()
        return self._tap_learned_or_ocr("tip", "ok", mcfg.get("tip_ok_texts") or ["OK"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "astuce (OK)")

    def handle_hint(self):
        """Popup 'Indices gratuits' : 'Passer' / 'Non merci' en priorite, sinon
        'Regarder' (a eviter si possible : lance une video)."""
        mcfg = config.section("menus")
        self._pop_reset()
        if self._tap_learned_or_ocr("hint", "close", mcfg.get("hint_close_texts") or ["Passer"],
                                      mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                      "indice (passer)"):
            return True
        return self._tap_learned_or_ocr("hint", "watch", mcfg.get("hint_watch_texts") or ["Regarder"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "indice (regarder)")

    def handle_friends(self):
        """Popup 'Jouer avec vos amis' (PARTAGEZ + Fermer, sans croix) : « Fermer »."""
        mcfg = config.section("menus")
        self._pop_reset()
        return self._tap_learned_or_ocr("friends", "close", mcfg.get("friends_close_texts") or ["Fermer"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "amis (fermer)")

    def handle_bonus(self):
        """Popup 'Bonus ...' : croix / 'Fermer' d'abord, sinon « Récupérer »."""
        mcfg = config.section("menus")
        step = self._pop_step("bonus")
        if step <= 1:
            if self._tap_learned_or_ocr("bonus", "close", mcfg.get("bonus_close_texts") or ["Fermer"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "bonus (fermer)"):
                return True
            pos = mcfg.get("bonus_close_pos")
            if pos and self._tap_frac_pos(pos, "bonus (croix)"):
                return True
        return self._tap_learned_or_ocr("bonus", "claim", mcfg.get("bonus_claim_texts") or ["Récupérer"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "bonus (récupérer)")

    def handle_win(self):
        """Popup 'Bien joué !' : bouton de fermeture (souvent « OK »)."""
        mcfg = config.section("menus")
        self._pop_reset()
        return self._tap_learned_or_ocr("win", "ok", mcfg.get("win_ok_texts") or ["OK"],
                                          mcfg.get("dismiss_wait_after_ms", [1100, 1700]),
                                          "bien joué (OK)")

    # ------------------------------------------------------------ watchdog anti-blocage
    def _tutorial_bottom_check(self):
        """Didacticiel coince ? Cherche un bouton « suivant » dans le BAS de
        l'ecran (accessibilite seule, rapide). Si oui : c'est le didacticiel,
        on appuie « terminer » (ou on enchaine la suite)."""
        mcfg = config.section("menus")
        if self.last_rgb is None:
            return False
        try:
            frac = float(mcfg.get("tutorial_bottom_frac", 0.72))
        except Exception:
            frac = 0.72
        try:
            res = LINK.call("ui_scan", max_items=400, timeout=15)
        except AgentOffline:
            return False
        H = self.last_rgb.shape[0]
        for it in res.get("items", []) or []:
            try:
                t = str(it.get("text") or "")
                b = it.get("b")
                if not b or "suivant" not in t.lower().replace("é", "e"):
                    continue
                if (b[1] + b[3]) / 2.0 < frac * H:
                    continue
                self.say("Didacticiel confirme (suivant en bas) → terminer")
                self.sleep(self.rand_ms([250, 600]))
                self.tap_box(b)
                self.sleep(self.rand_ms(mcfg.get("dismiss_wait_after_ms", [1100, 1700])))
                # La fin dit souvent « terminer » : on tente d'abord ce texte,
                # sinon on enchaine normalement (suivant -> valider -> ...).
                ends = mcfg.get("tutorial_end_texts") or ["terminer", "Terminer"]
                if not self._tap_first_text(ends, mcfg.get("dismiss_wait_after_ms",
                                                           [1100, 1700]),
                                            "didacticiel (terminer)"):
                    self.handle_tutorial()
                return True
            except Stopped:
                raise
            except AgentOffline:
                return False
            except Exception:
                continue
        return False

    def _watchdog(self, rgb, grid):
        """Surveille l'absence de progres (coups, ecrans traites...).

        - > watchdog_stuck_s (defaut 7 s) : cherche « suivant » en bas
          (didacticiel coince ?) -> terminer + suite.
        - > watchdog_lost_s (defaut 11 s) : OCR de localisation pour savoir
          exactement ou l'on en est ; sans grille : on relance/continue une
          nouvelle partie, sinon simple balayage didacticiel (jamais de
          relance en plein niveau).
        Renvoie True si quelque chose a ete fait (sauter le reste du cycle).
        """
        mcfg = config.section("menus")
        if not mcfg.get("watchdog_enabled", True):
            return False
        now = time.time()
        if now < self._loading_until:
            return False  # chargement en cours : normal de ne rien faire
        try:
            s1 = float(mcfg.get("watchdog_stuck_s", 7))
        except Exception:
            s1 = 7.0
        try:
            s2 = float(mcfg.get("watchdog_lost_s", 11))
        except Exception:
            s2 = 11.0
        idle = now - self._last_progress_t
        if idle < s1:
            return False
        if self._tutorial_bottom_check():
            self._note_action("tuto_watchdog")
            return True
        if idle < s2:
            return False
        if grid is not None:
            # En plein niveau : pas de relance, juste un balayage OCR
            # didacticiel (borne par cooldown).
            try:
                cool = float(mcfg.get("watchdog_ocr_cooldown_s", 8))
            except Exception:
                cool = 8.0
            if now - self._watchdog_ocr_t < cool:
                return False
            self._watchdog_ocr_t = now
            self.log("Toujours bloque en niveau (%.0f s) : balayage OCR" % idle, "warn")
            try:
                b = self.scan_blocking(use_ocr=True)
                if b and self.handle_blocking(b):
                    self._note_action("popup:" + b)
                    return True
            except AgentOffline:
                pass
            except Stopped:
                raise
            except Exception as e:
                self.log("Watchdog OCR : %s" % e, "warn")
            return False
        return self._lost_sweep()

    def _lost_sweep(self):
        """On est bloque SANS grille : 1 OCR pour savoir ou l'on en est,
        puis on relance/continue une nouvelle partie si possible, sinon un
        « retour » borne par cooldown (pubs, ecrans inconnus...)."""
        mcfg = config.section("menus")
        bl = mcfg.get("blacklist", [])
        by_name = {r.get("name"): r for r in mcfg.get("rules", [])}
        r_new = by_name.get("nouvelle_partie")
        r_deb = by_name.get("debutant")
        new_texts = (r_new.get("texts", []) if r_new else ["Nouvelle partie", "New game"])
        deb_texts = (r_deb.get("texts", []) if r_deb else ["Débutant", "Debutant"])
        flame_texts = mcfg.get("streak_flame_texts") or ["Touchez la flamme"]
        retour_texts = mcfg.get("streak_retour_texts") or ["retour", "continuer"]
        tuto_titles = mcfg.get("tutorial_title_texts") or ["Dernier chiffre possible"]
        tuto_btns = mcfg.get("tutorial_next_texts") or ["suivant", "valider", "terminer"]
        rate_texts = mcfg.get("rate_texts") or ["Aimez-vous", "Evaluez-nous"]
        groups = [new_texts, flame_texts + retour_texts, tuto_titles + tuto_btns,
                  rate_texts, deb_texts]
        try:
            res = self._call_find_any(groups, [False] * len(groups), bl)
        except AgentOffline:
            return False
        if res is None:
            return False
        grp = None
        hits = res.get("hits") if res.get("found") else None
        if isinstance(hits, list) and hits:
            try:
                grp = min(int(h.get("group", 99)) for h in hits)
            except Exception:
                grp = None
        elif res.get("found"):
            try:
                grp = int(res.get("group", -1))
            except Exception:
                grp = None
        if grp == 0:
            self.say("Relance : « Nouvelle partie » retrouve")
            if self._start_new_game():
                self._note_action("nouvelle_partie")
                return True
            return False
        if grp == 1:
            self.say("Relance : ecran serie retrouve")
            if self.handle_streak():
                self._note_action("serie_relance")
                return True
            return False
        if grp == 2:
            if self.handle_tutorial():
                self._note_action("tuto_relance")
                return True
            return False
        if grp == 3:
            if self.handle_rate():
                self._note_action("rate")
                return True
            return False
        if grp == 4 and r_deb is not None:
            if self._tap_rule(r_deb, res):
                self.wait_for_grid()
                self._note_action("debutant")
                return True
            return False
        # Vraiment perdu de vue : un « retour » de temps en temps (pub, etc.),
        # borne par cooldown, jamais en pleine partie (ici : pas de grille).
        try:
            cool = float(mcfg.get("watchdog_back_cooldown_s", 20))
        except Exception:
            cool = 20.0
        now = time.time()
        if now - self._watchdog_back_t < cool:
            return False
        if not mcfg.get("stuck_back", True) or self.test:
            return False
        try:
            grace = float(mcfg.get("unknown_back_grace_s", 20))
        except Exception:
            grace = 20.0
        if not self._post_game and (now - self.last_grid_t < grace):
            return False
        self.log("Perdu de vue : retour (cooldown %.0f s)" % cool, "warn")
        try:
            LINK.call("back")
        except AgentOffline:
            return False
        self._watchdog_back_t = now
        self.sleep(1200)
        return True

    def wait_for_grid(self, max_ms=None):
        """Attend la grille apres appui sur « Debutant » (lancement partie).

        Le delai initial est reglable par UNE seule variable simple :
        `game.detect_delay_ms` (defaut 2500 ms). Augmente-la si la grille
        met du temps a apparaitre (telephone lent / animation), diminue-la
        pour enchaine plus vite. Repli sur menus.loading_wait_ms si absente.
        `max_ms` borne le total (express post-partie : court).
        """
        mcfg = config.section("menus")
        gcfg = config.section("game")
        try:
            initial_ms = float(gcfg.get("detect_delay_ms", mcfg.get("loading_wait_ms", 1500)))
        except Exception:
            initial_ms = 1500.0
        self.say("Chargement de la partie… (detection dans %d ms)" % int(initial_ms))
        try:
            total = float(max_ms if max_ms is not None else mcfg.get("loading_max_ms", 15000))
        except Exception:
            total = 15000.0
        end = time.time() + total / 1000.0
        self._loading_until = end  # le watchdog ne s'en mele pas pendant ce temps
        self.sleep(initial_ms)
        while time.time() < end:
            rgb = self.shot()
            if vision.detect_grid(rgb, self.dead_top):
                self.sleep(gcfg.get("grid_found_wait_ms", 400))
                return True
            self.sleep(500)
        return False

    # ------------------------------------------------------------ jeu
    def _grid_counts(self, cells, N):
        """Compte les occurrences par chiffre dans les cases lues (score correct exigé).

        Les chiffres ROUGES ne comptent pas : le jeu dit qu'ils sont mal
        places, ils ne doivent pas servir (pave partiel, verification...)."""
        cnt = {v: 0 for v in range(1, N + 1)}
        try:
            for row in cells or []:
                for x in row:
                    v = int(x.get("v") or 0)
                    if 1 <= v <= N and float(x.get("score", 0)) >= 0.30 \
                            and x.get("color") != "error":
                        cnt[v] += 1
        except Exception:
            pass
        return cnt

    def get_keypad(self, rgb, grid, cells=None):
        """Detecte le pave. Renvoie (touches, masques) : `masques` = chiffres
        absents des boutons natifs (accessibilite) quand elle repond, c'est-
        a-dire consideres comme termines par le jeu (touches masquees)."""
        H, W = rgb.shape[:2]
        N = grid["n"]
        key = (W, H, grid["n"], int(grid["x0"] / 8), int(grid["y1"] / 8))
        vcfg = config.section("vision")
        counts = self._grid_counts(cells, N) if cells else None
        skeys, shidden = self.keypad_from_spots(rgb, grid)
        vkeys = None
        if skeys is None:
            vkeys = vision.detect_keypad(rgb, grid, self.templates, vcfg, counts=counts)
        else:
            self.log("Pavé spots : %d touches" % len(skeys))
        akeys = None
        if vcfg.get("use_accessibility_keypad", True):
            # l'accessibilité donne les valeurs exactes même quand le pavé est partiel
            try:
                akeys = self.keypad_from_accessibility(grid)
            except Exception:
                akeys = None
        hidden = set(range(1, N + 1)) - set(k["v"] for k in akeys or [])
        if akeys is None:
            # Sans accessibilite : les slots absents des hitboxes = masques.
            hidden = set(shidden) if skeys is not None else set()
        keys = None
        detected = skeys if skeys is not None else vkeys
        if detected and akeys:
            # fusion par valeur : positions detectees/memorisees prioritaires,
            # valeurs accessibilité en renfort pour les touches manquées
            by_v = {k["v"]: k for k in akeys}
            for k in detected:
                by_v[k["v"]] = k
            keys = [by_v[v] for v in sorted(by_v)]
            srcs = sorted(set(k.get("src", "?") for k in keys))
            for k in keys:
                if k.get("src") != "accessibilité":
                    k["src"] = "+".join(srcs) if len(srcs) > 1 else k.get("src")
        elif detected:
            keys = detected
        elif akeys:
            keys = akeys
        elif key in self.keypad_cache:
            # ultime secours : le pavé mémorisé, mais REVALIDE (positions
            # potentiellement périmées ou fausses : toolbar...). Jamais
            # d'appuis aveugles sur un pave perime.
            keys = self._validate_keys(self.keypad_cache[key], grid)
            if keys:
                self.log("Pavé : réutilisation du cache (%d touches)" % len(keys), "warn")
                return keys, hidden
            self.log("Pavé en cache invalide : re-detection", "warn")
            del self.keypad_cache[key]
        if keys:
            keys = self._validate_keys(keys, grid)
        if keys:
            self.keypad_cache[key] = keys
            ys = [k["y"] for k in keys]
            self.log("Pavé détecté : %d/%d touches (%s, y=%.1f..%.1fch)" % (
                len(keys), N, keys[0].get("src"),
                (min(ys) - grid["y1"]) / grid["ch"], (max(ys) - grid["y1"]) / grid["ch"]))
            vision.keypad_to_templates(keys, self.templates, vision.symbols_for(grid["n"], vcfg))
        return keys, hidden

    def _validate_keys(self, keys, grid):
        """Garde-fous finaux sur les touches (toutes sources : vision,
        accessibilite, cache). Une touche VALIDE est dans la zone pave
        (bien sous la grille, jamais sur la toolbar ni le toggle) et porte
        de l'encre bleue (seuls les chiffres a placer sont bleus et grands).
        Renvoie la liste filtree (vide = pave inutilisable, pas d'appuis)."""
        if not keys:
            return []
        vcfg = config.section("vision")
        N = grid["n"]
        try:
            min_row = float(vcfg.get("keypad_min_row_ch", 2.5))
        except Exception:
            min_row = 2.5
        want_blue = bool(vcfg.get("keypad_acc_blue", True))
        y1, ch = grid["y1"], grid["ch"]
        out = []
        for k in keys:
            try:
                y = float(k.get("y", -1))
            except Exception:
                continue
            if N <= 9 and y < y1 + min_row * ch:
                continue  # toolbar / toggle : jamais le pave (pave a ~3.2ch)
            if self.last_rgb is not None and want_blue:
                try:
                    w, h = float(k.get("w", 0)), float(k.get("h", 0))
                except Exception:
                    w, h = 0, 0
                if not self._is_keypad_blue(float(k.get("x", 0)), y, w, h):
                    continue
            out.append(k)
        if out and len(out) != len(keys):
            self.log("Pavé : %d/%d touches rejetees (zone/encre)" %
                     (len(keys) - len(out), len(keys)), "warn")
        return out

    def keypad_from_accessibility(self, grid):
        try:
            res = LINK.call("ui_scan", max_items=400, timeout=15)
        except AgentOffline:
            return None
        N = grid["n"]
        items = res.get("items", []) or []
        # Bande de la rangee d'outils (Annuler/Effacer/Crayon/Astuce, avec son
        # badge compteur) : JAMAIS le pave. Sans elle, un appui vise au pavé
        # pouvait atterrir sur « Astuce intelligente » et ouvrir l'aide en boucle.
        mcfg = config.section("menus")
        tools = [t.lower() for t in (mcfg.get("toolbar_texts") or []) if t]
        band_top, band_bot = None, None
        if tools:
            for it in items:
                try:
                    t = str(it.get("text") or "").lower()
                    b = it.get("b")
                    if b and any(w and w in t for w in tools):
                        band_top = b[1] if band_top is None else min(band_top, b[1])
                        band_bot = b[3] if band_bot is None else max(band_bot, b[3])
                except Exception:
                    continue
        vcfg = config.section("vision")
        try:
            min_ratio = float(vcfg.get("keypad_acc_min_h_ratio", 0.6))
        except Exception:
            min_ratio = 0.6
        cands = []
        for it in items:
            t = (it.get("text") or "").strip()
            b = it.get("b")
            if not b or not t:
                continue
            # le texte peut contenir le chiffre + le compteur restant ("8\n2") : ne garder que le 1er chiffre
            digits = "".join(ch for ch in t.split()[0] if ch.isdigit()) if t.split() else ""
            if not digits:
                continue
            try:
                v = int(digits[:2] if N > 9 else digits[0])
            except Exception:
                continue
            cy = (b[1] + b[3]) / 2.0
            if not (1 <= v <= N and cy > grid["y1"]):
                continue
            if band_top is not None and band_top <= cy <= band_bot:
                continue  # dans la rangee d'outils (badge, libelle...) : pas une touche
            cands.append({"v": v, "x": (b[0] + b[2]) / 2.0, "y": cy,
                          "w": b[2] - b[0], "h": b[3] - b[1], "src": "accessibilité"})
        if not cands:
            return None
        # Encre BLEUE obligatoire (comme les vraies touches) : les chiffres
        # a placer sont les seuls ecrits en bleu et les plus grands. Le badge
        # (blanc/bleu marine) et les compteurs (gris) sous les touches sont
        # ainsi exclus meme s'ils portent un chiffre. Verifie sur la derniere
        # capture (coordonnees capture = celles de l'accessibilite).
        vcfg = config.section("vision")
        if self.last_rgb is not None and vcfg.get("keypad_acc_blue", True):
            kept = [c for c in cands if self._is_keypad_blue(c["x"], c["y"], c["w"], c["h"])]
            if kept:
                cands = kept
        # Micro-noeuds (badge compteur...) : hauteur netement sous la mediane.
        hs = sorted(c["h"] for c in cands)
        med_h = hs[len(hs) // 2]
        if len(cands) >= 2:
            cands = [c for c in cands if c["h"] >= min_ratio * med_h]
        if not cands:
            return None
        # Une seule rangee (N <= 9) : les compteurs sont SOUS les chiffres,
        # le badge est AU-DESSUS. On garde la rangee principale.
        if N <= 9 and len(cands) >= 2:
            ys = sorted(c["y"] for c in cands)
            row_y = ys[len(ys) // 2]
            main = [c for c in cands if abs(c["y"] - row_y) <= 0.8 * med_h]
            if main:
                cands = main
        found = {}
        for c in cands:
            v = c["v"]
            if v not in found or c["y"] > found[v]["y"]:
                found[v] = c
        if 1 <= len(found) <= N:
            # partiel accepté : le jeu masque les chiffres déjà entièrement placés
            return [found[v] for v in sorted(found)]
        return None

    def _is_keypad_blue(self, x, y, w, h):
        """Encre bleue autour du point (capture) ? Echantillonne le coeur de
        la touche (les 8% les plus sombres = l'encre, pas le fond blanc)."""
        try:
            H, W = self.last_rgb.shape[:2]
            x0 = max(0, int(x - w * 0.3))
            x1 = min(W, int(x + w * 0.3) + 1)
            y0 = max(0, int(y - h * 0.3))
            y1 = min(H, int(y + h * 0.3) + 1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                return True  # trop petit pour juger : on ne rejette pas
            sub = self.last_rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.int32)
            if len(sub) == 0:
                return True
            dark = sub[np.argsort(sub.mean(axis=1))[:max(4, len(sub) // 12)]]
            r, g, b = (float(v) for v in np.median(dark, axis=0))
            return b > 110 and (b - r) > 40 and (b - g) > 30
        except Exception:
            return True  # en cas de doute on ne rejette pas

    def _handle_missing_keys(self, moves, keys, hidden, cells, N):
        """Filtre les coups dont la touche est absente du pave. Ne bloque JAMAIS :
        - chiffre confirme (compte N fois en grille ou masque par le jeu) :
          normal, on joue simplement le reste ;
        - chiffre douteux (ni complet ni masque : detection ratee ?) : on
          retente une lecture bornee (cache vide + chasse aux popups qui
          recouvrent le pave), puis on joue le reste de toute facon.
        Renvoie les coups jouables (liste vide = relecture au prochain cycle).
        """
        vcfg = config.section("vision")
        try:
            max_retries = max(0, int(vcfg.get("keypad_missing_retries", 3)))
        except Exception:
            max_retries = 3
        have = set(k["v"] for k in keys or [])
        need = set(v for (_, _, v) in moves)
        missing = sorted(need - have)
        if not missing:
            self._keypad_retries = 0
            return moves
        counts = self._grid_counts(cells, N)
        ok, ko = verify_missing(missing, counts, hidden or set(), N)
        if ok:
            detail = ", ".join("%d (%d/%d en grille%s)" % (
                v, counts.get(v, 0), N,
                ", masque par le jeu" if hidden and v in hidden else "")
                for v in ok)
            self.log("Pave partiel %d/%d : %s deja complet(s), on joue le reste"
                     % (len(keys or []), N, detail))
        if ko:
            self._keypad_retries += 1
            detail = ", ".join("%d (%d/%d en grille)" % (v, counts.get(v, 0), N)
                                for v in ko)
            if self._keypad_retries <= max_retries:
                self.log("Touche(s) %s introuvable(s) : nouvel essai %d/%d "
                         "(pas de blocage)" % (detail, self._keypad_retries, max_retries),
                         "warn")
                self.keypad_cache.clear()
                try:
                    b = self.scan_blocking(use_ocr=False)
                    if b:
                        self.handle_blocking(b)
                except AgentOffline:
                    pass
                except Stopped:
                    raise
                except Exception as e:
                    self.log("Chasse popup pave : %s" % e, "warn")
                self.sleep(900)
            else:
                self.log("Touche(s) %s toujours introuvable(s) : on joue sans "
                         "(pas de blocage)" % detail, "warn")
                self._keypad_retries = 0
        playable = [m for m in moves if m[2] in have]
        if playable:
            self._keypad_retries = 0
        else:
            # Rien de jouable avec ce pave : relecture au prochain cycle, et
            # surveillance anti-blocage (escalade si cela se repete).
            self._note_action("attente_touches")
        return playable

    def ocr_read(self, grid, cells):
        """Lecture de secours via l'OCR d'AutoJs6 : complete/corrige les cases peu sures."""
        try:
            res = LINK.call("ocr_region", region=[int(grid["x0"]), int(grid["y0"]), int(grid["x1"]), int(grid["y1"])],
                            timeout=20)
        except AgentOffline:
            return 0
        N = grid["n"]
        changed = 0
        for it in res.get("items", []):
            txt = "".join(ch for ch in (it.get("text") or "") if ch.isdigit())
            b = it.get("b")
            if not txt or not b:
                continue
            parts = [txt] if (N > 9 and len(txt) <= 2 and int(txt) <= N) else list(txt)
            w = (b[2] - b[0]) / float(len(parts))
            for i, p in enumerate(parts):
                v = int(p)
                if not 1 <= v <= N:
                    continue
                cx = b[0] + w * (i + 0.5)
                cy = (b[1] + b[3]) / 2.0
                c = int((cx - grid["x0"]) / grid["cw"])
                r = int((cy - grid["y0"]) / grid["ch"])
                if 0 <= r < N and 0 <= c < N:
                    cell = cells[r][c]
                    if cell["v"] != v and cell["score"] < 0.75:
                        cell["alts"] = [(cell["v"], cell["score"])] + cell["alts"]
                        cell["v"] = v
                        cell["score"] = 0.74
                        changed += 1
        return changed

    def solve_cells(self, cells, br, bc, only_given=False):
        N = len(cells)
        banned = banned_from_cells(cells)
        g = [[0] * N for _ in range(N)]
        for row in cells:
            for x in row:
                if x["v"] > 0 and x["color"] != "error" and (not only_given or x["color"] == "given"):
                    g[x["r"]][x["c"]] = x["v"]
        tried = {}
        for _ in range(4 * N):
            bad = solver.conflicts(g, br, bc)
            if not bad:
                cnt, sol = solver.solve(g, br, bc, limit=2, timeout=6, banned=banned)
                if cnt >= 1:
                    violated = [(r, c) for (r, c), vals in banned.items()
                                if sol[r][c] in vals]
                    if not violated:
                        return g, sol, cnt
                    # La solution reprend un chiffre rouge : lecture
                    # incoherente, on reessaie une capture (jamais de coups).
                    self.log("Solution reprend un rouge en %s : relecture" %
                             ",".join("%d,%d" % (r + 1, c + 1) for r, c in violated),
                             "warn")
                    return None
                pool = [(r, c) for r in range(N) for c in range(N) if g[r][c]]
            else:
                pool = list(bad)
            if not pool:
                return None
            r, c = min(pool, key=lambda rc: cells[rc[0]][rc[1]]["score"] + 0.3 * tried.get(rc, 0))
            k = tried.get((r, c), 0)
            alts = [a for a in cells[r][c]["alts"] if a[1] > 0.3]
            g[r][c] = alts[k][0] if k < len(alts) else 0
            tried[(r, c)] = k + 1
        return None

    def play(self, rgb, grid):
        vcfg = config.section("vision")
        N, br, bc = grid["n"], grid["br"], grid["bc"]
        self.msg = "Grille %dx%d (blocs %dx%d)" % (N, N, br, bc)
        # 1er passage : pavé sans aide de la grille, puis lecture des cases
        keys, hidden = self.get_keypad(rgb, grid)
        cells = vision.read_cells(rgb, grid, self.templates, vcfg)
        # 2e passage : si le pavé est partiel/absent, on réessaie en s'aidant
        # des chiffres déjà complets dans la grille (le jeu masque ces touches)
        if not keys or len(keys) < N:
            keys2, hidden2 = self.get_keypad(rgb, grid, cells)
            if keys2 and (not keys or len(keys2) > len(keys)):
                keys, hidden = keys2, hidden2
                # les nouveaux glyphes du pavé améliorent la lecture : on relit
                cells = vision.read_cells(rgb, grid, self.templates, vcfg)
        self._debug_overlay(rgb, grid, cells, keys)
        self.rev += 1
        if not keys:
            self.say("Pavé numérique introuvable → réglez vision.keypad_manual")
            self._note_action("pave_introuvable")
            self.sleep(2500)
            return
        res = self.solve_cells(cells, br, bc) or self.solve_cells(cells, br, bc, only_given=True)
        if not res and vcfg.get("use_agent_ocr_fallback", True):
            n = self.ocr_read(grid, cells)
            self.log("OCR de secours : %d cases corrigées" % n)
            res = self.solve_cells(cells, br, bc)
        if not res:
            self.say("Lecture incohérente, nouvelle capture")
            self.sleep(700)
            return
        givens, sol, cnt = res
        if cnt > 1:
            self.log("Attention : plusieurs solutions possibles (lecture douteuse)", "warn")
        self._debug_overlay(rgb, grid, cells, keys, solution=sol)
        self.rev += 1
        # apprentissage des gabarits sur les chiffres confirmes
        if vcfg.get("learn_templates", True):
            symbols = vision.symbols_for(N, vcfg)
            n_new = 0
            for row in cells:
                for x in row:
                    if x["v"] > 0 and "glyph" in x and givens[x["r"]][x["c"]] == x["v"] == sol[x["r"]][x["c"]]:
                        if self.templates.learn(symbols[x["v"] - 1], x["glyph"], x["aspect"],
                                                vcfg.get("learn_max_per_digit", 14)):
                            n_new += 1
            if n_new:
                self.templates.save()
        moves = [(r, c, sol[r][c]) for r in range(N) for c in range(N)
                 if cells[r][c]["v"] != sol[r][c] or cells[r][c]["color"] == "error"]
        if self.test:
            self.say("(test) %d coups calculés, aucun appui" % len(moves))
            self.sleep(2500)
            return
        # pavé partiel accepté : un chiffre deja complet en grille (ou masque
        # par le jeu) n'a pas de touche, c'est normal. On verifie chaque
        # chiffre manquant dans la grille actuelle, on joue le reste, et on
        # ne bloque JAMAIS (voir _handle_missing_keys).
        if moves:
            moves = self._handle_missing_keys(moves, keys, hidden, cells, N)
            if not moves:
                return
        hcfg = config.section("human")
        cc = solver.Sudoku(givens, br, bc,
                           banned=banned_from_cells(cells)).candidates_count(sol)
        moves = human.order_moves(moves, hcfg.get("order", "human"), br, bc, cc)
        self.say("Résolution : %d cases à remplir (%s)" % (len(moves), self.speed.label()))
        # Clin d'oeil humain (rare, configurable, UNE fois par partie max) :
        # au milieu d'une partie, ouvrir une fois « Astuce intelligente »
        # pour de vrai. La suite (didacticiel a refermer) passe par le
        # circuit normal au cycle suivant.
        target, detour = self._plan_astuce(moves)
        result = self.execute(target, grid, keys)
        if result == "blocked":
            # Une popup recouvrait la grille (vue personnalisee) : deja fermee
            # par le filet d'execute(), on laisse le cycle suivant relire.
            return "blocked"
        if result in ("finished", "complete"):
            self.games += 1
            self.just_finished = True
            self.last_finish = time.time()
            self._mark_progress()
            self._post_game = True  # les ecrans flamme/serie sont optionnels desormais
            self._astuce_used = False  # partie suivante : nouveau clin d'oeil possible
            self._spots_keys_ok = True
            self.speed.save_perf()
            self.keypad_cache.clear()
            if self.batch_div > 1:
                # Remonte d'un cran vers le rythme normal (penalite temporaire).
                self.batch_div = max(1, self.batch_div // 2)
                self.log("Reprise : salves /%d" % self.batch_div)
            self.say("Partie %d terminée" % self.games)
            # Fin de partie VERIFIEE (« complete » = grille pleine, « finished »
            # = ecran de fin) : enchainement DIRECT « Nouvelle partie » puis
            # « Débutant » par hitboxes, au plus 1 s apres la verification —
            # voir _postgame_express (repli texte/OCR si un ecran s'interpose).
            # L'ancienne attente fixe d'ecran de fin est supprimee.
            self._postgame_express(grid)
            mg = config.section("game").get("max_games", 0)
            if mg and self.games >= mg:
                self.control("stop")
            return
        if detour and self.state == "running" and result in ("incomplete", "changed"):
            # Moitie jouee et partie en cours : on ouvre l'astuce pour de vrai,
            # puis relecture fraiche (le reste + la popup seront geres au cycle
            # suivant). Si la grille a disparu (lost), on ne touche a rien.
            self._astuce_used = True
            self._tap_astuce_button()
            return "changed"

    def _plan_astuce(self, moves):
        """Decoupe eventuelle (premier tiers + detour astuce), une fois par
        partie max (sinon le tirage par cycle l'ouvrirait ~1 partie sur 2)."""
        try:
            prob = float(config.section("human").get("astuce_prob", 0.065))
        except Exception:
            prob = 0.065
        if len(moves) >= 4 and not self._astuce_used and random.random() < prob:
            return moves[:len(moves) // 2], True
        return moves, False

    def _tap_astuce_button(self):
        """Ouvre volontairement « Astuce intelligente » (clin d'oeil humain)."""
        mcfg = config.section("menus")
        sp = self.spot_box("game", "astuce")
        if sp is not None:
            self.log("Clin d'oeil humain : ouverture de l'astuce (mémorisée)")
            self.sleep(self.rand_ms([250, 600]))
            self.tap_box(sp)
            self.sleep(self.rand_ms(mcfg.get("dismiss_wait_after_ms", [1100, 1700])))
            return True
        self.log("Clin d'oeil humain : ouverture de l'astuce")
        if self._tap_learned_or_ocr("astuce", "button",
                                   mcfg.get("astuce_texts") or ["Astuce intelligente"],
                                   1500, "astuce (volontaire)"):
            return True
        pos = mcfg.get("astuce_pos")
        return bool(pos) and self._tap_frac_pos(pos, "astuce (position)")

    def _ensure_quick_off(self, rgb, grid):
        """« Placement rapide » actif (bleu) ? On le desactive d'un appui puis
        on reprend un screenshot frais et on continue. Borne a
        quick_toggle_max_taps essais + cooldown (jamais de martelage)."""
        mcfg = config.section("menus")
        if not mcfg.get("quick_toggle_enabled", True):
            return rgb
        try:
            max_taps = max(1, int(mcfg.get("quick_toggle_max_taps", 2)))
        except Exception:
            max_taps = 2
        try:
            cool = float(mcfg.get("quick_toggle_cooldown_s", 8))
        except Exception:
            cool = 8.0
        now = time.time()
        if now - self._quick_tap_t < cool:
            return rgb  # deja tente recemment : on ne martelle pas
        for _ in range(max_taps):
            try:
                st = vision.detect_quick_toggle(rgb, grid)
            except Exception:
                return rgb
            if (not st or not st.get("active")) and self.spots_on():
                # Filet : echantillonne le bleu DANS l'emplacement memorise
                # (la detection par amas peut rater une pilule partielle...).
                try:
                    tb = (self.spots.get("game") or {}).get("toggle")
                    if tb and self.last_rgb is not None:
                        H0, W0 = self.last_rgb.shape[:2]
                        box = spotsmod.box_frac_to_px(tb, W0, H0)
                        if spotsmod.box_blue_frac(rgb, box) > 0.15:
                            cx = (box[0] + box[2]) / 2.0
                            cy = (box[1] + box[3]) / 2.0
                            st = {"active": True, "x": cx, "y": cy, "frac": 1.0}
                            self.log("Placement rapide actif (via emplacement)", "warn")
                except Exception:
                    pass
            if not st or not st.get("active"):
                return rgb
            self.log("Placement rapide actif -> desactivation", "warn")
            self.sleep(self.rand_ms([250, 600]))
            # Point d'appui : l'emplacement memorise si calibre, sinon le
            # centroide detecte (tous deux au centre de la pilule).
            cx, cy = st["x"], st["y"]
            try:
                if self.spots_on():
                    tb = (self.spots.get("game") or {}).get("toggle")
                    if tb and self.last_rgb is not None:
                        H0, W0 = self.last_rgb.shape[:2]
                        cx, cy = float(tb["x"]) * W0, float(tb["y"]) * H0
            except Exception:
                pass
            dx = dy = 26
            self.tap_box([cx - dx, cy - dy, cx + dx, cy + dy])
            self._quick_tap_t = time.time()
            self.sleep(500)
            try:
                rgb = self.shot()
            except AgentOffline:
                return rgb
        try:
            st = vision.detect_quick_toggle(rgb, grid)
        except Exception:
            st = None
        if st and st.get("active"):
            self.log("Placement rapide toujours actif apres %d appuis" % max_taps, "warn")
        return rgb

    def build_taps(self, batch, grid, keys, lv, hcfg):
        fat = 1.0
        if hcfg.get("fatigue", True):
            fat += min(hcfg.get("fatigue_max", 0.18), self.games * hcfg.get("fatigue_per_game", 0.015))
        # le pavé peut être partiel (chiffres terminés masqués) : indexation par valeur
        kmap = {k["v"]: k for k in keys or []}
        taps, t, pts = [], 0, []
        dead = self.dead_top
        try:
            edge_prob = float(hcfg.get("edge_prob", 0.05))
        except Exception:
            edge_prob = 0.05
        try:
            edge_max = float(hcfg.get("edge_max", 0.46))
        except Exception:
            edge_max = 0.46
        for (r, c, v) in batch:
            if v not in kmap:
                raise KeyError("touche %d absente du pavé" % v)
            t += human.think_pause(hcfg, lv, fat)
            x0, y0, x1, y1 = vision.cell_rect(grid, r, c)
            px, py = human.point_in((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0,
                                    hcfg.get("jitter_sigma", 0.13), hcfg.get("jitter_max", 0.3),
                                    edge_prob, edge_max)
            py = max(py, dead + 4)
            d1 = human.tap_duration(lv)
            taps.append([int(px), int(py), int(t), d1])
            pts.append((px, py))
            t += d1 + human.delay(lv["cell_to_key_ms"], fat)
            kd = kmap[v]
            size = max(kd["w"], kd["h"]) * 1.25
            kx, ky = human.point_in(kd["x"], kd["y"], size, size,
                                    hcfg.get("key_jitter_sigma", 0.16), hcfg.get("key_jitter_max", 0.3),
                                    edge_prob, edge_max)
            d2 = human.tap_duration(lv)
            taps.append([int(kx), int(ky), int(t), d2])
            pts.append((kx, ky))
            t += d2 + human.delay(lv["between_ms"], fat)
        return taps, pts

    def _grid_needs_work(self, rgb, grid):
        """Porte rapide de fin de partie : reste-t-il une case vide (ou un
        chiffre rouge a corriger) ? Instantane (encre par case, sans lecture
        complete ni OCR : l'OCR ne voit pas les cases vides). True = analyse
        complete requise, False = grille pleine."""
        try:
            N = grid["n"]
            ink_min = float(config.section("menus").get("spots_slot_ink", 0.01))
        except Exception:
            ink_min = 0.01
        vcfg = config.section("vision")
        try:
            delta = float(vcfg.get("ink_delta", 70))
        except Exception:
            delta = 70.0
        H, W = rgb.shape[:2]
        for r in range(N):
            for c in range(N):
                try:
                    x0, y0, x1, y1 = vision.cell_rect(grid, r, c)
                    cw, ch = x1 - x0, y1 - y0
                    m = 0.10
                    l = max(0, int(x0 + cw * m))
                    t = max(0, int(y0 + ch * m))
                    rr = min(W, int(x1 - cw * m) + 1)
                    bb = min(H, int(y1 - ch * m) + 1)
                    if rr - l < 4 or bb - t < 4:
                        return True  # doute : analyse complete
                    patch = rgb[t:bb, l:rr].astype(np.float32)
                    bg = np.median(patch.reshape(-1, 3), axis=0)
                    dist = np.sqrt(((patch - bg) ** 2).sum(axis=2))
                    ink = dist > delta
                    if float(ink.mean()) < ink_min:
                        return True  # case vide
                    # Encre rouge = chiffre refuse : a corriger, pas finie.
                    px = patch[ink].reshape(-1, 3)
                    if len(px) and vision.classify_color(np.median(px, axis=0), vcfg) == "error":
                        return True
                except Exception:
                    return True  # doute : analyse complete
        return False

    def _keypad_has_digit(self, rgb, grid):
        """Au moins un chiffre visible au pave ? (slots memorises : encre).
        Sans slots calibres : False (on est en post-partie, l'appelant a
        deja verifie _post_game ; la boucle de verification protege)."""
        try:
            N = grid["n"]
            slots = (self.spots.get("game") or {}).get("keys") or []
            if not self.spots_on() or len(slots) != N:
                return False
            H, W = rgb.shape[:2]
            try:
                ink_min = float(config.section("menus").get("spots_slot_ink", 0.01))
            except Exception:
                ink_min = 0.01
            return len(spotsmod.present_slots(rgb, slots, W, H, ink_min=ink_min)) > 0
        except Exception:
            return False

    def _express_blinds(self, W, H):
        """Points a taper en aveugle pendant la rafale : boutons appris +
        croix positionnelles (JAMAIS de bouton a pub : Recuperer/Regarder
        exclus, ils passent par le chemin normal si besoin)."""
        pts = []
        try:
            learned = (self.spots.get("learned") or {})
            for key in sorted(learned):
                # Ni pubs ni chaines de didacticiel en aveugle.
                if key.startswith("bonus:claim") or key.startswith("hint:watch"):
                    continue
                try:
                    box = spotsmod.box_frac_to_px(learned[key], W, H)
                    pts.append((((box[0] + box[2]) / 2.0), ((box[1] + box[3]) / 2.0)))
                except Exception:
                    continue
        except Exception:
            pass
        mcfg = config.section("menus")
        for ck in ("rate_close_pos", "bonus_close_pos", "win_ok_pos"):
            try:
                pos = mcfg.get(ck)
                if pos:
                    pts.append((float(pos[0]) * W, float(pos[1]) * H))
            except Exception:
                continue
        # Dedup grossier (points proches : un seul appui).
        out = []
        for (x, y) in pts:
            if not any(abs(x - ox) < W * 0.05 and abs(y - oy) < H * 0.03 for (ox, oy) in out):
                out.append((x, y))
        return out

    def _wait_seen(self, texts, bounds, what="écran"):
        """Attente ADAPTATIVE : rend la main des que `texts` est visible, au
        lieu d'attendre un delai fixe. Instrument : ui_scan (scan acces.
        seul : ni capture ni OCR, aucun find_any -> les contrats 'bouton
        appris = zero scan' restent vrais). Bornes [min, max] (plancher
        humain / plafond de repli si le texte echappe a l'accessibilite)."""
        mcfg = config.section("menus")
        try:
            lo, hi = bounds
            lo, hi = max(0.0, float(lo)), max(float(lo), float(hi))
        except Exception:
            lo, hi = 350.0, 1200.0
        try:
            poll = max(60.0, float(mcfg.get("appear_poll_ms", 150)))
        except Exception:
            poll = 150.0
        wants = [w for w in (spotsmod.norm(t) for t in (texts or [])) if w]
        if not wants:
            self.sleep(hi)
            return False
        self.say("Attente « %s »…" % what)
        self.sleep(lo)
        steps = int(max(0.0, hi - lo) / poll) + 2
        for _ in range(steps):
            try:
                res = LINK.call("ui_scan", max_items=400, timeout=3)
                for it in res.get("items", []) or []:
                    t = spotsmod.norm(it.get("text"))
                    if t and any(w in t for w in wants):
                        return True
            except AgentOffline:
                return False  # agent muet : plafond deja ecoule ? on continue
            except Exception:
                pass
            self.sleep(poll)
        return False

    def _express_reopen(self, sp_deb, W, H):
        """Repli de _postgame_express : la partie n'a pas demarre (menu
        « Série quotidienne » interpose, popup, appui premature).

        1) Fermer ce qui est interpose : serie (flamme puis retour), popups
           connues (bouton appris ou croix), sinon rafale aveugle pour une
           vue sans texte (boutons connus, JAMAIS de bouton de pub).
        2) Re-cliquer « Nouvelle partie » via le texte (accessibilite/OCR :
           la mise en page a change), puis « Débutant » par son hitbox.
        Renvoie True si la grille est apparue.
        """
        mcfg = config.section("menus")
        bl = mcfg.get("blacklist", [])
        popups = [
            mcfg.get("rate_texts") or ["Aimez-vous", "Evaluez-nous"],
            mcfg.get("win_title_texts") or ["Bien joue"],
            mcfg.get("bonus_title_texts") or ["Bonus"],
            mcfg.get("tutorial_title_texts") or ["Dernier chiffre possible"],
            mcfg.get("tip_title_texts") or ["Astuce"],
            mcfg.get("hint_title_texts") or ["Indices gratuits"],
            mcfg.get("friends_title_texts") or ["avec des amis"],
        ]
        streak = [
            mcfg.get("streak_flame_texts") or ["Touchez la flamme"],
            mcfg.get("streak_retour_texts") or ["Série quotidienne", "retour"],
        ]
        groups = popups + streak
        res = self._call_find_any(groups, [False] * len(groups), bl,
                                  skip_ocr=True, timeout=2.0)
        if res is None:
            return False  # agent muet : le cycle normal reprend la main
        by_group = {}
        hits = res.get("hits")
        if isinstance(hits, list):
            for h in hits:
                try:
                    by_group[int(h.get("group"))] = h
                except Exception:
                    pass
        else:
            try:
                by_group[int(res.get("group", -1))] = res
            except Exception:
                pass
        # 1) Serie quotidienne (son menu s'interpose) : ses boutons ferment
        #    l'ecran — flamme d'abord, puis le bouton « retour » qui suit.
        if mcfg.get("streak_enabled", True) and (7 in by_group or 8 in by_group):
            if 7 in by_group:
                self._tap_streak_flame(by_group[7])
                r2 = self._call_find_any(
                    [["retour", "Retour", "RETOUR", "continuer", "Continuer",
                      "Continue"]], [False], bl, skip_ocr=True, timeout=2.0)
                if r2 is not None and r2.get("found"):
                    self._tap_streak_retour(r2)
            else:
                self._tap_streak_retour(by_group[8])
        # 2) Popups connues : bouton appris ou croix (fermeture immediate).
        if 0 in by_group:
            if not self._tap_learned_or_ocr(
                    "rate", "dismiss", mcfg.get("rate_dismiss_texts") or [],
                    mcfg.get("rate_wait_after_ms", [1100, 1700]),
                    "avis (bouton)"):
                self.sleep(self.rand_ms([250, 600]))
                self.tap_rate_close(by_group[0].get("b"))
                self.sleep(self.rand_ms(mcfg.get("rate_wait_after_ms", [1100, 1700])))
        if 1 in by_group:
            self.handle_win()
        if 2 in by_group:
            self.handle_bonus()
        if 3 in by_group:
            self.handle_tutorial()
        if 4 in by_group:
            self.handle_tip()
        if 5 in by_group:
            self.handle_hint()
        if 6 in by_group:
            self.handle_friends()
        if not by_group:
            # Vue couvrante SANS texte (personnalisee / pub) : rafale aveugle
            # (boutons appris + croix, jamais les boutons de pub).
            for (x, y) in self._express_blinds(W, H):
                self.tap_box([x - 24, y - 24, x + 24, y + 24])
                self.sleep(self.rand_ms(mcfg.get("express_gap_ms", [250, 400])))
        # 3) Re-cliquer « Nouvelle partie » via le texte (mise en page
        #    changee)…
        new_texts = mcfg.get("loss_new_game_texts") or ["Nouvelle partie", "New game"]
        if not self._tap_first_text(new_texts, 0, "nouvelle partie (repli)"):
            return False
        self._home_done = True
        # …puis « Débutant » par son hitbox (emplacement predetermine).
        self.countdown_sleep(self.rand_ms(mcfg.get("loss_wait_after_ms", [850, 1050])))
        self.tap_box(spotsmod.box_frac_to_px(sp_deb, W, H))
        return self.wait_for_grid(max_ms=2500)

    def _postgame_express(self, grid):
        """Fin de partie (spots requis, sinon False -> chemin normal) :
        verification puis « Nouvelle partie » puis « Débutant » DIRECT par
        hitboxes — au plus 1 s apres la verification (une capture de porte +
        l'appui : ni scan ni attente d'animations).

        Verification (rattrape un placement non pris en compte) : tant que le
        pave montre un chiffre, la partie n'est pas finie -> le cycle normal
        la termine, SAUF ecran « Perdu » (3 erreurs -> handle_loss). Si la
        partie ne demarre pas (menu « Série quotidienne », popup, appui
        premature), _express_reopen() ferme puis re-clique via le texte.
        Borne : menus.express_rounds tours, arret si l'ecran ne bouge plus.
        """
        if not self._post_game or not self.spots_on():
            return False
        mcfg = config.section("menus")
        N = grid["n"]
        slots = (self.spots.get("game") or {}).get("keys") or []
        sp_new = (self.spots.get("fin_partie") or {}).get("nouvelle_partie")
        sp_deb = (self.spots.get("niveaux") or {}).get("debutant")
        if len(slots) != N or not sp_new or not sp_deb:
            return False
        try:
            rounds = max(1, int(mcfg.get("express_rounds", 3)))
        except Exception:
            rounds = 3
        prev = rgb = None
        for _ in range(rounds):
            self.checkpoint()
            if rgb is None:
                try:
                    rgb = self.shot()
                except AgentOffline:
                    return False
            if self._keypad_has_digit(rgb, grid):
                # Le pave montre encore des chiffres : la partie n'est pas
                # finie (placement non pris en compte ?) -> le cycle normal la
                # termine... SAUF ecran « Perdu » (3 erreurs) : son bouton
                # « Nouvelle partie » n'est PAS a l'emplacement habituel, il
                # passe par le texte (scan acces., sans OCR) + handle_loss().
                try:
                    if self.scan_blocking(use_ocr=False) == "loss":
                        self.handle_loss()
                        return True
                except AgentOffline:
                    pass
                return False
            H, W = rgb.shape[:2]
            # 1) « Nouvelle partie » DIRECT sur son hitbox : aucune attente ni
            #    scan entre la verification et cet appui (budget 1 s).
            self.tap_box(spotsmod.box_frac_to_px(sp_new, W, H))
            self._home_done = True
            # 2) « Débutant » sur son hitbox une fois le menu affiche.
            self.countdown_sleep(self.rand_ms(mcfg.get("loss_wait_after_ms", [850, 1050])))
            self.tap_box(spotsmod.box_frac_to_px(sp_deb, W, H))
            # 3) La partie demarre ? Sinon repli (serie quotidienne, popup,
            #    appui premature) : fermer, puis re-cliquer via le texte.
            if self.wait_for_grid(max_ms=2500):
                return True
            if self._express_reopen(sp_deb, W, H):
                return True
            # L'ecran n'a pas bouge d'un poil ? Inutile d'insister : normal.
            try:
                if prev is not None and vision.frame_diff(prev, rgb) < 1.0:
                    return False
            except Exception:
                pass
            prev, rgb = rgb, None
        return False

    def execute(self, moves, grid, keys):
        vcfg = config.section("vision")
        hcfg = config.section("human")
        kmap = {k["v"]: k for k in keys or []}
        # si le pavé a changé en cours de route (chiffre masqué entre-temps), on
        # ne joue que les coups possibles puis on laisse la boucle relire la grille
        playable = [m for m in moves if m[2] in kmap]
        skipped = [m for m in moves if m[2] not in kmap]
        if skipped:
            self.log("Pavé partiel en cours de partie : %d coup(s) reporté(s) (touches %s)" %
                     (len(skipped), sorted(set(m[2] for m in skipped))), "warn")
        remaining = list(playable) + list(skipped)
        fails = collections.Counter()
        first_batch = True
        while remaining:
            self.checkpoint()
            lv = self.speed.level()
            # Apres un Perdu, les salves sont temporairement divisees (/2, /4...)
            # jusqu'a 1 coup par salve (menus.loss_halve_batch). Voir halve_batch().
            raw_batch = max(1, int(lv.get("batch", 1)))
            bsize = halve_batch(raw_batch, self.batch_div)
            if self.speed.one_go():
                # « tout d'une traite » : la salve couvre tous les coups jouables
                # restants ; la grille n'est re-verifiee qu'une fois la salve jouee.
                bsize = max(bsize, sum(1 for m in remaining if m[2] in kmap))
            elif bsize != raw_batch:
                self.msg = "%d restantes · %s (/%d)" % (len(remaining), self.speed.label(), self.batch_div)
            batch = remaining[:bsize]
            try:
                taps, pts = self.build_taps(batch, grid, keys, lv, hcfg)
            except KeyError as e:
                # une touche a disparu (chiffre terminé masqué) : on réessaie sans elle
                self.log("Touche disparue (%s) : relecture du pavé" % e, "warn")
                self.keypad_cache.clear()
                return "changed"
            res = self.tap_batch(taps)
            done_moves = int(res.get("done", 0)) // 2
            executed = batch[:done_moves]
            self.moves += done_moves
            if done_moves:
                self._mark_progress()
            remaining = batch[done_moves:] + remaining[len(batch):]
            if not remaining:
                # dernier coup joue : le jeu passe a l'ecran de fin. On ne re-appuie JAMAIS ici ;
                # si la grille est encore incomplete, la boucle principale la relira et la reprendra.
                self.sleep(vcfg.get("settle_ms", 220) + vcfg.get("finish_extra_ms", 300))
                rgb = self.shot()
                g2 = self.spots_grid(rgb, grid)
                if g2 is None:
                    g2 = vision.detect_grid(rgb, self.dead_top)
                if g2:
                    rgb = self._ensure_quick_off(rgb, g2)
                    # Porte rapide : grille pleine (aucune case vide ni rouge) ?
                    # Si oui on saute la relecture complete (ni OCR ni solveur).
                    # Sinon analyse entiere comme avant.
                    if not self._grid_needs_work(rgb, g2):
                        self.speed.report(int(res.get("lag_max", 0)), 0, self.shot_ms, len(taps))
                        return "complete"
                    cells = vision.read_cells(rgb, g2, self.templates, vcfg)
                    self._debug_overlay(rgb, g2, cells, keys)
                    self.rev += 1
                    if not all(x["v"] > 0 for row in cells for x in row):
                        return "incomplete"
                self.speed.report(int(res.get("lag_max", 0)), 0, self.shot_ms, len(taps))
                return "finished"
            if res.get("aborted"):
                self.state = "paused"
                self.say("Pause (depuis la barre)")
                self.checkpoint()
            self.sleep(vcfg.get("settle_ms", 220))
            rgb = self.shot(stable=bsize > 1)
            g2 = self.spots_grid(rgb, grid)
            if g2 is None:
                g2 = vision.detect_grid(rgb, self.dead_top)
            if g2:
                # L'option a pu s'activer pendant la salve (appui proche) :
                # on la recoupe ici aussi, puis on continue sur du frais.
                rgb = self._ensure_quick_off(rgb, g2)
            if not g2:
                if not remaining:
                    return "finished"
                self.log("Grille disparue en cours de partie", "warn")
                return "lost"
            if g2["n"] != grid["n"] or abs(g2["x0"] - grid["x0"]) > grid["cw"] * 0.3:
                return "changed"
            cells = vision.read_cells(rgb, g2, self.templates, vcfg)
            self._debug_overlay(rgb, g2, cells, keys, taps=pts)
            self.rev += 1
            misses = 0
            for (r, c, v) in executed:
                x = cells[r][c]
                if x["v"] != v or x["color"] == "error":
                    misses += 1
                    fails[(r, c)] += 1
                    if fails[(r, c)] <= 3:
                        remaining.insert(0, (r, c, v))
                    else:
                        self.log("Case %d,%d refuse %d" % (r + 1, c + 1, v), "warn")
            if first_batch and executed and misses == len(executed) and len(executed) >= 2:
                # Rien ne s'est inscrit : une popup (vue personnalisee, invisible
                # a l'accessibilite) recouvre probablement la grille. Balayage
                # OCR de secours, fermeture si trouve, sinon on continue.
                first_batch = False
                self.log("Salve sans effet (%d/%d) → popup par-dessus ? (scan OCR)" %
                         (misses, len(executed)), "warn")
                try:
                    name = self.scan_blocking(use_ocr=True)
                    if name and self.handle_blocking(name):
                        return "blocked"
                except AgentOffline:
                    pass
                except Stopped:
                    raise
                except Exception as e:
                    self.log("Filet popup : %s" % e, "warn")
            else:
                first_batch = False
            changed = self.speed.report(int(res.get("lag_max", 0)), misses, self.shot_ms, len(taps))
            if changed:
                self.log("Auto : vitesse %s → %s" % ("↑" if changed == "up" else "↓", self.speed.label()))
            self.msg = "%d restantes · %s" % (len(remaining), self.speed.label())
            if not remaining:
                # derniere verification : grille complete ?
                if all(x["v"] > 0 for row in cells for x in row):
                    return "finished"
        return "finished"

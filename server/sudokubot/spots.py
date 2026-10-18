"""Emplacements memorises : hitboxes calibrees sur CE telephone.

L'utilisateur depose des captures de reference dans `emplacements/` :
  game.jpg        partie fraiche (grille vide, pave COMPLET, placement rapide OFF)
  fin_partie.jpg  ecran de victoire (bouton « Nouvelle partie »)
  niveaux.jpg     liste des difficultes (ligne « Debutant »)

Au demarrage, `calibrate()` en extrait les coordonnees (fractions d'ecran,
donc robustes), stockees dans `data/spots.json` (fichier local, hors git).
Ensuite le bot tape directement dans les hitboxes au lieu de chercher chaque
bouton par OCR. Sans images, ou en mode --auto, tout passe par l'OCR
(ancien code, conserve a l'identique).
"""
import base64
import io
import json
import os
import threading
import time
import unicodedata

import numpy as np
from PIL import Image

from . import config, vision

ROOT = config.ROOT
REFS_DIR = os.path.join(ROOT, "emplacements")
OUT_PATH = os.path.join(config.DATA, "spots.json")

REF_DEFS = ("game", "fin_partie", "niveaux")
EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# Textes cherches par OCR dans les references (minuscules sans accents).
OCR_NEED = {
    "fin_partie": ("nouvelle_partie", ["nouvelle partie", "new game"]),
    "niveaux": ("debutant", ["debutant", "beginner"]),
    "game": ("astuce", ["astuce intelligente", "smart tip", "smart hint"]),
}


def norm(s):
    try:
        s = unicodedata.normalize("NFD", str(s or ""))
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    except Exception:
        pass
    return " ".join(str(s).lower().split())


def ensure_dirs():
    try:
        os.makedirs(REFS_DIR, exist_ok=True)
    except Exception:
        pass


def find_refs(refs_dir=REFS_DIR):
    """{nom: chemin} des references presentes (toutes extensions)."""
    out = {}
    try:
        files = os.listdir(refs_dir)
    except OSError:
        return out
    for name in REF_DEFS:
        for f in files:
            base, ext = os.path.splitext(f)
            if base == name and ext.lower() in EXTS:
                out[name] = os.path.join(refs_dir, f)
                break
    return out


def load_image(path):
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.uint8)
    return rgb, img.width, img.height


def load(path=OUT_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save(spots, path=OUT_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(spots, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return spots


def _num(x):
    try:
        v = float(x)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return min(1.0, max(0.0, v))
    except (TypeError, ValueError):
        return None


def sanitize(raw):
    """Nettoie des hitboxes venues du web : que des nombres dans [0,1],
    cles connues uniquement (jamais de crash a la relecture)."""
    clean = {}
    if not isinstance(raw, dict):
        return clean
    for ref in REF_DEFS:
        section = raw.get(ref)
        if not isinstance(section, dict):
            continue
        out = {}
        grid = section.get("grid")
        if isinstance(grid, dict):
            g = {}
            for k in ("x0", "y0", "x1", "y1"):
                v = _num(grid.get(k))
                if v is not None:
                    g[k] = v
            for k in ("n", "br", "bc"):
                try:
                    g[k] = int(grid.get(k))
                except (TypeError, ValueError):
                    pass
            if len([k for k in ("x0", "y0", "x1", "y1") if k in g]) == 4:
                out["grid"] = g
        for key in ("nouvelle_partie", "debutant", "toggle", "astuce"):
            b = section.get(key)
            if isinstance(b, dict):
                nb = {}
                for k in ("x", "y", "w", "h"):
                    v = _num(b.get(k))
                    if v is not None:
                        nb[k] = v
                if len(nb) == 4:
                    out[key] = nb
                elif key == "toggle" and "x" in nb and "y" in nb:
                    # Vieux format (point seul) : dimensions par defaut
                    # de la pilule, plutot que de perdre l'emplacement.
                    nb.setdefault("w", 0.063)
                    nb.setdefault("h", 0.018)
                    out[key] = nb
        keys = section.get("keys")
        if isinstance(keys, list):
            ks = []
            for s in keys[:16]:
                if not isinstance(s, dict):
                    continue
                try:
                    v = int(s.get("v"))
                except (TypeError, ValueError):
                    continue
                nb = {"v": v}
                ok = True
                for k in ("x", "y", "w", "h"):
                    vv = _num(s.get(k))
                    if vv is None:
                        ok = False
                        break
                    nb[k] = vv
                if ok:
                    ks.append(nb)
            if ks:
                out["keys"] = ks
        if out:
            clean[ref] = out
    meta = raw.get("_meta")
    if isinstance(meta, dict):
        clean["_meta"] = {k: v for k, v in meta.items() if isinstance(v, (str, int, float))}
    learned = raw.get("learned")
    if isinstance(learned, dict):
        keep = {}
        for k, b in learned.items():
            if not isinstance(k, str) or ":" not in k or not isinstance(b, dict):
                continue
            nb = {}
            for f in ("x", "y", "w", "h"):
                v = _num(b.get(f))
                if v is not None:
                    nb[f] = v
            if len(nb) == 4:
                keep[k] = nb
        if keep:
            clean["learned"] = keep
    return clean


def is_stale(refs=None, path=OUT_PATH):
    """True si les references sont plus neuves que spots.json (ou absent),
    ou si spots.json est d'un vieux format (toggle sans dimensions)."""
    refs = find_refs() if refs is None else refs
    if not refs:
        return False
    if needs_upgrade(load(path)):
        return True
    try:
        out_mtime = os.path.getmtime(path)
    except OSError:
        return True
    for p in refs.values():
        try:
            if os.path.getmtime(p) > out_mtime:
                return True
        except OSError:
            pass
    return False


def needs_calibration(spots, refs=None):
    """Hitboxes absentes ou perimees pour les references presentes ?

    Complement indispensable de is_stale() : une simple sauvegarde de bouton
    appris rafraichit le mtime de spots.json et masque la staleness ; ici on
    verifie le CONTENU (cles attendues) et les mtime enregistres en _meta au
    moment de la calibration. Sans cette verifie, un spots.json ne contenant
    que des boutons appris ne serait JAMAIS recalibre (les boutons partent
    alors a la recherche OCR a chaque fois = lent).
    """
    refs = find_refs() if refs is None else refs
    if not refs:
        return False
    want = {"game": ("grid",), "fin_partie": ("nouvelle_partie",),
            "niveaux": ("debutant",)}
    spots = spots or {}
    meta = spots.get("_meta") or {}
    for ref in refs:
        keys = want.get(ref)
        if not keys:
            continue
        s = spots.get(ref) or {}
        if not all(s.get(k) for k in keys):
            return True
        m = meta.get(ref) or {}
        try:
            if float(m.get("mtime", -1)) < os.path.getmtime(refs[ref]):
                return True
        except Exception:
            return True
    return False


def needs_upgrade(spots):
    """Vieux spots.json (toggle enregistre sans dimensions) ?"""
    try:
        tg = (spots.get("game") or {}).get("toggle")
        if tg and not all(k in tg for k in ("x", "y", "w", "h")):
            return True
    except Exception:
        pass
    return False


def box_frac_to_px(box, W, H):
    """{'x','y','w','h'} fractions (centre+taille) -> [l,t,r,b] pixels."""
    cx, cy = float(box["x"]) * W, float(box["y"]) * H
    w, h = float(box["w"]) * W, float(box["h"]) * H
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def grid_frac_to_px(gfrac, W, H):
    """Grille fractions {x0,y0,x1,y1,n,br,bc} -> dict geometrie pixels."""
    import numpy as _np
    n = int(gfrac["n"])
    x0, y0 = float(gfrac["x0"]) * W, float(gfrac["y0"]) * H
    x1, y1 = float(gfrac["x1"]) * W, float(gfrac["y1"]) * H
    xs = _np.linspace(x0, x1, n + 1).tolist()
    ys = _np.linspace(y0, y1, n + 1).tolist()
    return {"n": n, "br": int(gfrac.get("br", 3)), "bc": int(gfrac.get("bc", 3)),
            "x0": x0, "y0": y0, "x1": x1, "y1": y1, "xs": xs, "ys": ys,
            "cw": (x1 - x0) / n, "ch": (y1 - y0) / n, "w": W, "h": H,
            "score": 1.0, "box_src": "spots"}


def slot_ink_frac(rgb, box_px, delta=60.0):
    """Part d'encre dans une case (0 = vide)."""
    try:
        H, W = rgb.shape[:2]
        l = max(0, int(box_px[0]))
        t = max(0, int(box_px[1]))
        r = min(W, int(box_px[2]) + 1)
        b = min(H, int(box_px[3]) + 1)
        if r - l < 4 or b - t < 4:
            return 0.0
        patch = rgb[t:b, l:r].astype(np.float32)
        bg = np.median(patch.reshape(-1, 3), axis=0)
        dist = np.sqrt(((patch - bg) ** 2).sum(axis=2))
        return float((dist > delta).mean())
    except Exception:
        return 0.0


def box_blue_frac(rgb, box_px):
    """Part de pixels bleus (B dominant) dans une boite pixels : filet de
    detection de l'interrupteur via son emplacement memorise."""
    try:
        H, W = rgb.shape[:2]
        l = max(0, int(box_px[0]))
        t = max(0, int(box_px[1]))
        r = min(W, int(box_px[2]) + 1)
        b = min(H, int(box_px[3]) + 1)
        if r - l < 6 or b - t < 6:
            return 0.0
        sub = rgb[t:b, l:r].astype(np.int16)
        B, R, G = sub[..., 2], sub[..., 0], sub[..., 1]
        return float(((B > 140) & (B - R > 50) & (B - G > 40)).mean())
    except Exception:
        return 0.0


def present_slots(rgb, slots_frac, W, H, ink_min=0.01):
    """Slots occupes (touches visibles) : [{'v','x','y','w','h'}] pixels."""
    out = []
    for s in slots_frac or []:
        try:
            box = box_frac_to_px(s, W, H)
            if slot_ink_frac(rgb, box) < ink_min:
                continue  # touche masquee (chiffre termine) : absente, normal
            cx = (box[0] + box[2]) / 2.0
            cy = (box[1] + box[3]) / 2.0
            out.append({"v": int(s["v"]), "x": cx, "y": cy,
                        "w": box[2] - box[0], "h": box[3] - box[1],
                        "src": "spots"})
        except Exception:
            continue
    return out


def lattice_present(rgb, grid, delta=12.0, need=0.15):
    """La grille `grid` (pixels) est-elle visible ? (verification bon marche
    des hitboxes memorisees : energie sombre le long des lignes attendues).
    """
    try:
        from .vision import gray
        g = gray(rgb)
        x0, y0 = int(grid["x0"]) + 1, int(grid["y0"]) + 1
        x1, y1 = int(grid["x1"]) - 1, int(grid["y1"]) - 1
        if x1 - x0 < 30 or y1 - y0 < 30:
            return False
        scores = []
        for x in grid["xs"][1:-1]:
            xi = int(round(x))
            if xi <= x0 + 2 or xi >= x1 - 2:
                continue
            strip = g[y0:y1, max(x0, xi - 2):min(x1, xi + 3)]
            bg = float(np.median(g[y0:y1, max(x0, xi - 10):min(x1, xi + 11)]))
            scores.append(float((strip < bg - delta).mean()))
        for y in grid["ys"][1:-1]:
            yi = int(round(y))
            if yi <= y0 + 2 or yi >= y1 - 2:
                continue
            strip = g[max(y0, yi - 2):min(y1, yi + 3), x0:x1]
            bg = float(np.median(g[max(y0, yi - 10):min(y1, yi + 11), x0:x1]))
            scores.append(float((strip < bg - delta).mean()))
        if not scores:
            return False
        return float(np.mean(scores)) >= need
    except Exception:
        return False


# ------------------------------------------------------------ calibration
def locate_toggle(rgb, grid):
    """Position ET taille de la pilule (grise OU bleue) : plus gros amas
    non-blanc dans la zone sous la grille a droite. Renvoie
    (x, y, w, h) pixels (centre+taille) ou None."""
    try:
        H, W = rgb.shape[:2]
        x0 = max(0, int(grid["x1"] - W * 0.12))
        x1 = W - 1
        # Juste SOUS la bordure basse de la grille (sinon elle fait un
        # faux amas sur toute la largeur).
        ya = max(0, int(grid["y1"] + grid["ch"] * 0.08))
        yb = min(H - 1, int(grid["y1"] + grid["ch"] * 0.9))
        if x1 - x0 < 20 or yb - ya < 10:
            return None
        sub = rgb[ya:yb, x0:x1].astype(np.int16)
        white = np.median(sub.reshape(-1, 3), axis=0)
        mask = (np.sqrt(((sub - white) ** 2).sum(axis=2)) > 45)
        colc = mask.sum(axis=0)
        best, run = None, None
        for i, v in enumerate(colc):
            if v >= 3:
                run = (i, i) if run is None else (run[0], i)
            else:
                if run is not None and 15 <= run[1] - run[0] <= 130:
                    wsum = float(colc[run[0]:run[1] + 1].sum())
                    if best is None or wsum > best[0]:
                        best = (wsum, run)
                run = None
        if run is not None and 15 <= run[1] - run[0] <= 130:
            wsum = float(colc[run[0]:run[1] + 1].sum())
            if best is None or wsum > best[0]:
                best = (wsum, run)
        if best is None:
            return None
        a, b = best[1]
        band = mask[:, a:b + 1]
        rows = np.flatnonzero(band.sum(axis=1) >= 2)
        if len(rows) == 0:
            return None
        cx = x0 + (a + b) / 2.0
        cy = ya + (int(rows[0]) + int(rows[-1])) / 2.0
        w = float(b - a + 1)
        h = float(int(rows[-1]) - int(rows[0]) + 1)
        if cx >= W - 2 or cy >= H - 2:
            return None
        return (float(cx), float(cy), w, h)
    except Exception:
        return None


def _ocr_ref(call, path, log):
    """OCR d'une image de reference via l'agent (nouveau primitif ocr_image).
    Renvoie (items, W, H) ou (None, 0, 0)."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("ascii")
        res = call("ocr_image", image=b64, timeout=30)
        if not res or not res.get("ok"):
            log("OCR reference %s : %s" % (os.path.basename(path),
                                           (res or {}).get("error", "echec")))
            return None, 0, 0
        return res.get("items", []) or [], int(res.get("w", 0)), int(res.get("h", 0))
    except Exception as e:
        log("OCR reference %s : %s" % (os.path.basename(path), e))
        return None, 0, 0


def _pick_hit(items, wants, W, H):
    """Meilleur hit OCR (fractions). Par niveau de confiance :
    1. la phrase entiere dans UN bloc (prefere le plus large : boutons) ;
    2. la phrase a cheval sur des blocs adjacents de la MEME ligne
       (l'OCR coupe souvent « Astuce » / « intelligente ») : boite union ;
    3. un mot distinctif seul (>= 5 lettres : « astuce », « nouvelle »...).
    """
    normed = []
    for it in items or []:
        t = norm(it.get("text"))
        b = it.get("b")
        if not t or not b:
            continue
        try:
            l, tt, r, bb = (float(v) for v in b)
        except Exception:
            continue
        normed.append({"t": t, "l": l, "tt": tt, "r": r, "bb": bb,
                       "area": max(0.0, r - l) * max(0.0, bb - tt)})

    def frac(b):
        return {"x": (b["l"] + b["r"]) / 2.0 / W, "y": (b["tt"] + b["bb"]) / 2.0 / H,
                "w": (b["r"] - b["l"]) / W, "h": (b["bb"] - b["tt"]) / H}

    for w in wants:
        cands = [n for n in normed if w in n["t"]]
        if cands:
            best = max(cands, key=lambda n: n["area"])
            return frac(best)
    # Blocs adjacents sur la meme ligne (chevauchement vertical > 50%).
    lines = []
    for n in sorted(normed, key=lambda n: n["l"]):
        placed = False
        for line in lines:
            ref = line[-1]
            overlap = min(n["bb"], ref["bb"]) - max(n["tt"], ref["tt"])
            hmin = min(n["bb"] - n["tt"], ref["bb"] - ref["tt"])
            if hmin > 0 and overlap > 0.5 * hmin:
                line.append(n)
                placed = True
                break
        if not placed:
            lines.append([n])
    for w in wants:
        words = w.split()
        best = None
        for line in lines:
            for i in range(len(line)):
                for j in range(i + 1, min(len(line), i + len(words) + 1) + 1):
                    joined = " ".join(n["t"] for n in line[i:j])
                    if w in joined:
                        l = min(n["l"] for n in line[i:j])
                        tt = min(n["tt"] for n in line[i:j])
                        r = max(n["r"] for n in line[i:j])
                        bb = max(n["bb"] for n in line[i:j])
                        area = (r - l) * (bb - tt)
                        # Le plus resserre d'abord (2 blocs plutot que 3),
                        # puis le plus large (boutons).
                        key = (j - i, -area)
                        if best is None or key < best[0]:
                            best = (key, {"l": l, "tt": tt, "r": r, "bb": bb})
        if best:
            return frac(best[1])
    # Mot distinctif seul.
    for w in wants:
        for word in w.split():
            if len(word) < 5:
                continue
            cands = [n for n in normed if word in n["t"]]
            if cands:
                return frac(max(cands, key=lambda n: n["area"]))
    return None


def calibrate_game(path, log):
    """Analyse LOCALE de game.jpg : grille, pave complet, toggle, (astuce via OCR)."""
    rgb, W, H = load_image(path)
    out = {}
    grid = vision.detect_grid(rgb, 0)
    if not grid:
        log("game.jpg : grille introuvable")
        return out, (rgb, W, H)
    n = grid["n"]
    out["grid"] = {"x0": grid["x0"] / W, "y0": grid["y0"] / H,
                   "x1": grid["x1"] / W, "y1": grid["y1"] / H,
                   "n": n, "br": grid["br"], "bc": grid["bc"]}
    log("game.jpg : grille %dx%d" % (n, n))
    tpls = vision.Templates()
    tpls.clear_src("learned")
    keys = vision.detect_keypad(rgb, grid, tpls, config.section("vision")) or []
    slots = [{"v": k["v"], "x": k["x"] / W, "y": k["y"] / H,
              "w": k["w"] / W, "h": k["h"] / H} for k in keys]
    out["keys"] = slots
    log("game.jpg : %d touches memorisees" % len(slots))
    pos = locate_toggle(rgb, grid)
    if pos:
        out["toggle"] = {"x": pos[0] / W, "y": pos[1] / H,
                         "w": pos[2] / W, "h": pos[3] / H}
        st = vision.detect_quick_toggle(rgb, grid)
        log("game.jpg : interrupteur memorise (actif=%s dans la reference)" %
            (st.get("active") if st else "?"))
    return out, (rgb, W, H)


def calibrate(call=None, log=print, refs=None, out_path=OUT_PATH):
    """Calibre les references -> spots.json. `call` = LINK.call (OCR des
    boutons) ; sans agent, seules les parties locales (grille/pave/toggle)
    sont calibrees, le reste garde les anciennes valeurs."""
    refs = find_refs() if refs is None else refs
    spots = load(out_path)
    if not refs:
        log("emplacements/ : aucune reference (game/fin_partie/niveaux.*)")
        return spots
    meta = spots.get("_meta", {})
    if "game" in refs:
        try:
            gout, (_rgb, _W, _H) = calibrate_game(refs["game"], log)
            spots["game"] = gout
            meta["game"] = {"file": os.path.basename(refs["game"]),
                            "mtime": os.path.getmtime(refs["game"])}
        except Exception as e:
            log("game.jpg : %s" % e)
    if call is not None:
        for ref_name in ("fin_partie", "niveaux", "game"):
            if ref_name not in refs or ref_name not in OCR_NEED:
                continue
            key, wants = OCR_NEED[ref_name]
            items, W, H = _ocr_ref(call, refs[ref_name], log)
            if not items or not W:
                continue
            hit = _pick_hit(items, wants, W, H)
            if hit:
                spots.setdefault(ref_name, {})[key] = hit
                meta[ref_name] = {"file": os.path.basename(refs[ref_name]),
                                  "mtime": os.path.getmtime(refs[ref_name])}
                log("%s : '%s' memorise" % (os.path.basename(refs[ref_name]), key))
            else:
                log("%s : texte '%s' introuvable (OCR: %d blocs)" %
                    (os.path.basename(refs[ref_name]), wants[0], len(items)))
    else:
        log("Agent hors ligne : boutons non calibres (OCR agent requis)")
    import time as _time
    meta["calibrated_at"] = _time.strftime("%Y-%m-%d %H:%M:%S")
    spots["_meta"] = meta
    save(spots, out_path)
    return spots


def ensure_current(call, log, timeout=600, on_done=None):
    """Au demarrage : calibre le local + les boutons si l'agent est la, sinon
    en tache de fond (des que l'agent se connecte, max `timeout` s : l'agent
    AutoJs6 est souvent lance apres le serveur). `on_done` recoit les spots
    finaux (ex. pour rafraichir le controleur)."""
    from .bus import LINK
    ensure_dirs()
    refs = find_refs()
    if not refs:
        return load()
    have = load()
    if not is_stale(refs) and not needs_calibration(have, refs):
        return have
    if LINK.online():
        out = calibrate(call, log, refs)
        if on_done:
            on_done(out)
        return out
    # Local tout de suite (sans agent), OCR en fond.
    partial = calibrate(None, log, refs)

    def _bg():
        end = time.time() + timeout
        while time.time() < end:
            if LINK.online():
                try:
                    out = calibrate(call, log, find_refs())
                    if on_done:
                        on_done(out)
                except Exception as e:
                    log("Calibration fond : %s" % e)
                return
            time.sleep(2)

    threading.Thread(target=_bg, daemon=True).start()
    return partial

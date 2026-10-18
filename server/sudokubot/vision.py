"""Vision generique pour sudoku (numpy + Pillow uniquement, compatible Termux).

- detect_grid()   : trouve la grille quelle que soit sa taille (4x4, 6x6, 9x9, 12x12...) et la
                    taille d'ecran, par detection de "cretes" (lignes fines) + reseau regulier.
- detect_boxes()  : deduit la forme des blocs (3x3, 2x3, ...) via l'epaisseur/contraste des lignes.
- detect_keypad() : trouve la rangee de chiffres en bas (ordre gauche->droite = 1..N).
- read_cells()    : lit chaque case par correspondance de gabarits (appris, pave, polices systeme).
"""
import io
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import config


# ---------------------------------------------------------------- utilitaires
def to_rgb_array(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def gray(rgb):
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def frame_diff(a, b):
    """Difference moyenne entre deux captures (sous-echantillonnees)."""
    if a is None or b is None or a.shape != b.shape:
        return 999.0
    ga = gray(a[::8, ::8])
    gb = gray(b[::8, ::8])
    return float(np.mean(np.abs(ga - gb)))


def is_blank(rgb):
    small = rgb[::16, ::16]
    return float(small.max()) < 12.0


def _groups(idx, gap=2):
    if idx.size == 0:
        return []
    out = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def _candidates(cov, thr, offset=0):
    idx = np.flatnonzero(cov >= thr)
    cands = []
    for a, b in _groups(idx, 2):
        seg = cov[a:b + 1]
        w = seg.sum()
        center = a + float((seg * np.arange(len(seg))).sum() / max(w, 1e-6))
        cands.append({"pos": center + offset, "a": a + offset, "b": b + offset,
                      "thick": b - a + 1, "cov": float(seg.mean())})
    if len(cands) > 70:
        cands = sorted(cands, key=lambda c: -c["cov"])[:70]
    return cands


def _ridges(g, k, delta, axis):
    """Masque des pixels plus sombres (ou plus clairs) que leurs voisins a distance k (lignes fines)."""
    if axis == 0:   # lignes horizontales : comparer haut/bas
        up = np.empty_like(g); up[k:] = g[:-k]; up[:k] = g[:k]
        dn = np.empty_like(g); dn[:-k] = g[k:]; dn[-k:] = g[-k:]
    else:           # lignes verticales : comparer gauche/droite
        up = np.empty_like(g); up[:, k:] = g[:, :-k]; up[:, :k] = g[:, :k]
        dn = np.empty_like(g); dn[:, :-k] = g[:, k:]; dn[:, -k:] = g[:, -k:]
    d1 = up - g
    d2 = dn - g
    return ((d1 > delta) & (d2 > delta)) | ((d1 < -delta) & (d2 < -delta))


def find_lattice(cands, sizes, tol, min_score, min_span):
    if len(cands) < 3:
        return None
    pos = np.array(sorted(c["pos"] for c in cands), dtype=np.float64)
    m = len(pos)
    best = None
    for i in range(m):
        for j in range(i + 1, m):
            L = pos[j] - pos[i]
            if L < min_span:
                continue
            for N in sizes:
                step = L / N
                if step < 8:
                    continue
                exp = pos[i] + step * np.arange(N + 1)
                k = np.searchsorted(pos, exp)
                k0 = np.clip(k - 1, 0, m - 1)
                k1 = np.clip(k, 0, m - 1)
                d0 = np.abs(pos[k0] - exp)
                d1 = np.abs(pos[k1] - exp)
                d = np.minimum(d0, d1)
                ok = d <= tol * step
                matched = int(ok.sum())
                if matched / float(N + 1) < min_score:
                    continue
                key = (matched, L)
                if best is None or key > best[0]:
                    near = np.where(d0 < d1, pos[k0], pos[k1])
                    lines = np.where(ok, near, exp)
                    best = (key, N, lines)
    if not best:
        return None
    return {"n": best[1], "lines": [float(x) for x in best[2]], "matched": best[0][0]}


# ---------------------------------------------------------------- grille
def detect_grid(rgb, dead_top=0, cfg=None):
    cfg = cfg or config.section("vision")
    H, W = rgb.shape[:2]
    g = gray(rgb)
    top = int(max(dead_top, cfg.get("ignore_top_frac", 0.04) * H))
    bot = int(H * (1.0 - cfg.get("ignore_bottom_frac", 0.0)))
    k = max(3, int(round(W * cfg.get("ridge_k_frac", 0.009))))
    sizes = [cfg["force_size"]] if cfg.get("force_size") else list(cfg.get("allowed_sizes", [4, 6, 8, 9, 12, 16]))
    minw = cfg.get("grid_min_width_frac", 0.5)
    for delta in (cfg.get("line_delta", 18), cfg.get("line_delta", 18) * 0.55):
        res = _detect_grid_once(g, top, bot, k, delta, sizes, cfg, W, H, minw)
        if res:
            res["delta"] = delta
            res.update(detect_boxes(g, res, cfg))
            return res
    return None


def _detect_grid_once(g, top, bot, k, delta, sizes, cfg, W, H, minw):
    thr = cfg.get("line_min_cover", 0.42)
    tol = cfg.get("lattice_tol", 0.16)
    mins = cfg.get("lattice_min_score", 0.8)
    band_x0 = int(W * (0.5 - minw / 2.0))
    band_x1 = int(W * (0.5 + minw / 2.0))
    sub = g[top:bot]
    hmask = _ridges(sub, k, delta, 0)
    hcov = hmask[:, band_x0:band_x1].mean(axis=1)
    hc = _candidates(hcov, thr, offset=top)
    hl = find_lattice(hc, sizes, tol, mins, minw * W * 0.9)
    if not hl:
        return None
    N = hl["n"]
    y0, y1 = hl["lines"][0], hl["lines"][-1]
    ya, yb = int(y0 + 2), int(y1 - 2)
    if yb - ya < 20:
        return None
    vsub = g[ya:yb]
    vmask = _ridges(vsub, k, delta, 1)
    vcov = vmask.mean(axis=0)
    vc = _candidates(vcov, thr + 0.1, offset=0)
    span = y1 - y0
    vl = find_lattice(vc, [N], tol, mins, span * (1 - cfg.get("square_tol", 0.12)))
    if not vl:
        return None
    x0, x1 = vl["lines"][0], vl["lines"][-1]
    w = x1 - x0
    if w < minw * W * 0.9:
        return None
    if abs(w / float(span) - 1.0) > cfg.get("square_tol", 0.12):
        return None
    # second passage horizontal restreint a la largeur trouvee (plus precis)
    xa, xb = int(x0 + 2), int(x1 - 2)
    hcov2 = hmask[:, xa:xb].mean(axis=1)
    hc2 = _candidates(hcov2, thr + 0.1, offset=top)
    hl2 = find_lattice(hc2, [N], tol, mins, span * 0.9)
    if hl2 and hl2["matched"] >= hl["matched"]:
        hl = hl2
    ys = hl["lines"]
    xs = vl["lines"]
    return {
        "n": N, "x0": xs[0], "y0": ys[0], "x1": xs[-1], "y1": ys[-1], "xs": xs, "ys": ys,
        "cw": (xs[-1] - xs[0]) / N, "ch": (ys[-1] - ys[0]) / N,
        "score": (hl["matched"] + vl["matched"]) / (2.0 * (N + 1)), "w": W, "h": H,
    }


def _line_strength(g, grid, horizontal):
    """Force (contraste x epaisseur) de chaque ligne interieure."""
    N = grid["n"]
    xs, ys = grid["xs"], grid["ys"]
    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]), int(ys[0]), int(ys[-1])
    region = g[y0:y1, x0:x1]
    dark_lines = float(np.median(region)) > 128   # theme clair : lignes plus sombres que le fond
    out = []
    lines = ys if horizontal else xs
    step = grid["ch"] if horizontal else grid["cw"]
    win = max(3, int(step * 0.14))
    lim = g.shape[0] if horizontal else g.shape[1]
    for kk in range(1, N):
        p = int(round(lines[kk]))
        qa, qb = max(0, p - win), min(lim - 1, p + win)
        if horizontal:
            prof = np.median(g[qa:qb + 1, x0 + 3:x1 - 3], axis=1)
        else:
            prof = np.median(g[y0 + 3:y1 - 3, qa:qb + 1], axis=0)
        if not dark_lines:
            prof = 255.0 - prof
        sl, sr = float(prof[0]), float(prof[-1])
        base = min(sl, sr)                 # cote le plus sombre : insensible aux surlignages
        center = float(prof.min())
        depth = base - center
        if depth <= 3:
            out.append(0.0)
            continue
        thick = int((prof < center + depth * 0.5).sum())
        out.append(depth * thick)
    return out


def _best_divisor(strengths, N, ratio_min):
    best_b, best_r = None, 0.0
    for b in range(2, N):
        if N % b:
            continue
        thick = [strengths[k - 1] for k in range(1, N) if k % b == 0]
        thin = [strengths[k - 1] for k in range(1, N) if k % b != 0]
        if not thick or not thin:
            continue
        r = (np.mean(thick) + 1e-3) / (np.mean(thin) + 1e-3)
        # exige que la plus faible des "epaisses" depasse la plus forte des "fines"
        sep = min(thick) / (max(thin) + 1e-3)
        if r > best_r and sep > 1.0:
            best_b, best_r = b, r
    if best_r >= ratio_min:
        return best_b, best_r
    return None, best_r


def detect_boxes(g, grid, cfg):
    N = grid["n"]
    if cfg.get("force_box"):
        br, bc = cfg["force_box"]
        return {"br": int(br), "bc": int(bc), "box_src": "force"}
    rmin = cfg.get("thick_ratio", 1.35)
    hs = _line_strength(g, grid, True)
    vs = _line_strength(g, grid, False)
    br, _ = _best_divisor(hs, N, rmin)
    bc, _ = _best_divisor(vs, N, rmin)
    src = "lignes"
    if br and not bc and N % br == 0:
        bc = N // br
    if bc and not br and N % bc == 0:
        br = N // bc
    if not br or not bc or br * bc != N:
        t = cfg.get("box_table", {}).get(str(N))
        if t:
            br, bc = t
        else:
            r = int(math.sqrt(N))
            while r > 1 and N % r:
                r -= 1
            br, bc = r, N // r
        src = "table"
    return {"br": int(br), "bc": int(bc), "box_src": src, "hs": hs, "vs": vs}


def cell_rect(grid, r, c):
    xs, ys = grid["xs"], grid["ys"]
    return xs[c], ys[r], xs[c + 1], ys[r + 1]


def cell_center(grid, r, c):
    x0, y0, x1, y1 = cell_rect(grid, r, c)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


# ---------------------------------------------------------------- glyphes
def glyph_from_weights(wimg, S):
    """wimg : poids d'encre (0..1) deja recadre sur la boite englobante -> vecteur SxS."""
    bh, bw = wimg.shape
    if bh < 2 or bw < 1:
        return None, 1.0
    scale = (S - 4) / float(bh)
    if bw * scale > S - 4:
        scale = (S - 4) / float(bw)
    nw = max(1, int(round(bw * scale)))
    nh = max(1, int(round(bh * scale)))
    im = Image.fromarray(np.clip(wimg * 255, 0, 255).astype(np.uint8)).resize((nw, nh), Image.BILINEAR)
    canvas = np.zeros((S, S), dtype=np.float32)
    ox = (S - nw) // 2
    oy = (S - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = np.asarray(im, dtype=np.float32) / 255.0
    return canvas.reshape(-1), bw / float(bh)


def ink_patch(patch, ink_delta, bg=None):
    p = patch.astype(np.float32)
    if bg is None:
        bg = np.median(p.reshape(-1, 3), axis=0)
    dist = np.sqrt(((p - bg) ** 2).sum(axis=2))
    return dist, bg


def analyse_patch(patch, cfg, min_h_ref, bg=None):
    """Renvoie dict(glyph, aspect, frac, hfrac, color) ou None si vide."""
    delta = cfg.get("ink_delta", 70)
    dist, bg = ink_patch(patch, delta, bg)
    ink = dist > delta
    frac = float(ink.mean())
    if frac < cfg.get("ink_min_frac", 0.012):
        return None
    rows = np.flatnonzero(ink.sum(axis=1) >= 1)
    cols = np.flatnonzero(ink.sum(axis=0) >= 1)
    if rows.size == 0 or cols.size == 0:
        return None
    ra, rb, ca, cb = rows[0], rows[-1], cols[0], cols[-1]
    hfrac = (rb - ra + 1) / float(min_h_ref)
    if hfrac < cfg.get("glyph_min_h", 0.28):
        return None
    w = np.clip((dist[ra:rb + 1, ca:cb + 1] - delta * 0.5) / float(delta), 0, 1)
    vec, aspect = glyph_from_weights(w, int(cfg.get("glyph_size", 28)))
    if vec is None:
        return None
    # Couleur de l'ENCRE (pas du fond !) : médiane sur les pixels d'encre du
    # COEUR du patch (moitie centrale). Un cercle/surlignage rouge autour
    # d'un chiffre noir reste en peripherie : le coeur reste noir -> "given",
    # pas "error". Repli sur tous les pixels si le coeur est quasi vide.
    hh, ww = ink.shape
    core = ink[hh // 4:hh - hh // 4, ww // 4:ww - ww // 4]
    core_px = patch[hh // 4:hh - hh // 4, ww // 4:ww - ww // 4][core]
    if core_px.shape[0] >= 10:
        col = np.median(core_px.reshape(-1, 3), axis=0)
    else:
        ink_px = patch[ink].reshape(-1, 3)
        col = np.median(ink_px, axis=0) if ink_px.size else np.array([0, 0, 0])
    return {"glyph": vec, "aspect": aspect, "frac": frac, "hfrac": hfrac,
            "color": classify_color(col, cfg), "rgb": [int(x) for x in col],
            "bbox": [int(ca), int(ra), int(cb), int(rb)]}


def classify_color(rgb, cfg=None):
    """Encre rouge (vif OU pale) -> 'error' ; bleue/verte -> 'user' ; sinon 'given'.

    Un chiffre noir sur case rose (surlignage) reste 'given' : seule compte la
    couleur de l'encre. Seuils reglables (vision.red_min_*), calibres sur
    captures reelles : rouge pale (232,174,188) -> error, noir (49,50,56) ->
    given, avec ~20 points de marge de chaque cote.
    """
    cfg = cfg or {}
    try:
        min_r = float(cfg.get("red_min_r", 110))
    except Exception:
        min_r = 110.0
    try:
        min_rg = float(cfg.get("red_min_rg", 30))
    except Exception:
        min_rg = 30.0
    try:
        min_rb = float(cfg.get("red_min_rb", 25))
    except Exception:
        min_rb = 25.0
    r, g, b = [float(x) for x in rgb]
    if r > min_r and r > g + min_rg and r > b + min_rb:
        return "error"
    if b > r + 35 and b > 90:
        return "user"
    if g > r + 40 and g > b + 10:
        return "user"
    return "given"


def _norm(v):
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


# ---------------------------------------------------------------- gabarits
class Templates:
    """Gabarits etiquetes : sources 'learned' (persistes), 'keypad' (ecran courant), 'font'."""

    def __init__(self):
        self.items = []   # (label, src, vecnorm, aspect)
        self._mat = None
        self.font_done = set()
        self.load()

    def add(self, label, src, vec, aspect):
        self.items.append((str(label), src, _norm(np.asarray(vec, dtype=np.float32)), float(aspect)))
        self._mat = None

    def clear_src(self, src):
        self.items = [t for t in self.items if t[1] != src]
        self._mat = None

    def count(self, src=None):
        return sum(1 for t in self.items if src is None or t[1] == src)

    def _build(self):
        if self._mat is None and self.items:
            self._mat = np.stack([t[2] for t in self.items])
            self._labels = [t[0] for t in self.items]
            self._srcs = [t[1] for t in self.items]
            self._asp = np.array([t[3] for t in self.items], dtype=np.float32)
        return self._mat

    def match(self, vec, aspect, allowed=None):
        mat = self._build()
        if mat is None:
            return []
        v = _norm(np.asarray(vec, dtype=np.float32))
        sc = mat @ v
        sc = sc - 0.22 * np.abs(self._asp - aspect)
        bonus = {"learned": 0.03, "keypad": 0.0, "font": -0.03}
        best = {}
        for i, s in enumerate(sc):
            lab = self._labels[i]
            if allowed is not None and lab not in allowed:
                continue
            s = float(s) + bonus.get(self._srcs[i], 0)
            if s > best.get(lab, -9):
                best[lab] = s
        return sorted(best.items(), key=lambda kv: -kv[1])

    # ---- polices systeme (secours, toujours disponibles sur Android)
    def ensure_fonts(self, symbols, cfg):
        key = tuple(symbols)
        if key in self.font_done:
            return
        self.font_done.add(key)
        S = int(cfg.get("glyph_size", 28))
        fonts = []
        for p in cfg.get("system_fonts", []):
            if os.path.exists(p):
                try:
                    fonts.append(ImageFont.truetype(p, 110))
                except Exception:
                    pass
        if not fonts:
            try:
                fonts.append(ImageFont.load_default(size=110))
            except Exception:
                pass
        for f in fonts:
            for s in symbols:
                im = Image.new("L", (260, 200), 255)
                d = ImageDraw.Draw(im)
                d.text((20, 20), s, fill=0, font=f)
                a = 255.0 - np.asarray(im, dtype=np.float32)
                ink = a > 60
                rows = np.flatnonzero(ink.any(axis=1))
                cols = np.flatnonzero(ink.any(axis=0))
                if rows.size == 0:
                    continue
                w = np.clip(a[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] / 255.0, 0, 1)
                vec, asp = glyph_from_weights(w, S)
                if vec is not None:
                    self.add(s, "font", vec, asp)

    # ---- persistance des gabarits appris
    def path(self):
        return config.data_path("templates.npz")

    def load(self):
        try:
            z = np.load(self.path(), allow_pickle=False)
            for lab, vec, asp in zip(z["labels"], z["vecs"], z["asp"]):
                self.add(str(lab), "learned", vec, float(asp))
        except Exception:
            pass

    def save(self):
        learned = [t for t in self.items if t[1] == "learned"]
        if not learned:
            return
        np.savez(self.path(), labels=np.array([t[0] for t in learned]),
                 vecs=np.stack([t[2] for t in learned]), asp=np.array([t[3] for t in learned]))

    def learn(self, label, vec, aspect, max_per):
        label = str(label)
        same = [i for i, t in enumerate(self.items) if t[1] == "learned" and t[0] == label]
        v = _norm(np.asarray(vec, dtype=np.float32))
        for i in same:   # evite les doublons quasi identiques
            if float(self.items[i][2] @ v) > 0.985:
                return False
        if len(same) >= max_per:
            del self.items[same[0]]
        self.items.append((label, "learned", v, float(aspect)))
        self._mat = None
        return True


def symbols_for(n, cfg):
    sy = cfg.get("symbols")
    if isinstance(sy, dict):
        sy = sy.get(str(n))
    if sy and len(sy) == n:
        return [str(s) for s in sy]
    return [str(i) for i in range(1, n + 1)]


# ---------------------------------------------------------------- lecture des cases
def read_cells(rgb, grid, templates, cfg=None):
    cfg = cfg or config.section("vision")
    N = grid["n"]
    symbols = symbols_for(N, cfg)
    templates.ensure_fonts(symbols, cfg)
    allowed = set(symbols)
    m = cfg.get("cell_margin", 0.13)
    cells = []
    for r in range(N):
        row = []
        for c in range(N):
            x0, y0, x1, y1 = cell_rect(grid, r, c)
            cw, ch = x1 - x0, y1 - y0
            xa, xb = int(x0 + cw * m), int(x1 - cw * m)
            ya, yb = int(y0 + ch * m), int(y1 - ch * m)
            patch = rgb[ya:yb, xa:xb]
            info = {"r": r, "c": c, "v": 0, "score": 1.0, "alts": [], "color": None}
            a = analyse_patch(patch, cfg, ch) if patch.size else None
            if a:
                ranking = templates.match(a["glyph"], a["aspect"], allowed)
                info["glyph"] = a["glyph"]
                info["aspect"] = a["aspect"]
                info["color"] = a["color"]
                if ranking:
                    lab, sc = ranking[0]
                    info["v"] = symbols.index(lab) + 1
                    info["score"] = sc
                    info["alts"] = [(symbols.index(l) + 1, s) for l, s in ranking[1:4]]
                    second = ranking[1][1] if len(ranking) > 1 else -1
                    info["margin"] = sc - second
                else:
                    info["v"] = -1
                    info["score"] = 0.0
            row.append(info)
        cells.append(row)
    return cells


def cells_to_grid(cells):
    return [[max(0, x["v"]) for x in row] for row in cells]


# ---------------------------------------------------------------- pave numerique
def _single_row(blobs, tol=0.6):
    """Vrai si tous les blobs sont sur la meme ligne (rejette les bandes
    melangeant pave + toolbar/compteurs : ecarts en y contenus)."""
    if len(blobs) < 2:
        return True
    ys = sorted(q["y"] for q in blobs)
    hs = sorted(q["h"] for q in blobs)
    med, hmed = ys[len(ys) // 2], hs[len(hs) // 2]
    return max(abs(y - med) for y in ys) <= tol * hmed


def _roughly_regular(bl, tol=0.25):
    """Comme regular(), mais tolere les TROUS du pave partiel : le pas est le
    PLUS PETIT ecart (touches adjacentes), et chaque ecart doit en etre un
    multiple entier (simple, double/triple si des chiffres masques...).
    Utilise pour les bandes partielles (l'etiquetage par gabarits/comptes
    tranche ensuite) ; la voie complete garde regular()."""
    if len(bl) < 3:
        return True
    d = np.diff(sorted(q["x"] for q in bl))
    d = d[d > 0]
    if len(d) == 0:
        return False
    m = float(d.min())
    if m < 1:
        return False
    for v in d:
        k = max(1, int(round(v / m)))
        if abs(v - k * m) > max(2.0, tol * m):
            return False
    return True


def _keypad_verify_order(blobs, N, templates, cfg, rgb):
    """Verifie l'etiquetage gauche->droite = 1..N par les gabarits.

    Un desaccord CONFIANT (gabarit sur, autre chiffre) invalide la rangee
    (ex. rangee partielle + blob parasite qui decalerait tout). Abstention
    si les gabarits sont muets (on ne rejette pas sans preuve).
    """
    symbols = symbols_for(N, cfg)
    templates.ensure_fonts(symbols, cfg)
    try:
        thr = max(0.30, float(cfg.get("match_min_score", 0.45)) - 0.15)
    except Exception:
        thr = 0.32
    for i, bl in enumerate(sorted(blobs, key=lambda q: q["x"])):
        patch = _pad_crop(rgb, bl)
        if patch.size == 0:
            return False
        an = analyse_patch(patch, dict(cfg, ink_min_frac=0.002, glyph_min_h=0.3),
                           max(1, patch.shape[0]))
        rk = templates.match(an["glyph"], an["aspect"], set(symbols)) if an else []
        if rk and rk[0][1] > thr and rk[0][0] != symbols[i]:
            return False
    return True


def _keypad_label_with_templates(blobs, N, templates, cfg, rgb):
    """Etiquette des blobs (tries par x) via les gabarits. Renvoie la liste des valeurs ou None."""
    symbols = symbols_for(N, cfg)
    templates.ensure_fonts(symbols, cfg)
    # seuil un peu plus bas que pour les cases : les chiffres du pave sont grands et nets
    try:
        thr = max(0.30, float(cfg.get("match_min_score", 0.45)) - 0.15)
    except Exception:
        thr = 0.32
    labs = []
    for bl in sorted(blobs, key=lambda q: q["x"]):
        patch = _pad_crop(rgb, bl)
        if patch.size == 0:
            return None
        an = analyse_patch(patch, dict(cfg, ink_min_frac=0.002, glyph_min_h=0.3), patch.shape[0])
        rk = templates.match(an["glyph"], an["aspect"], set(symbols)) if an else []
        if rk and rk[0][1] > thr:
            labs.append(symbols.index(rk[0][0]) + 1)
        else:
            return None
    if len(set(labs)) != len(labs):
        return None
    if any(not 1 <= v <= N for v in labs):
        return None
    # le jeu compacte les chiffres restants vers la gauche : l'ordre doit rester croissant
    if labs != sorted(labs):
        return None
    return labs


def _keypad_infer_from_counts(blobs, N, counts):
    """Deduit les valeurs par l'ordre + les chiffres deja complets dans la grille.

    counts : dict {v: nb d'occurrences} ou liste [c1..cN]. Les chiffres avec
    count >= N sont consideres comme termines (masques par le jeu).
    Si le nombre de chiffres attendus == nb de blobs, on associe dans l'ordre.
    """
    if not counts:
        return None
    try:
        if isinstance(counts, dict):
            cnt = [int(counts.get(v, 0)) for v in range(1, N + 1)]
        else:
            cnt = [int(c) for c in counts]
        if len(cnt) != N:
            return None
    except Exception:
        return None
    expected = [v for v in range(1, N + 1) if cnt[v - 1] < N]
    if len(expected) != len(blobs):
        return None
    # blobs deja tries par x par l'appelant
    return list(expected)


def detect_keypad(rgb, grid, templates=None, cfg=None, counts=None):
    cfg = cfg or config.section("vision")
    H, W = rgb.shape[:2]
    N = grid["n"]
    ch = grid["ch"]
    if cfg.get("keypad_manual"):
        km = cfg["keypad_manual"]
        y = km.get("y_frac", 0.86) * H
        xa, xb = km.get("x0_frac", 0.06) * W, km.get("x1_frac", 0.94) * W
        step = (xb - xa) / float(N)
        return [{"v": i + 1, "x": xa + step * (i + 0.5), "y": y, "w": step * 0.6, "h": ch * 0.6, "src": "manuel"}
                for i in range(N)]
    ya = int(grid["y1"] + ch * 0.15)
    yb = int(H * (1.0 - cfg.get("ignore_bottom_frac", 0.0)))
    if yb - ya < 10:
        return None
    sub = rgb[ya:yb].astype(np.float32)
    bg = np.median(sub, axis=1, keepdims=True)
    dist = np.sqrt(((sub - bg) ** 2).sum(axis=2))
    ink = dist > cfg.get("keypad_ink_delta", 60)
    rowc = ink.sum(axis=1)
    bands = _groups(np.flatnonzero(rowc >= 2), 2)
    hmin = cfg.get("keypad_min_h_frac", 0.22) * ch
    hmax = cfg.get("keypad_max_h_frac", 2.2) * ch
    found = []
    for (a, b) in bands:
        bh = b - a + 1
        if bh < hmin or bh > hmax:
            continue
        colc = ink[a:b + 1].sum(axis=0)
        segs = _groups(np.flatnonzero(colc >= 1), max(1, int(bh * cfg.get("keypad_merge_gap_frac", 0.18))))
        blobs = []
        for (ca, cb) in segs:
            sl = ink[a:b + 1, ca:cb + 1]
            rr = np.flatnonzero(sl.any(axis=1))
            if rr.size == 0:
                continue
            bw = cb - ca + 1
            hh = rr[-1] - rr[0] + 1
            if hh < hmin * 0.8 or bw > hh * 2.2:
                continue
            blobs.append({"x0": ca, "x1": cb, "y0": a + rr[0] + ya, "y1": a + rr[-1] + ya,
                          "x": (ca + cb) / 2.0, "y": ya + a + (rr[0] + rr[-1]) / 2.0, "w": bw, "h": hh})
        if not blobs:
            continue
        hmed = float(np.median([bl["h"] for bl in blobs]))
        blobs = [bl for bl in blobs if bl["h"] >= 0.55 * hmed]
        found.append({"a": a, "b": b, "h": hmed, "blobs": blobs})

    def regular(bl):
        if len(bl) < 3:
            return True
        cx = np.array([q["x"] for q in bl])
        d = np.diff(cx)
        return d.min() > 0 and (d.std() / (d.mean() + 1e-6)) < 0.3

    best = None
    for f in found:      # une seule rangee de N chiffres
        if len(f["blobs"]) == N and regular(f["blobs"]) and _single_row(f["blobs"]):
            key = (f["h"], -f["a"])
            if best is None or key > best[0]:
                best = (key, f["blobs"])
    if best is not None and templates is not None:
        # L'ordre gauche->droite = 1..N n'est valable que pour un pave
        # COMPLET : on le fait confirmer par les gabarits, sinon un blob
        # parasite (toolbar, compteur) decalerait toutes les valeurs.
        if not _keypad_verify_order(best[1], N, templates, cfg, rgb):
            best = None
    if best is None and N > 9:     # deux rangees consecutives (grilles > 9 ;
        # un pave <= 9 tient sur une seule rangee : ne jamais fusionner)
        for i in range(len(found) - 1):
            b1, b2 = found[i]["blobs"], found[i + 1]["blobs"]
            if len(b1) + len(b2) == N and regular(b1) and regular(b2):
                best = ((0, 0), b1 + b2)
                break
    keys = None
    if best is not None:
        ordered = sorted(best[1], key=lambda q: q["x"])
        keys = [dict(bl, v=i + 1, src="ordre") for i, bl in enumerate(ordered)]
    else:
        # --- pavé partiel : certains chiffres sont masqués car déjà placés ---
        # bandes candidates (1..N chiffres), les plus hautes d'abord. Les trous
        # (chiffre masque au milieu) sont toleres : _roughly_regular accepte
        # les pas simples/doubles, l'etiquetage tranche ensuite.
        cands = [f for f in found if 1 <= len(f["blobs"]) <= N and _roughly_regular(f["blobs"])
                 and _single_row(f["blobs"])]
        cands = sorted(cands, key=lambda f: -f["h"])
        # + paires de rangées consécutives (GRILLES > 9 SEULEMENT : un pave
        # <= 9 tient sur une rangee, fusionner melangerait toolbar/compteurs)
        if N > 9:
            for i in range(len(found) - 1):
                b1, b2 = found[i]["blobs"], found[i + 1]["blobs"]
                tot = len(b1) + len(b2)
                if 2 <= tot <= N and regular(b1) and regular(b2):
                    h = max(found[i]["h"], found[i + 1]["h"])
                    # tri gauche->droite pour l'étiquetage (les 2 rangées sont déjà
                    # ordonnées, on réordonne par x uniquement si une seule rangée)
                    merged_x = sorted(b1 + b2, key=lambda q: q["x"])
                    cands.append({"a": found[i]["a"], "b": found[i + 1]["b"], "h": h * 0.99,
                                  "blobs": merged_x})
        if templates is not None:
            for f in cands:
                ordered = sorted(f["blobs"], key=lambda q: q["x"])
                labs = _keypad_label_with_templates(ordered, N, templates, cfg, rgb)
                if labs:
                    keys = [dict(bl, v=v, src="gabarit") for bl, v in zip(ordered, labs)]
                    break
        if keys is None and counts:
            for f in cands:
                # L'inference par les comptes ne verifie pas les formes : on
                # l'interdit au-dessus de la zone pave (toolbar...), sinon des
                # icones regulierement espacees recevraient des chiffres.
                try:
                    band_y = sum(q["y"] for q in f["blobs"]) / max(1, len(f["blobs"]))
                    if band_y < grid["y1"] + 2.5 * grid["ch"]:
                        continue
                except Exception:
                    pass
                ordered = sorted(f["blobs"], key=lambda q: q["x"])
                labs = _keypad_infer_from_counts(ordered, N, counts)
                if labs:
                    keys = [dict(bl, v=v, src="grille") for bl, v in zip(ordered, labs)]
                    break
    if not keys:
        return None
    # glyphes du pave = gabarits pour la lecture des cases
    for kd in keys:
        patch = _pad_crop(rgb, kd)
        an = analyse_patch(patch, dict(cfg, ink_min_frac=0.002, glyph_min_h=0.3), max(1, patch.shape[0]))
        if an:
            kd["glyph"] = an["glyph"]
            kd["aspect"] = an["aspect"]
    return keys


def _pad_crop(rgb, bl):
    H, W = rgb.shape[:2]
    pad = max(3, int(bl["h"] * 0.3))
    y0 = max(0, int(bl["y0"]) - pad)
    y1 = min(H, int(bl["y1"]) + pad + 1)
    x0 = max(0, int(bl["x0"]) - pad)
    x1 = min(W, int(bl["x1"]) + pad + 1)
    return rgb[y0:y1, x0:x1]


def keypad_to_templates(keys, templates, symbols):
    templates.clear_src("keypad")
    for kd in keys or []:
        if "glyph" in kd and 1 <= kd["v"] <= len(symbols):
            templates.add(symbols[kd["v"] - 1], "keypad", kd["glyph"], kd["aspect"])


def detect_quick_toggle(rgb, grid):
    """Interrupteur « placement rapide » sous la grille, a droite.

    Bleu = actif (a desactiver d'un appui), gris/blanc = inactif.
    Renvoie {"active": bool, "x": cx, "y": cy, "frac": part de bleu} ou None
    si la zone est inexploitable. Mesure sur capture reelle 1080x2340 :
    bleu en x 0.93..0.99, y = y1 + 0.17..0.50 ch (frac ~0.08 vs 0.00).
    """
    try:
        H, W = rgb.shape[:2]
        x0 = max(0, int(grid["x1"] - W * 0.20))
        x1 = max(0, W - 1)
        ya = max(0, int(grid["y1"]))
        yb = min(H - 1, int(grid["y1"] + grid["ch"] * 0.9))
        if x1 - x0 < 20 or yb - ya < 10:
            return None
        sub = rgb[ya:yb, x0:x1].astype(np.int16)
        B, R, G = sub[..., 2], sub[..., 0], sub[..., 1]
        blue = (B > 140) & (B - R > 50) & (B - G > 40)
        frac = float(blue.mean())
        if frac <= 0.03:
            return {"active": False, "x": None, "y": None, "frac": frac}
        ys, xs = np.nonzero(blue)
        cx = float(x0 + xs.mean())
        cy = float(ya + ys.mean())
        if cx >= W - 2 or cy >= H - 2:
            return None
        return {"active": True, "x": cx, "y": cy, "frac": frac}
    except Exception:
        return None


# ---------------------------------------------------------------- overlay de debug
def render(rgb, grid=None, cells=None, keys=None, taps=None, solution=None, scale=0.5):
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    if grid:
        N = grid["n"]
        for i, x in enumerate(grid["xs"]):
            w = 6 if i % grid.get("bc", N) == 0 else 2
            d.line([(x, grid["y0"]), (x, grid["y1"])], fill=(30, 90, 255), width=w)
        for i, y in enumerate(grid["ys"]):
            w = 6 if i % grid.get("br", N) == 0 else 2
            d.line([(grid["x0"], y), (grid["x1"], y)], fill=(30, 90, 255), width=w)
        try:
            font = ImageFont.load_default(size=max(14, int(grid["ch"] * 0.32)))
        except Exception:
            font = ImageFont.load_default()
        if cells:
            for row in cells:
                for cinfo in row:
                    x0, y0, _, _ = cell_rect(grid, cinfo["r"], cinfo["c"])
                    if cinfo["v"]:
                        col = (0, 170, 60) if cinfo["score"] >= 0.6 else (255, 140, 0)
                        d.text((x0 + 4, y0 + 2), str(cinfo["v"]), fill=col, font=font)
                    elif solution:
                        d.text((x0 + 4, y0 + 2), str(solution[cinfo["r"]][cinfo["c"]]), fill=(200, 0, 200), font=font)
    if keys:
        for kd in keys:
            d.rectangle([kd["x"] - kd["w"] / 2 - 6, kd["y"] - kd["h"] / 2 - 6,
                         kd["x"] + kd["w"] / 2 + 6, kd["y"] + kd["h"] / 2 + 6], outline=(230, 0, 160), width=4)
    for t in taps or []:
        d.ellipse([t[0] - 14, t[1] - 14, t[0] + 14, t[1] + 14], outline=(230, 0, 160), width=4)
    if scale != 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=70)
    return buf.getvalue()


def thumb(rgb, width=180):
    im = Image.fromarray(rgb)
    h = int(im.height * width / float(im.width))
    im = im.resize((width, h), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=60)
    return buf.getvalue()

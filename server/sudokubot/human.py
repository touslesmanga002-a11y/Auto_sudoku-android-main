"""Humanisation : points d'appui disperses, durees variables, rythme irregulier, ordre de jeu naturel."""
import random


def gauss_clip(mean, sd, lo=None, hi=None):
    v = random.gauss(mean, sd)
    if lo is not None:
        v = max(lo, v)
    if hi is not None:
        v = min(hi, v)
    return v


def point_in(cx, cy, w, h, sigma, maxf, edge_prob=0.0, edge_max=0.46):
    """Point gaussien autour du centre, borne a maxf * taille (reste dans la case).

    Avec une proba edge_prob, la borne passe a edge_max (presque au bord,
    jamais dessus) : un humain ne tape pas toujours au centre.
    """
    try:
        bound = maxf
        if random.random() < float(edge_prob or 0.0):
            bound = max(maxf, float(edge_max if edge_max else maxf))
    except Exception:
        bound = maxf
    dx = gauss_clip(0, sigma * w, -bound * w, bound * w)
    dy = gauss_clip(0, sigma * h, -bound * h, bound * h)
    # legere tendance droitier : un peu bas-droite
    dx += w * 0.02
    dy += h * 0.03
    return cx + dx, cy + dy


def tap_duration(level):
    lo, hi = level["tap_ms"]
    # distribution asymetrique (log-normale simplifiee)
    u = random.random() ** 1.6
    return int(lo + (hi - lo) * u)


def delay(pair, factor=1.0, floor=15):
    mean, sd = pair
    return max(floor, int(gauss_clip(mean, sd, mean * 0.35, mean * 2.6) * factor))


def think_pause(hcfg, level, factor=1.0):
    p = hcfg.get("think_prob", 0.06) * level.get("think_factor", 0.5)
    if random.random() < p:
        a, b = hcfg.get("think_ms", [500, 1600])
        return int(random.uniform(a, b) * factor)
    return 0


def order_moves(moves, mode, br, bc, cand_count=None):
    """moves : liste de (r, c, v). Renvoie une liste reordonnee."""
    mv = list(moves)
    if mode == "rows":
        return sorted(mv, key=lambda m: (m[0], m[1]))
    if mode == "random":
        random.shuffle(mv)
        return mv
    if mode == "digit":
        random.shuffle(mv)
        return sorted(mv, key=lambda m: m[2])
    if mode == "box":
        return sorted(mv, key=lambda m: ((m[0] // br), (m[1] // bc), random.random()))
    # "human" : commence par les cases evidentes, puis enchaine par proximite / meme chiffre
    cc = cand_count or {}
    remaining = mv[:]
    out = []
    cur = None
    while remaining:
        def score(m):
            s = cc.get((m[0], m[1]), 3) * 1.0
            if cur is not None:
                same_digit = m[2] == cur[2]
                same_box = (m[0] // br == cur[0] // br) and (m[1] // bc == cur[1] // bc)
                same_line = m[0] == cur[0] or m[1] == cur[1]
                s -= 1.2 if same_digit else 0
                s -= 0.8 if same_box else 0
                s -= 0.4 if same_line else 0
                s += 0.08 * (abs(m[0] - cur[0]) + abs(m[1] - cur[1]))
            return s + random.uniform(0, 1.1)
        nxt = min(remaining, key=score)
        remaining.remove(nxt)
        out.append(nxt)
        cur = nxt
    return out

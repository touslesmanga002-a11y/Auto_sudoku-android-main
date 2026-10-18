"""Paliers de vitesse + mode adaptatif ("auto")."""
import json
import threading
import time

from . import config


class Speed:
    def __init__(self):
        self.lock = threading.Lock()
        s = config.section("speed")
        self.choice = s.get("default", "auto")
        self.auto_level = s.get("auto", {}).get("start", "moyen")
        self.good = 0
        self.bad = 0
        self.history = []

    # ----- infos
    def order(self):
        return config.section("speed").get("order", [])

    def levels(self):
        return config.section("speed").get("levels", {})

    def choices(self):
        out = [{"id": "auto", "label": "Auto (adaptatif)"}]
        for k in self.order():
            out.append({"id": k, "label": self.levels().get(k, {}).get("label", k)})
        return out

    def set_choice(self, c):
        with self.lock:
            if c == "auto" or c in self.levels():
                self.choice = c
                self.good = self.bad = 0
                return True
            return False

    def effective(self):
        with self.lock:
            return self.auto_level if self.choice == "auto" else self.choice

    def level(self):
        name = self.effective()
        lv = dict(self.levels().get(name) or self.levels().get("moyen") or {})
        lv["name"] = name
        return lv

    def label(self):
        eff = self.effective()
        lab = self.levels().get(eff, {}).get("label", eff)
        return ("Auto > " + lab) if self.choice == "auto" else lab

    def one_go(self):
        """True = toute la partie dans une seule salve (cle speed.one_go) :
        aucune pause de verification en cours de partie."""
        return bool(config.section("speed").get("one_go", False))

    # ----- adaptation
    def report(self, lag_ms, misses, shot_ms, taps):
        """Appele apres chaque salve verifiee."""
        a = config.section("speed").get("auto", {})
        rec = {"t": time.time(), "level": self.effective(), "lag": lag_ms, "miss": misses,
               "shot": shot_ms, "taps": taps}
        with self.lock:
            self.history.append(rec)
            self.history = self.history[-200:]
            if self.choice != "auto":
                return None
            order = self.order()
            lo = order.index(a.get("min", order[0])) if a.get("min") in order else 0
            hi = order.index(a.get("max", order[-1])) if a.get("max") in order else len(order) - 1
            cur = order.index(self.auto_level) if self.auto_level in order else lo
            bad = misses > 0 or lag_ms > a.get("lag_bad_ms", 160) or shot_ms > a.get("shot_bad_ms", 1800)
            ok = misses == 0 and lag_ms <= a.get("lag_ok_ms", 45)
            changed = None
            if bad:
                self.bad += 1
                self.good = 0
                if self.bad >= a.get("down_after", 1) and cur > lo:
                    cur -= 1
                    changed = "down"
                    self.bad = 0
            elif ok:
                self.good += 1
                self.bad = 0
                if self.good >= a.get("up_after", 2) and cur < hi:
                    cur += 1
                    changed = "up"
                    self.good = 0
            self.auto_level = order[cur]
            return changed

    def save_perf(self):
        try:
            with open(config.data_path("perf.json"), "w") as f:
                json.dump({"auto_level": self.auto_level, "history": self.history[-50:]}, f)
        except Exception:
            pass

    def load_perf(self):
        try:
            with open(config.data_path("perf.json")) as f:
                d = json.load(f)
            if d.get("auto_level") in self.levels():
                self.auto_level = d["auto_level"]
        except Exception:
            pass

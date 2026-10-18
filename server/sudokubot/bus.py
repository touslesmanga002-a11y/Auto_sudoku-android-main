"""AgentLink : file d'ordres entre le controleur (cerveau) et l'agent AutoJs6 (long-polling HTTP).

Agent  -> POST /api/agent/exchange {info, result, wait}  -> recoit {cmd} ou {cmd:null}
Controleur -> LINK.call(type, **params) (bloquant) -> dict resultat
"""
import itertools
import threading
import time


class AgentOffline(Exception):
    pass


class AgentLink:
    def __init__(self):
        self.cv = threading.Condition()
        self.pending = None        # ordre en attente d'envoi
        self.inflight = None       # ordre envoye, resultat attendu
        self.results = {}
        self.ids = itertools.count(1)
        self.last_seen = 0.0
        self.info = {}
        self.cancel_all = False

    # ---------- cote agent ----------
    def exchange(self, payload, wait=4.0):
        res = payload.get("result")
        with self.cv:
            self.last_seen = time.time()
            if payload.get("info"):
                self.info = payload["info"]
            if res and "id" in res:
                self.results[res["id"]] = res
                if self.inflight and self.inflight.get("id") == res["id"]:
                    self.inflight = None
                self.cv.notify_all()
            end = time.time() + max(0.0, min(float(wait), 10.0))
            while self.pending is None:
                left = end - time.time()
                if left <= 0:
                    return {"cmd": None}
                self.cv.wait(left)
                self.last_seen = time.time()
            cmd = self.pending
            self.pending = None
            self.inflight = cmd
            return {"cmd": cmd}

    # ---------- cote controleur ----------
    def online(self, max_age=12.0):
        return (time.time() - self.last_seen) < max_age

    def call(self, ctype, timeout=25.0, **params):
        cid = next(self.ids)
        cmd = dict(params)
        cmd["id"] = cid
        cmd["type"] = ctype
        with self.cv:
            # attend que la place soit libre
            t0 = time.time()
            while self.pending is not None or self.inflight is not None:
                if time.time() - t0 > timeout:
                    self.pending = None
                    self.inflight = None
                    break
                self.cv.wait(0.2)
            self.pending = cmd
            self.cv.notify_all()
            end = time.time() + timeout
            while cid not in self.results:
                left = end - time.time()
                if left <= 0:
                    if self.pending is cmd:
                        self.pending = None
                    if self.inflight is cmd:
                        self.inflight = None
                    raise AgentOffline("agent: pas de reponse a %s" % ctype)
                self.cv.wait(min(left, 0.5))
            res = self.results.pop(cid)
        if not res.get("ok", False):
            res.setdefault("error", "echec %s" % ctype)
        return res


LINK = AgentLink()

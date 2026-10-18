"""Serveur HTTP (stdlib) : API agent + API page web + fichiers statiques web/.

python -m sudokubot.server --port 8765 [--test] [--beta] [--disable-popup]
"""
import argparse
import io
import json
import mimetypes
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Permet `python server/sudokubot/server.py` (execution directe du fichier) :
# en mode script, les imports relatifs (`from . import ...`) sont impossibles,
# on bascule alors sur des imports absolus avec `server/` dans sys.path.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from sudokubot import config, vision
    from sudokubot.bus import LINK
    from sudokubot.controller import Controller
    from sudokubot import spots as spotsmod
else:
    from . import config, vision
    from .bus import LINK
    from .controller import Controller
    from . import spots as spotsmod

WEB = os.path.join(config.ROOT, "web")
CTRL = None
OPTS = {}


class Handler(BaseHTTPRequestHandler):
    server_version = "SudokuBot/1.0"

    def log_message(self, fmt, *args):   # silence
        pass

    # ---------------- helpers
    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def _json(self):
        raw = self._body()
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    # ---------------- GET
    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        q = parse_qs(u.query)
        if p == "/api/status":
            return self._send(200, CTRL.status())
        if p == "/api/logs":
            since = int((q.get("since") or ["0"])[0])
            return self._send(200, [l for l in list(CTRL.logs) if l["seq"] > since])
        if p == "/api/config":
            return self._send(200, config.get())
        if p == "/api/spots":
            refs = spotsmod.find_refs()
            info = {}
            for name, path in refs.items():
                info[name] = {"file": os.path.basename(path)}
                try:
                    from PIL import Image as _Im
                    with _Im.open(path) as _im:
                        info[name]["w"], info[name]["h"] = _im.width, _im.height
                except Exception:
                    pass
            return self._send(200, {"spots": CTRL.spots, "refs": info,
                                    "stale": spotsmod.is_stale(refs)
                                    or spotsmod.needs_calibration(CTRL.spots, refs),
                                    "auto": OPTS.get("auto", False),
                                    "enabled": config.section("menus").get("spots_enabled", True)})
        if p.startswith("/spots/img/"):
            name = os.path.basename(p[len("/spots/img/"):])
            if name not in spotsmod.REF_DEFS:
                return self._send(404, {"error": "reference inconnue"})
            refs = spotsmod.find_refs()
            if name not in refs:
                return self._send(404, {"error": "ajoutez %s.jpg dans emplacements/" % name})
            path = refs[name]
            ext = os.path.splitext(path)[1].lower()
            ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                     "webp": "image/webp", "bmp": "image/bmp"}.get(ext.lstrip("."), "image/jpeg")
            try:
                with open(path, "rb") as f:
                    return self._send(200, f.read(), ctype)
            except OSError:
                return self._send(404, {"error": "illisible"})
        if p == "/api/agent/gate":
            return self._send(200, {"go": CTRL.state != "idle"})
        if p == "/api/popup/state":
            st = CTRL.status()
            pcfg = config.section("popup")
            return self._send(200, {
                "enabled": pcfg.get("enabled", True) and not OPTS.get("disable_popup"),
                "position": pcfg.get("position", "top"), "poll_ms": pcfg.get("poll_ms", 600),
                "alpha": pcfg.get("alpha", 0.92),
                "state": st["state"], "msg": st["msg"], "label": st["speed_label"],
                "choice": st["speed_choice"], "choices": st["choices"], "games": st["games"],
                "moves": st["moves"], "rev": st["rev"],
            })
        if p == "/api/debug/latest.jpg":
            if CTRL.overlay:
                return self._send(200, CTRL.overlay, "image/jpeg")
            return self._send(404, {"error": "pas encore d'image"})
        if p == "/api/popup/thumb.jpg":
            if CTRL.thumb:
                return self._send(200, CTRL.thumb, "image/jpeg")
            return self._send(404, {"error": "pas de miniature"})
        if p == "/api/shot.jpg":
            if CTRL.last_rgb is not None:
                from PIL import Image
                buf = io.BytesIO()
                Image.fromarray(CTRL.last_rgb).save(buf, "JPEG", quality=80)
                return self._send(200, buf.getvalue(), "image/jpeg")
            return self._send(404, {"error": "pas de capture"})
        # fichiers statiques
        if p == "/":
            p = "/index.html"
        path = os.path.normpath(os.path.join(WEB, p.lstrip("/")))
        try:
            inside = os.path.commonpath([path, WEB]) == WEB
        except ValueError:
            inside = False  # chemins de racines differentes
        if inside and os.path.isfile(path):
            with open(path, "rb") as f:
                data = f.read()
            ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if ctype.startswith("text/") or ctype.endswith("javascript"):
                ctype += "; charset=utf-8"
            return self._send(200, data, ctype)
        return self._send(404, {"error": "introuvable"})

    # ---------------- POST
    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/api/agent/exchange":
            payload = self._json()
            wait = float(payload.get("wait", config.section("server").get("agent_wait_s", 4)))
            return self._send(200, LINK.exchange(payload, wait))
        if p == "/api/control":
            d = self._json()
            action = d.pop("action", "")
            return self._send(200, CTRL.control(action, **d))
        if p == "/api/config":
            d = self._json()
            if not isinstance(d, dict) or not d:
                return self._send(400, {"error": "JSON invalide"})
            config.save(d)
            CTRL.log("Configuration enregistrée")
            return self._send(200, {"ok": True})
        if p == "/api/config/patch":
            d = self._json()
            if not d.get("path") or "value" not in d:
                return self._send(400, {"error": "path/value manquants"})
            config.patch(d["path"], d["value"])
            CTRL.log("Réglage %s = %s" % (d["path"], json.dumps(d["value"], ensure_ascii=False)))
            return self._send(200, {"ok": True})
        if p == "/api/analyze":
            raw = self._body()
            try:
                rgb = vision.to_rgb_array(raw)
                t0 = time.time()
                grid = vision.detect_grid(rgb, 0)
                out = {"grid": None, "ms": 0}
                if grid:
                    keys = vision.detect_keypad(rgb, grid, CTRL.templates)
                    if keys:
                        vision.keypad_to_templates(keys, CTRL.templates, vision.symbols_for(grid["n"], config.section("vision")))
                    cells = vision.read_cells(rgb, grid, CTRL.templates)
                    CTRL.overlay = vision.render(rgb, grid, cells, keys)
                    CTRL.rev += 1
                    out["grid"] = {k: grid[k] for k in ("n", "br", "bc", "x0", "y0", "x1", "y1", "score", "box_src")}
                    out["values"] = vision.cells_to_grid(cells)
                    out["keypad"] = [{"v": k["v"], "x": k["x"], "y": k["y"]} for k in keys or []]
                out["ms"] = int((time.time() - t0) * 1000)
                return self._send(200, out)
            except Exception as e:
                return self._send(500, {"error": str(e)})
        if p == "/api/popup/close":
            config.patch("popup.enabled", False)
            return self._send(200, {"ok": True})
        if p == "/api/spots":
            d = self._json()
            raw = d.get("spots") if isinstance(d, dict) else None
            if not isinstance(raw, dict):
                return self._send(400, {"error": "JSON invalide (spots manquant)"})
            clean = spotsmod.sanitize(raw)
            spotsmod.save(clean)
            CTRL.spots = spotsmod.load()
            CTRL.log("Emplacements enregistrés")
            return self._send(200, {"ok": True, "spots": CTRL.spots})
        if p == "/api/spots/calibrate":
            try:
                new = spotsmod.calibrate(LINK.call, CTRL.log)
            except Exception as e:
                return self._send(500, {"error": str(e)})
            CTRL.spots = spotsmod.load()
            return self._send(200, {"ok": True, "spots": CTRL.spots})
        return self._send(404, {"error": "introuvable"})


def main():
    global CTRL, OPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--test", action="store_true", help="analyse seule, aucun appui")
    ap.add_argument("--beta", action="store_true", help="attend « Démarrer » sur la page web")
    ap.add_argument("--disable-popup", action="store_true")
    ap.add_argument("--auto", action="store_true",
                    help="tout a l'OCR, ignore emplacements/ (aucune image requise)")
    a = ap.parse_args()
    config.load()
    OPTS = {"test": a.test, "beta": a.beta, "disable_popup": a.disable_popup, "auto": a.auto}
    scfg = config.section("server")
    port = a.port or scfg.get("port", 8765)
    host = scfg.get("host", "127.0.0.1")
    CTRL = Controller(OPTS)
    CTRL.start()
    if not OPTS["auto"]:
        # Reconnaissance des references AVANT de jouer (bloque rien : le local
        # est immediat, l'OCR des boutons attend l'agent en fond si besoin).
        try:
            fresh = spotsmod.ensure_current(LINK.call, CTRL.log,
                                            on_done=lambda s: setattr(CTRL, "spots", s))
            CTRL.spots = fresh
            if fresh:
                CTRL.log("Emplacements : %s" %
                         ", ".join(sorted(k for k in fresh if not k.startswith("_"))))
        except Exception as e:
            CTRL.log("Emplacements : %s" % e, "warn")
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    with open(config.data_path("server.pid"), "w") as f:
        f.write(str(os.getpid()))
    print("SudokuBot sur http://%s:%d" % (host, port), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())

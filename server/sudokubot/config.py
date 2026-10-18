"""Chargement / sauvegarde de la configuration (config.json fusionne sur config.example.json)."""
import copy
import json
import os
import threading

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLE = os.path.join(ROOT, "config.example.json")
USER = os.path.join(ROOT, "config.json")
DATA = os.path.join(ROOT, "data")

_lock = threading.RLock()
_cfg = {}


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def load():
    global _cfg
    with _lock:
        _cfg = deep_merge(_read(EXAMPLE), _read(USER))
        os.makedirs(DATA, exist_ok=True)
        return _cfg


def get():
    with _lock:
        if not _cfg:
            load()
        return _cfg


def section(name):
    return get().get(name, {})


def save(new_cfg):
    """Remplace toute la config utilisateur (valide le JSON avant)."""
    global _cfg
    with _lock:
        merged = deep_merge(_read(EXAMPLE), new_cfg)
        tmp = USER + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(new_cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, USER)
        _cfg = merged
        return _cfg


def patch(path, value):
    """patch('speed.default', 'rapide') : modifie une cle et sauvegarde."""
    with _lock:
        user = _read(USER)
        cur = user
        keys = path.split(".")
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        cur[keys[-1]] = value
        return save(user)


def data_path(*parts):
    os.makedirs(DATA, exist_ok=True)
    return os.path.join(DATA, *parts)

#!/usr/bin/env python3
"""Lanceur unique SudokuBot : fonctionne sans PYTHONPATH manuel.

Utilisation (depuis la racine du projet) :
    python run.py [--port 8765] [--test] [--beta] [--disable-popup]
    python run.py --check   # verifie les imports + lance les tests unitaires

Ce script ajoute automatiquement `server/` a sys.path puis delegue a
`sudokubot.server:main`. C'est le moyen recommande de lancer le serveur
a la main (`./start.sh` l'utilise aussi en interne).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SERVER_DIR = os.path.join(ROOT, "server")
for _p in (SERVER_DIR, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import sudokubot  # noqa: F401
except ModuleNotFoundError as e:
    sys.stderr.write(
        "ERREUR: paquet 'sudokubot' introuvable.\n"
        "  - Lancez depuis la racine du projet :  python run.py\n"
        "  - Ou installez le projet :  pip install -e .\n"
        "  - Ou exportez :  PYTHONPATH=%s python -m sudokubot.server\n"
        % SERVER_DIR
    )
    raise SystemExit(1) from e


def _check():
    import unittest

    loader = unittest.TestLoader()
    tests_dir = os.path.join(ROOT, "tests")
    try:
        suite = loader.discover(tests_dir, top_level_dir=ROOT)
    except ImportError:
        # `tests/` sans __init__.py : decouverte classique
        suite = loader.discover(tests_dir)
    runner = unittest.TextTestRunner(verbosity=2)
    res = runner.run(suite)
    raise SystemExit(0 if res.wasSuccessful() else 1)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if "--check" in argv:
        argv.remove("--check")
        if not argv:
            return _check()
    from sudokubot.server import main as server_main

    sys.argv = [sys.argv[0]] + argv
    raise SystemExit(server_main())


if __name__ == "__main__":
    main()

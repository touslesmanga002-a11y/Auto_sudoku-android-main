"""Point d'entree `python -m sudokubot` (necessite `server/` dans sys.path).

Lancement recommande (aucune variable d'environnement requise) :
    python run.py
Lancement module (depuis la racine) :
    PYTHONPATH=server python -m sudokubot
"""
import os
import sys

# Secours si execute comme fichier (`python server/sudokubot/__main__.py`) :
if __package__ in (None, ""):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from sudokubot.server import main
else:
    from .server import main

raise SystemExit(main())

"""Bootstrap import : rend `sudokubot` importable sans PYTHONPATH manuel.

Pytest charge ce fichier avant la collection, donc `pytest tests/` fonctionne
depuis la racine, quel que soit l'environnement (Termux ou autre).
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_ROOT, "server"), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

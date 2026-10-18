import os
import sys
import unittest

# Bootstrap : rend `sudokubot` importable sans PYTHONPATH manuel,
# quel que soit le mode de lancement :
#   pytest | python -m unittest | python tests/test_solver.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sudokubot import solver

P9 = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

P6 = [
    [0, 0, 3, 0, 1, 0],
    [5, 6, 0, 3, 2, 0],
    [0, 5, 4, 2, 0, 3],
    [2, 0, 6, 4, 5, 0],
    [0, 1, 2, 0, 4, 5],
    [0, 4, 0, 1, 0, 0],
]

S4 = [
    [1, 2, 3, 4],
    [3, 4, 1, 2],
    [2, 1, 4, 3],
    [4, 3, 2, 1],
]

G4 = [row[:] for row in S4]
G4[3][3] = 0  # un seul trou -> solution unique S4


class SolverTest(unittest.TestCase):
    def check(self, sol, br, bc):
        n = len(sol)
        full = set(range(1, n + 1))
        for i in range(n):
            self.assertEqual(set(sol[i]), full)
            self.assertEqual({sol[r][i] for r in range(n)}, full)
        for b0 in range(0, n, br):
            for b1 in range(0, n, bc):
                self.assertEqual({sol[r][c] for r in range(b0, b0 + br) for c in range(b1, b1 + bc)}, full)

    def test_9x9(self):
        cnt, sol = solver.solve(P9, 3, 3)
        self.assertEqual(cnt, 1)
        self.check(sol, 3, 3)

    def test_6x6(self):
        cnt, sol = solver.solve(P6, 2, 3)
        self.assertGreaterEqual(cnt, 1)
        self.check(sol, 2, 3)

    def test_conflict(self):
        bad = [row[:] for row in P9]
        bad[0][2] = 5
        self.assertIn((0, 0), solver.conflicts(bad, 3, 3))
        self.assertEqual(solver.solve(bad, 3, 3)[0], 0)

    def test_banned_true_value_no_solution(self):
        # Le seul chiffre coherent est interdit (rouge) -> 0 solution.
        cnt, sol = solver.solve(G4, 2, 2, banned={(3, 3): {1}})
        self.assertEqual(cnt, 0)
        self.assertIsNone(sol)

    def test_banned_other_value_keeps_solution(self):
        cnt, sol = solver.solve(G4, 2, 2, banned={(3, 3): {2, 3, 4}})
        self.assertEqual(cnt, 1)
        self.assertEqual(sol, S4)

    def test_banned_ignored_cells_untouched(self):
        # Sans bannis : comportement inchange (defaut None).
        cnt, sol = solver.solve(G4, 2, 2)
        self.assertEqual(cnt, 1)
        self.assertEqual(sol, S4)


if __name__ == "__main__":
    unittest.main()

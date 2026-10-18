"""Solveur sudoku generique : grille N x N, blocs de br lignes x bc colonnes (N = br*bc).

grid : liste de N listes de N entiers (0 = vide, 1..N = valeur).
"""
import time


class Sudoku:
    def __init__(self, grid, br, bc, banned=None):
        """banned : {(r, c): {valeurs interdites}} (ex. chiffres vus en rouge :
        le jeu dit qu'ils ne vont pas a cet emplacement)."""
        self.n = len(grid)
        self.br, self.bc = br, bc
        if br * bc != self.n:
            raise ValueError("bloc %dx%d incompatible avec N=%d" % (br, bc, self.n))
        self.full = (1 << self.n) - 1
        self.g = [row[:] for row in grid]
        n = self.n
        self.rows = [0] * n
        self.cols = [0] * n
        self.boxes = [0] * n
        self.ban = [[0] * n for _ in range(n)]
        if banned:
            for (r, c), vals in banned.items():
                try:
                    if 0 <= r < n and 0 <= c < n:
                        m = 0
                        for v in vals:
                            if 1 <= int(v) <= n:
                                m |= 1 << (int(v) - 1)
                        self.ban[r][c] |= m
                except Exception:
                    pass

    def box(self, r, c):
        return (r // self.br) * (self.n // self.bc) + (c // self.bc)

    def conflicts(self):
        """Renvoie l'ensemble des cases (r,c) en conflit (doublons)."""
        n = self.n
        bad = set()
        groups = []
        for i in range(n):
            groups.append([(i, c) for c in range(n)])
            groups.append([(r, i) for r in range(n)])
        for b0 in range(0, n, self.br):
            for b1 in range(0, n, self.bc):
                groups.append([(r, c) for r in range(b0, b0 + self.br) for c in range(b1, b1 + self.bc)])
        for grp in groups:
            seen = {}
            for (r, c) in grp:
                v = self.g[r][c]
                if v:
                    seen.setdefault(v, []).append((r, c))
            for cells in seen.values():
                if len(cells) > 1:
                    bad.update(cells)
        return bad

    def _init_masks(self):
        n = self.n
        self.rows = [0] * n
        self.cols = [0] * n
        self.boxes = [0] * n
        for r in range(n):
            for c in range(n):
                v = self.g[r][c]
                if v:
                    bit = 1 << (v - 1)
                    b = self.box(r, c)
                    if (self.rows[r] | self.cols[c] | self.boxes[b]) & bit:
                        return False
                    self.rows[r] |= bit
                    self.cols[c] |= bit
                    self.boxes[b] |= bit
        return True

    def solve(self, limit=2, timeout=8.0):
        """Renvoie (nb_solutions_trouvees (<=limit), premiere_solution|None)."""
        if not self._init_masks():
            return 0, None
        n = self.n
        empties = [(r, c) for r in range(n) for c in range(n) if not self.g[r][c]]
        self.sols = []
        self.limit = limit
        self.deadline = time.time() + timeout
        self.nodes = 0
        self._search(empties)
        return len(self.sols), (self.sols[0] if self.sols else None)

    def _search(self, empties):
        if len(self.sols) >= self.limit:
            return
        self.nodes += 1
        if (self.nodes & 1023) == 0 and time.time() > self.deadline:
            return
        if not empties:
            self.sols.append([row[:] for row in self.g])
            return
        # MRV : case avec le moins de candidats (interdits exclus)
        best_i, best_mask, best_cnt = -1, 0, 99
        for i, (r, c) in enumerate(empties):
            m = self.full & ~(self.rows[r] | self.cols[c] | self.boxes[self.box(r, c)])
            m &= ~self.ban[r][c]
            cnt = bin(m).count("1")
            if cnt < best_cnt:
                best_i, best_mask, best_cnt = i, m, cnt
                if cnt <= 1:
                    break
        if best_cnt == 0:
            return
        r, c = empties[best_i]
        rest = empties[:best_i] + empties[best_i + 1:]
        b = self.box(r, c)
        m = best_mask
        while m:
            bit = m & -m
            m ^= bit
            v = bit.bit_length()
            self.g[r][c] = v
            self.rows[r] |= bit
            self.cols[c] |= bit
            self.boxes[b] |= bit
            self._search(rest)
            self.rows[r] ^= bit
            self.cols[c] ^= bit
            self.boxes[b] ^= bit
            self.g[r][c] = 0
            if len(self.sols) >= self.limit:
                return

    def candidates_count(self, solution_grid):
        """Nb de candidats initiaux par case vide (sert a l'ordre 'humain')."""
        self._init_masks()
        out = {}
        for r in range(self.n):
            for c in range(self.n):
                if not self.g[r][c]:
                    m = self.full & ~(self.rows[r] | self.cols[c] | self.boxes[self.box(r, c)])
                    m &= ~self.ban[r][c]
                    out[(r, c)] = bin(m).count("1")
        return out


def solve(grid, br, bc, limit=2, timeout=8.0, banned=None):
    s = Sudoku(grid, br, bc, banned=banned)
    return s.solve(limit=limit, timeout=timeout)


def conflicts(grid, br, bc):
    return Sudoku(grid, br, bc).conflicts()

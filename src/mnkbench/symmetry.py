"""Board symmetries and canonical positions.

A square board has 8 symmetries (the dihedral group D4: 4 rotations, each
optionally mirrored); a rectangular board has 4 (identity, two mirrors and a
half-turn). Gravity breaks top/bottom symmetry, so a gravity board only keeps
the left/right mirror.

Each symmetry is stored as a permutation ``f`` where the stone on cell ``c``
moves to cell ``f[c]``. Tabular agents (MENACE, Q-learning, TD) look positions
up by their *canonical* form, the lexicographically smallest image under all
symmetries. This is how Michie got tic-tac-toe down to about 300 matchboxes.
"""

from __future__ import annotations

from functools import lru_cache
from operator import itemgetter

from .game import Game, GameConfig, get_game


def _transform(m: int, n: int, name: str):
    """Return a function (r, c) -> (r', c') for a named symmetry, or None if
    it does not map an m x n board onto itself."""
    square = m == n
    table = {
        "identity": lambda r, c: (r, c),
        "mirror_lr": lambda r, c: (r, n - 1 - c),
        "mirror_ud": lambda r, c: (m - 1 - r, c),
        "rot180": lambda r, c: (m - 1 - r, n - 1 - c),
    }
    if square:
        table.update(
            {
                "rot90": lambda r, c: (c, n - 1 - r),
                "rot270": lambda r, c: (n - 1 - c, r),
                "transpose": lambda r, c: (c, r),
                "antitranspose": lambda r, c: (n - 1 - c, m - 1 - r),
            }
        )
    return table.get(name)


SYMMETRY_NAMES = (
    "identity",
    "mirror_lr",
    "mirror_ud",
    "rot180",
    "rot90",
    "rot270",
    "transpose",
    "antitranspose",
)


@lru_cache(maxsize=64)
def symmetry_names(config: GameConfig) -> tuple[str, ...]:
    if config.gravity:
        return ("identity", "mirror_lr")
    return tuple(name for name in SYMMETRY_NAMES if _transform(config.m, config.n, name))


@lru_cache(maxsize=64)
def symmetries(config: GameConfig) -> tuple[tuple[int, ...], ...]:
    """All symmetry permutations ``f`` (cell -> image cell) valid for ``config``."""
    m, n = config.m, config.n
    perms = []
    for name in symmetry_names(config):
        t = _transform(m, n, name)
        perm = []
        for cell in range(m * n):
            r, c = divmod(cell, n)
            rr, cc = t(r, c)
            perm.append(rr * n + cc)
        perms.append(tuple(perm))
    return tuple(perms)


def invert(perm: tuple[int, ...]) -> tuple[int, ...]:
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return tuple(inv)


class Canonicalizer:
    """Maps positions to canonical keys and moves between frames.

    ``canonical(cells)`` returns ``(key, sym)``: the canonical encoding and the
    index of the symmetry that produced it. A move ``mv`` on the real board is
    ``to_canon(sym, mv)`` in the canonical frame, and a canonical-frame move
    ``cm`` is ``from_canon(sym, cm)`` on the real board.
    """

    def __init__(self, game: Game | GameConfig, use_symmetry: bool = True):
        if isinstance(game, GameConfig):
            game = get_game(game)
        self.game = game
        self.use_symmetry = use_symmetry
        perms = symmetries(game.config) if use_symmetry else (tuple(range(game.size)),)
        self.perms = perms
        self.inverses = tuple(invert(p) for p in perms)
        # transformed[j] = cells[inv[j]]  -> gather with itemgetter(*inv)
        self._getters = tuple(itemgetter(*inv) for inv in self.inverses) if game.size > 1 else None

    @property
    def num_symmetries(self) -> int:
        return len(self.perms)

    def transformed(self, cells: list[int], sym: int) -> bytes:
        if self._getters is None:
            return bytes(cells)
        return bytes(self._getters[sym](cells))

    def canonical(self, cells: list[int]) -> tuple[bytes, int]:
        if not self.use_symmetry or self._getters is None:
            return bytes(cells), 0
        best, best_sym = None, 0
        for sym, getter in enumerate(self._getters):
            img = bytes(getter(cells))
            if best is None or img < best:
                best, best_sym = img, sym
        return best, best_sym

    def to_canon(self, sym: int, move: int) -> int:
        return self.perms[sym][move]

    def from_canon(self, sym: int, move: int) -> int:
        return self.inverses[sym][move]

    def stabilizer(self, canon_key: bytes) -> list[int]:
        """Symmetries that map the canonical position onto itself."""
        cells = list(canon_key)
        return [s for s in range(len(self.perms)) if self.transformed(cells, s) == canon_key]

    def move_orbit_reps(self, canon_key: bytes, moves: list[int]) -> list[int]:
        """Reduce canonical-frame ``moves`` to one representative per orbit under
        the position's stabilizer (e.g. the empty 3x3 board has only 3 distinct
        opening moves: corner, edge and centre)."""
        stab = self.stabilizer(canon_key)
        if len(stab) <= 1:
            return list(moves)
        reps, seen = [], set()
        for mv in sorted(moves):
            if mv in seen:
                continue
            orbit = {self.perms[s][mv] for s in stab}
            seen |= orbit
            reps.append(min(orbit))
        return reps

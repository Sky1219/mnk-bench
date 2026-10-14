"""The m,n,k game engine.

An (m, n, k) game is played on a board with ``m`` rows and ``n`` columns. Two
players (X moves first, then O) take turns placing a stone on an empty cell,
and the first to get ``k`` stones in a row (horizontally, vertically or
diagonally) wins. If the board fills up first, the game is a draw.

Two rule flags cover common variants:

* ``gravity``: stones fall to the lowest empty cell of their column, as in
  Connect Four (which is the (6, 7, 4) game with gravity).
* ``exact_k``: only runs of *exactly* k stones win; longer runs ("overlines")
  do not, as in some gomoku rule sets.

Cells are numbered row-major: cell ``r * n + c`` is row ``r`` (0 = top), column
``c`` (0 = left). Moves are always cell indices, also under gravity, so every
agent sees the same move format. The human-readable name of a cell is its
column letter followed by its 1-based row number, e.g. ``"b3"``.

:class:`Game` holds the rules and precomputed lookup tables and is shared by
every :class:`State` of that game. :class:`State` is mutable and supports fast
``push``/``pop`` for search, plus ``play`` for an immutable-style copy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

EMPTY, X, O = 0, 1, 2
DRAW = 0  # value of State.winner for a drawn game (None while the game is running)
SYMBOLS = {EMPTY: ".", X: "X", O: "O"}
COLUMN_LETTERS = "abcdefghijklmnopqrstuvwxyz"

# (dr, dc) for the four line axes: horizontal, vertical, diagonal, anti-diagonal.
AXES = ((0, 1), (1, 0), (1, 1), (1, -1))


def other(player: int) -> int:
    """Return the opponent of ``player``."""
    return 3 - player


@dataclass(frozen=True)
class GameConfig:
    """The rules of one m,n,k game. Hashable, so it can key caches and results."""

    m: int
    n: int
    k: int
    gravity: bool = False
    exact_k: bool = False

    def __post_init__(self) -> None:
        if self.m < 1 or self.n < 1:
            raise ValueError(f"board must be at least 1x1, got {self.m}x{self.n}")
        if self.n > len(COLUMN_LETTERS):
            raise ValueError(f"at most {len(COLUMN_LETTERS)} columns are supported")
        if self.k < 1:
            raise ValueError(f"k must be positive, got {self.k}")

    @property
    def size(self) -> int:
        return self.m * self.n

    @property
    def label(self) -> str:
        """Short label such as ``(3,3,3)`` or ``(6,7,4)+gravity``."""
        flags = []
        if self.gravity:
            flags.append("gravity")
        if self.exact_k:
            flags.append("exact")
        return f"({self.m},{self.n},{self.k})" + "".join("+" + f for f in flags)

    @classmethod
    def parse(cls, text: str) -> GameConfig:
        """Parse ``"3,3,3"``, ``"(6,7,4)+gravity"``, ``"15x15x5 exact"`` and similar."""
        nums = [int(x) for x in re.findall(r"\d+", text)]
        if len(nums) != 3:
            raise ValueError(f"expected three integers m,n,k in {text!r}")
        lower = text.lower()
        return cls(
            *nums,
            gravity="gravity" in lower or "connect" in lower,
            exact_k="exact" in lower,
        )

    def to_dict(self) -> dict:
        return {"m": self.m, "n": self.n, "k": self.k, "gravity": self.gravity, "exact_k": self.exact_k}


# Well-known games.
TIC_TAC_TOE = GameConfig(3, 3, 3)
CONNECT_FOUR = GameConfig(6, 7, 4, gravity=True)
GOMOKU = GameConfig(15, 15, 5)
GOMOKU_EXACT = GameConfig(15, 15, 5, exact_k=True)


class Game:
    """Rules plus precomputed tables (rays, windows, neighbours) for one config.

    Use :func:`get_game` to obtain a cached instance.
    """

    def __init__(self, config: GameConfig):
        self.config = config
        self.m, self.n, self.k = config.m, config.n, config.k
        self.size = config.size
        self.gravity = config.gravity
        self.exact_k = config.exact_k
        m, n, k = self.m, self.n, self.k

        # rays[cell][axis] = (forward cells, backward cells), each up to k long.
        self.rays: list[tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]] = []
        for cell in range(self.size):
            r, c = divmod(cell, n)
            per_axis = []
            for dr, dc in AXES:
                fwd, bwd = [], []
                for step in range(1, k + 1):
                    rr, cc = r + dr * step, c + dc * step
                    if 0 <= rr < m and 0 <= cc < n:
                        fwd.append(rr * n + cc)
                    else:
                        break
                for step in range(1, k + 1):
                    rr, cc = r - dr * step, c - dc * step
                    if 0 <= rr < m and 0 <= cc < n:
                        bwd.append(rr * n + cc)
                    else:
                        break
                per_axis.append((tuple(fwd), tuple(bwd)))
            self.rays.append(tuple(per_axis))

        # All k-length windows (lines of k cells) and, for each cell, the
        # windows that contain it. Used by evaluation functions.
        windows: list[tuple[int, ...]] = []
        for r in range(m):
            for c in range(n):
                for dr, dc in AXES:
                    er, ec = r + dr * (k - 1), c + dc * (k - 1)
                    if 0 <= er < m and 0 <= ec < n:
                        windows.append(tuple((r + dr * i) * n + (c + dc * i) for i in range(k)))
        self.windows: tuple[tuple[int, ...], ...] = tuple(windows)
        cell_windows: list[list[int]] = [[] for _ in range(self.size)]
        for wi, w in enumerate(windows):
            for cell in w:
                cell_windows[cell].append(wi)
        self.cell_windows: tuple[tuple[int, ...], ...] = tuple(tuple(x) for x in cell_windows)

        # Neighbours within Chebyshev distance 1 and 2 (for candidate pruning).
        self.neighbors1 = tuple(self._neighbors(cell, 1) for cell in range(self.size))
        self.neighbors2 = tuple(self._neighbors(cell, 2) for cell in range(self.size))

        # Cells ordered from the centre outwards: a good default move ordering.
        cr, cc = (m - 1) / 2, (n - 1) / 2
        self.center_order = tuple(
            sorted(range(self.size), key=lambda i: (abs(i // n - cr) + abs(i % n - cc), i))
        )

    def _neighbors(self, cell: int, radius: int) -> tuple[int, ...]:
        r, c = divmod(cell, self.n)
        out = []
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.m and 0 <= cc < self.n:
                    out.append(rr * self.n + cc)
        return tuple(out)

    # ------------------------------------------------------------------ helpers
    def new_state(self) -> State:
        return State(self)

    def cell_name(self, cell: int) -> str:
        r, c = divmod(cell, self.n)
        return f"{COLUMN_LETTERS[c]}{r + 1}"

    def parse_cell(self, text: str) -> int:
        """Parse a cell name like ``"b3"`` (column letter, 1-based row)."""
        match = re.fullmatch(r"\s*([a-zA-Z])\s*(\d+)\s*", text)
        if not match:
            raise ValueError(f"cannot parse cell {text!r}")
        c = COLUMN_LETTERS.index(match.group(1).lower())
        r = int(match.group(2)) - 1
        if not (0 <= r < self.m and 0 <= c < self.n):
            raise ValueError(f"cell {text!r} is off the board")
        return r * self.n + c

    def __repr__(self) -> str:
        return f"Game{self.config.label}"


@lru_cache(maxsize=64)
def get_game(config: GameConfig) -> Game:
    """Return the shared :class:`Game` for ``config``."""
    return Game(config)


@dataclass
class State:
    """A mutable game position.

    ``winner`` is ``None`` while the game is in progress, ``X`` or ``O`` once
    someone has won, and ``DRAW`` (0) for a full board with no winner.
    """

    game: Game
    cells: list[int] = field(default_factory=list)
    to_move: int = X
    history: list[int] = field(default_factory=list)
    winner: int | None = None
    col_top: list[int] = field(default_factory=list)  # gravity: next free row per column

    def __post_init__(self) -> None:
        if not self.cells:
            self.cells = [EMPTY] * self.game.size
        if not self.col_top:
            self.col_top = [self.game.m - 1] * self.game.n

    # --------------------------------------------------------------- creation
    @classmethod
    def from_string(cls, game: Game | GameConfig, text: str) -> State:
        """Build a position from a diagram such as ``"X.O/.X./..."``.

        Rows are separated by ``/`` or newlines; ``X``/``O`` are stones and
        ``.``, ``-`` or ``_`` are empty. The side to move is inferred from the
        stone counts. The move history is synthetic (X stones then O stones,
        interleaved) and the winner is computed from the final position.
        """
        if isinstance(game, GameConfig):
            game = get_game(game)
        rows = [r.strip() for r in re.split(r"[/\n]", text.strip()) if r.strip()]
        rows = [r.replace(" ", "") for r in rows]
        if len(rows) != game.m or any(len(r) != game.n for r in rows):
            raise ValueError(f"diagram does not match a {game.m}x{game.n} board")
        cells = []
        for row in rows:
            for ch in row.upper():
                cells.append({"X": X, "O": O}.get(ch, EMPTY))
        xs = [i for i, v in enumerate(cells) if v == X]
        os_ = [i for i, v in enumerate(cells) if v == O]
        if not (len(xs) == len(os_) or len(xs) == len(os_) + 1):
            raise ValueError("X must have the same number of stones as O, or one more")
        state = cls(game)
        state.cells = cells
        state.to_move = X if len(xs) == len(os_) else O
        history = []
        for i in range(len(xs)):
            history.append(xs[i])
            if i < len(os_):
                history.append(os_[i])
        state.history = history
        if game.gravity:
            for c in range(game.n):
                top = game.m - 1
                while top >= 0 and cells[top * game.n + c] != EMPTY:
                    top -= 1
                state.col_top[c] = top
        state.winner = state._compute_winner()
        return state

    def copy(self) -> State:
        return State(
            self.game,
            self.cells[:],
            self.to_move,
            self.history[:],
            self.winner,
            self.col_top[:],
        )

    # ---------------------------------------------------------------- queries
    @property
    def config(self) -> GameConfig:
        return self.game.config

    @property
    def move_count(self) -> int:
        return len(self.history)

    @property
    def empty_count(self) -> int:
        return self.game.size - len(self.history)

    @property
    def is_terminal(self) -> bool:
        return self.winner is not None

    @property
    def last_move(self) -> int | None:
        return self.history[-1] if self.history else None

    def key(self) -> bytes:
        """A compact, hashable encoding of the position (not of the history)."""
        return bytes(self.cells)

    def legal_moves(self) -> list[int]:
        if self.winner is not None:
            return []
        if self.game.gravity:
            n = self.game.n
            return [top * n + c for c, top in enumerate(self.col_top) if top >= 0]
        cells = self.cells
        return [i for i in range(self.game.size) if cells[i] == EMPTY]

    def is_legal(self, move: int) -> bool:
        if self.winner is not None or not (0 <= move < self.game.size):
            return False
        if self.cells[move] != EMPTY:
            return False
        if self.game.gravity:
            return move // self.game.n == self.col_top[move % self.game.n]
        return True

    def run_length(self, cell: int, player: int, axis: int) -> int:
        """Length of ``player``'s run through ``cell`` along ``axis``, counting
        ``cell`` itself as ``player``'s stone."""
        cells = self.cells
        fwd, bwd = self.game.rays[cell][axis]
        count = 1
        for c in fwd:
            if cells[c] != player:
                break
            count += 1
        for c in bwd:
            if cells[c] != player:
                break
            count += 1
        return count

    def would_win(self, cell: int, player: int) -> bool:
        """Would ``player`` win by having a stone on ``cell``? (Ignores legality.)"""
        k = self.game.k
        exact = self.game.exact_k
        cells = self.cells
        for fwd, bwd in self.game.rays[cell]:
            count = 1
            for c in fwd:
                if cells[c] != player:
                    break
                count += 1
            for c in bwd:
                if cells[c] != player:
                    break
                count += 1
            if count == k or (count > k and not exact):
                return True
        return False

    def winning_moves(self, player: int) -> list[int]:
        """Legal moves that would immediately win for ``player`` (ignoring whose turn it is)."""
        if self.winner is not None:
            return []
        return [mv for mv in self.legal_moves() if self.would_win(mv, player)]

    # -------------------------------------------------------------- mutation
    def push(self, move: int) -> None:
        """Play ``move`` in place. Assumes it is legal (see :meth:`play`)."""
        player = self.to_move
        game = self.game
        self.cells[move] = player
        self.history.append(move)
        if game.gravity:
            self.col_top[move % game.n] -= 1
        if self.would_win(move, player):
            self.winner = player
        elif len(self.history) == game.size:
            self.winner = DRAW
        self.to_move = 3 - player

    def pop(self) -> int:
        """Undo the last move and return it."""
        move = self.history.pop()
        self.cells[move] = EMPTY
        if self.game.gravity:
            self.col_top[move % self.game.n] += 1
        self.winner = None
        self.to_move = 3 - self.to_move
        return move

    def play(self, move: int) -> State:
        """Return a new state with ``move`` played, after checking legality."""
        if not self.is_legal(move):
            raise IllegalMoveError(f"illegal move {move} in\n{self.render()}")
        child = self.copy()
        child.push(move)
        return child

    # --------------------------------------------------------------- display
    def _compute_winner(self) -> int | None:
        k = self.game.k
        exact = self.game.exact_k
        for cell, v in enumerate(self.cells):
            if v == EMPTY:
                continue
            for axis in range(4):
                # Only count runs starting at their first stone.
                bwd = self.game.rays[cell][axis][1]
                if bwd and self.cells[bwd[0]] == v:
                    continue
                length = self.run_length(cell, v, axis)
                if length == k or (length > k and not exact):
                    return v
        if all(v != EMPTY for v in self.cells):
            return DRAW
        return None

    def render(self, coords: bool = True) -> str:
        """ASCII diagram of the board, optionally with column letters and row numbers."""
        n = self.game.n
        lines = []
        width = len(str(self.game.m))
        if coords:
            lines.append(" " * (width + 1) + " ".join(COLUMN_LETTERS[:n]))
        for r in range(self.game.m):
            row = " ".join(SYMBOLS[v] for v in self.cells[r * n : (r + 1) * n])
            lines.append(f"{r + 1:>{width}} {row}" if coords else row)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render(coords=False)


class IllegalMoveError(ValueError):
    """Raised when an agent or caller tries to play an illegal move."""

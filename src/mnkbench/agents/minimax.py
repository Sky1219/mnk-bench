"""Exact search: negamax with alpha-beta pruning and a transposition table.

This plays perfectly and doubles as the *oracle* for the optimal-move-rate
metric. It solves small boards outright and refuses large ones, because the
number of positions it must visit grows exponentially with the board.

Scores are from the point of view of the side to move: a win is worth
``1 + (empty cells left after the winning move)``, so faster wins score
higher and slower losses score higher, a draw is 0.
"""

from __future__ import annotations

import time

from ..complexity import position_bound
from ..game import DRAW, Game, GameConfig, State, get_game
from ..symmetry import Canonicalizer
from .base import Agent, BudgetExceeded, CostEstimate

EXACT, LOWER, UPPER = 0, 1, 2

# Calibrated on this machine class (see tests/test_feasibility.py and
# docs/results/feasibility): search visits roughly position_bound ** 0.68
# nodes with symmetry, at ~12us each, and stores ~250 bytes per TT entry.
NODE_EXPONENT = 0.68
SECONDS_PER_NODE = 1.2e-5
BYTES_PER_ENTRY = 250


class Solver:
    """Alpha-beta solver with a persistent transposition table."""

    def __init__(
        self,
        game: Game | GameConfig,
        use_symmetry: bool = True,
        max_nodes: int | None = None,
        time_limit: float | None = None,
    ):
        self.game = game if isinstance(game, Game) else get_game(game)
        self.canon = Canonicalizer(self.game, use_symmetry)
        self.tt: dict[bytes, tuple[int, int]] = {}
        self.nodes = 0
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self._deadline: float | None = None
        size = self.game.size
        self.INF = size + 2
        if self.game.gravity:
            n = self.game.n
            centre = (n - 1) / 2
            self._rank = [abs(cell % n - centre) for cell in range(size)]
        else:
            rank = [0] * size
            for i, cell in enumerate(self.game.center_order):
                rank[cell] = i
            self._rank = rank

    # -------------------------------------------------------------- search
    def _check_budget(self) -> None:
        if self.max_nodes is not None and self.nodes > self.max_nodes:
            raise BudgetExceeded(f"solver exceeded {self.max_nodes} nodes")
        if self._deadline is not None and time.perf_counter() > self._deadline:
            raise BudgetExceeded(f"solver exceeded {self.time_limit}s")

    def _search(self, state: State, alpha: int, beta: int) -> int:
        self.nodes += 1
        if not self.nodes & 1023:
            self._check_budget()
        empty = state.empty_count
        player = state.to_move
        opp = 3 - player
        moves = state.legal_moves()
        would_win = state.would_win
        for mv in moves:
            if would_win(mv, player):
                return empty
        threats = [mv for mv in moves if would_win(mv, opp)]
        if len(threats) >= 2:
            return -(empty - 1)

        key, _ = self.canon.canonical(state.cells)
        entry = self.tt.get(key)
        if entry is not None:
            val, flag = entry
            if flag == EXACT:
                return val
            if flag == LOWER and val >= beta:
                return val
            if flag == UPPER and val <= alpha:
                return val

        # Mate-distance bounds: we cannot win this move, so the best possible
        # is winning on our next move; the worst is losing on the opponent's.
        hi, lo = empty - 2, -(empty - 1)
        if beta > hi:
            beta = hi
        if alpha < lo:
            alpha = lo
        if alpha >= beta:
            return alpha

        if threats:
            moves = threats
        else:
            rank = self._rank
            moves.sort(key=rank.__getitem__)

        a0 = alpha
        best = -self.INF
        for mv in moves:
            state.push(mv)
            v = 0 if state.winner == DRAW else -self._search(state, -beta, -alpha)
            state.pop()
            if v > best:
                best = v
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = UPPER if best <= a0 else LOWER if best >= beta else EXACT
        self.tt[key] = (best, flag)
        return best

    def _run(self, state: State, alpha: int, beta: int) -> int:
        if state.winner is not None:
            if state.winner == DRAW:
                return 0
            return -(state.empty_count + 1)
        self._deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        work = state.copy()
        return self._search(work, alpha, beta)

    # ----------------------------------------------------------------- API
    def value(self, state: State) -> int:
        """Exact score of ``state`` for the side to move."""
        return self._run(state, -self.INF, self.INF)

    def outcome(self, state: State) -> int:
        """Game-theoretic outcome for the side to move: +1 win, 0 draw, -1 loss."""
        v = self._run(state, -1, 1)
        return (v > 0) - (v < 0)

    def move_scores(self, state: State) -> dict[int, int]:
        """Exact score of every legal move (from the mover's point of view)."""
        scores = {}
        work = state.copy()
        for mv in state.legal_moves():
            work.push(mv)
            if work.winner == DRAW:
                scores[mv] = 0
            elif work.winner is not None:
                scores[mv] = work.empty_count + 1
            else:
                scores[mv] = -self._run(work, -self.INF, self.INF)
            work.pop()
        return scores

    def best_moves(self, state: State) -> list[int]:
        """All moves achieving the exact best score (fastest win / slowest loss)."""
        v = self.value(state)
        best = []
        work = state.copy()
        for mv in state.legal_moves():
            work.push(mv)
            if work.winner == DRAW:
                s_ok = v <= 0 <= v
            elif work.winner is not None:
                s_ok = work.empty_count + 1 >= v
            else:
                # Null-window test: is the child's score >= v?
                s_ok = -self._run(work, -v, -v + 1) >= v
            work.pop()
            if s_ok:
                best.append(mv)
        return best

    def outcome_preserving_moves(self, state: State) -> list[int]:
        """Moves that keep the game-theoretic outcome (win stays win, draw stays
        draw). This is the definition of "optimal" for the optimal-move rate.
        In a lost position every move is optimal."""
        o = self.outcome(state)
        moves = state.legal_moves()
        if o < 0:
            return moves
        good = []
        work = state.copy()
        for mv in moves:
            work.push(mv)
            if work.winner == DRAW:
                child = 0
            elif work.winner is not None:
                child = 1
            else:
                child = -self.outcome(work)
            work.pop()
            if child >= o:
                good.append(mv)
        return good

    def is_optimal(self, state: State, move: int) -> bool:
        return move in self.outcome_preserving_moves(state)

    def memory_bytes(self) -> float:
        return len(self.tt) * (BYTES_PER_ENTRY + self.game.size)


_SOLVERS: dict[GameConfig, Solver] = {}


def shared_solver(config: GameConfig) -> Solver:
    """A process-wide solver per config, so evaluation reuses its table."""
    if config not in _SOLVERS:
        _SOLVERS[config] = Solver(config)
    return _SOLVERS[config]


def estimate_nodes(config: GameConfig, use_symmetry: bool = True) -> float:
    bound = position_bound(config)
    nodes = bound**NODE_EXPONENT
    if not use_symmetry:
        nodes *= 3
    return max(nodes, 10.0)


class MinimaxAgent(Agent):
    """Perfect play by full alpha-beta search (small boards only).

    ``tiebreak="random"`` picks uniformly among equally good moves (seeded),
    which makes the opponent less predictable for learners; ``"first"`` is
    deterministic. ``max_nodes``/``time_limit`` make the search raise
    :class:`BudgetExceeded` instead of running forever.
    """

    name = "minimax"

    def __init__(
        self,
        seed: int | None = None,
        use_symmetry: bool = True,
        tiebreak: str = "random",
        max_nodes: int | None = None,
        time_limit: float | None = None,
    ):
        super().__init__(seed)
        self.use_symmetry = use_symmetry
        self.tiebreak = tiebreak
        self.max_nodes = max_nodes
        self.time_limit = time_limit
        self._solver: Solver | None = None

    def params(self) -> dict:
        return {
            "use_symmetry": self.use_symmetry,
            "tiebreak": self.tiebreak,
            "max_nodes": self.max_nodes,
            "time_limit": self.time_limit,
        }

    def solver_for(self, game: Game) -> Solver:
        if self._solver is None or self._solver.game.config != game.config:
            if self.use_symmetry and self.max_nodes is None and self.time_limit is None:
                self._solver = shared_solver(game.config)
            else:
                self._solver = Solver(game, self.use_symmetry, self.max_nodes, self.time_limit)
        return self._solver

    def select_move(self, state: State) -> int:
        solver = self.solver_for(state.game)
        before = solver.nodes
        moves = solver.best_moves(state)
        self.add_stat("nodes", solver.nodes - before)
        if self.tiebreak == "first":
            return min(moves, key=solver._rank.__getitem__)
        return self.rng.choice(moves)

    def memory_bytes(self) -> float:
        return self._solver.memory_bytes() if self._solver else 0.0

    def apply_budget(self, budget) -> None:
        # The first move solves the whole game, so allow a few move-budgets
        # for it; the node cap keeps the transposition table within memory.
        self.time_limit = self.time_limit or 3 * budget.time_per_move
        self.max_nodes = self.max_nodes or int(budget.memory_bytes / BYTES_PER_ENTRY)
        self._solver = None

    def estimate(self, config: GameConfig) -> CostEstimate:
        nodes = estimate_nodes(config, self.use_symmetry)
        per_node = SECONDS_PER_NODE * (1 + config.size / 40)
        return CostEstimate(
            time_per_move=nodes * per_node,
            memory_bytes=nodes * (BYTES_PER_ENTRY + config.size),
            notes=f"~{nodes:.2g} nodes to solve the opening position",
        )

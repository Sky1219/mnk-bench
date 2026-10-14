"""Heuristic search: depth-limited alpha-beta with a threat-based evaluation.

Unlike :mod:`minimax`, this never searches to the end of the game. It looks
``depth`` plies ahead and scores the leaves with a *threat* evaluation: every
k-cell window that only one player occupies is a potential line, worth
``base ** stones`` (so three in an open window of four is worth far more than
two). The score is kept up to date incrementally as stones are placed and
removed, so evaluating a leaf is O(1).

On top of that the search knows the tactical rules that decide m,n,k games:
take an immediate win, block the opponent's immediate win (all other moves
are pruned), and treat two simultaneous opponent threats as a loss.

To keep the branching factor manageable on big boards it only considers
empty cells near existing stones (``radius``) and, at each node, the
``max_candidates`` most promising of them by a cheap static score. With
``time_limit`` it deepens iteratively and plays the best move of the deepest
completed iteration, so it degrades gracefully instead of overrunning.
"""

from __future__ import annotations

import time

from ..game import DRAW, EMPTY, GameConfig, State
from .base import Agent, Budget, CostEstimate

WIN = 1_000_000_000
EXACT, LOWER, UPPER = 0, 1, 2


class _Timeout(Exception):
    pass


class _Board:
    """Search-side position with incremental window counts and evaluation."""

    def __init__(self, state: State, base: float, radius: int):
        game = state.game
        self.game = game
        self.state = state.copy()
        k = game.k
        self.weights = [0.0] + [float(base) ** i for i in range(1, k + 1)]
        nw = len(game.windows)
        self.wx = [0] * nw
        self.wo = [0] * nw
        self.sx = 0.0  # value of X's open lines
        self.so = 0.0  # value of O's open lines
        self.near = [0] * game.size
        self.neigh = game.neighbors1 if radius <= 1 else game.neighbors2
        cells = self.state.cells
        for wi, window in enumerate(game.windows):
            for c in window:
                if cells[c] == 1:
                    self.wx[wi] += 1
                elif cells[c] == 2:
                    self.wo[wi] += 1
            x, o = self.wx[wi], self.wo[wi]
            if o == 0:
                self.sx += self.weights[x]
            elif x == 0:
                self.so += self.weights[o]
        for mv in self.state.history:
            for c in self.neigh[mv]:
                self.near[c] += 1

    def push(self, mv: int) -> None:
        player = self.state.to_move
        weights = self.weights
        game = self.game
        if player == 1:
            own, opp = self.wx, self.wo
        else:
            own, opp = self.wo, self.wx
        gain = loss = 0.0
        for wi in game.cell_windows[mv]:
            o = opp[wi]
            if o == 0:
                c = own[wi]
                gain += weights[c + 1] - weights[c]
            elif own[wi] == 0:
                loss += weights[o]  # the opponent's line is now blocked
            own[wi] += 1
        if player == 1:
            self.sx += gain
            self.so -= loss
        else:
            self.so += gain
            self.sx -= loss
        for c in self.neigh[mv]:
            self.near[c] += 1
        self.state.push(mv)

    def pop(self) -> None:
        mv = self.state.pop()
        player = self.state.to_move
        weights = self.weights
        if player == 1:
            own, opp = self.wx, self.wo
        else:
            own, opp = self.wo, self.wx
        gain = loss = 0.0
        for wi in self.game.cell_windows[mv]:
            own[wi] -= 1
            o = opp[wi]
            if o == 0:
                c = own[wi]
                gain += weights[c + 1] - weights[c]
            elif own[wi] == 0:
                loss += weights[o]
        if player == 1:
            self.sx -= gain
            self.so += loss
        else:
            self.so -= gain
            self.sx += loss
        for c in self.neigh[mv]:
            self.near[c] -= 1

    def evaluate(self, tempo: float) -> float:
        """Static score for the side to move; its own lines count ``tempo`` x."""
        if self.state.to_move == 1:
            return tempo * self.sx - self.so
        return tempo * self.so - self.sx

    def candidates(self) -> list[int]:
        st = self.state
        if self.game.gravity or not st.history:
            moves = st.legal_moves()
            if not st.history and not self.game.gravity:
                return list(self.game.center_order[:1])
            return moves
        cells, near = st.cells, self.near
        return [c for c in range(self.game.size) if near[c] and cells[c] == EMPTY]

    def move_score(self, mv: int, player: int) -> float:
        """Cheap static ordering score: line value gained plus value denied."""
        weights = self.weights
        own_c = self.wx if player == 1 else self.wo
        opp_c = self.wo if player == 1 else self.wx
        s = 0.0
        for wi in self.game.cell_windows[mv]:
            own, opp = own_c[wi], opp_c[wi]
            if opp == 0:
                s += weights[own + 1]
            if own == 0:
                s += 0.9 * weights[opp + 1]
        return s


class AlphaBetaAgent(Agent):
    name = "alphabeta"

    def __init__(
        self,
        seed: int | None = None,
        depth: int = 4,
        time_limit: float | None = None,
        max_candidates: int = 12,
        radius: int = 2,
        base: float = 8.0,
        tempo: float = 2.0,
    ):
        super().__init__(seed)
        self.tempo = tempo
        self.depth = depth
        self.time_limit = time_limit
        self.max_candidates = max_candidates
        self.radius = radius
        self.base = base
        self.tt: dict[bytes, tuple[int, float, int, int]] = {}
        self.nodes = 0
        self._deadline: float | None = None

    def params(self) -> dict:
        return {"depth": self.depth, "time_limit": self.time_limit, "max_candidates": self.max_candidates,
                "radius": self.radius, "base": self.base, "tempo": self.tempo}

    # ----------------------------------------------------------------- search
    def _ordered(self, b: _Board, moves: list[int], player: int, tt_move: int | None, limit: bool) -> list[int]:
        scored = sorted(moves, key=lambda mv: -b.move_score(mv, player))
        if tt_move is not None and tt_move in moves:
            scored.remove(tt_move)
            scored.insert(0, tt_move)
        if limit and len(scored) > self.max_candidates:
            scored = scored[: self.max_candidates]
        return scored

    def _search(self, b: _Board, depth: int, alpha: float, beta: float, ply: int) -> float:
        self.nodes += 1
        if self._deadline is not None and not self.nodes & 255 and time.perf_counter() > self._deadline:
            raise _Timeout
        st = b.state
        player = st.to_move
        opp = 3 - player
        moves = b.candidates()
        would_win = st.would_win
        for mv in moves:
            if would_win(mv, player):
                return WIN - ply
        threats = [mv for mv in moves if would_win(mv, opp)]
        if len(threats) >= 2:
            return -(WIN - ply - 1)
        if depth <= 0:
            return b.evaluate(self.tempo)

        key = bytes(st.cells)
        entry = self.tt.get(key)
        tt_move = None
        if entry is not None:
            e_depth, e_val, e_flag, tt_move = entry
            if e_depth >= depth:
                if e_flag == EXACT:
                    return e_val
                if e_flag == LOWER and e_val >= beta:
                    return e_val
                if e_flag == UPPER and e_val <= alpha:
                    return e_val

        if threats:
            moves = threats
        else:
            moves = self._ordered(b, moves, player, tt_move, limit=True)
        a0 = alpha
        best, best_mv = -WIN * 2.0, moves[0]
        for mv in moves:
            b.push(mv)
            if st.winner == DRAW:
                v = 0.0
            else:
                v = -self._search(b, depth - 1, -beta, -alpha, ply + 1)
            b.pop()
            if v > best:
                best, best_mv = v, mv
                if v > alpha:
                    alpha = v
                    if alpha >= beta:
                        break
        flag = UPPER if best <= a0 else LOWER if best >= beta else EXACT
        self.tt[key] = (depth, best, flag, best_mv)
        return best

    def _root(self, b: _Board, depth: int, order: list[int]) -> tuple[int, float, list[tuple[float, int]]]:
        alpha, beta = -WIN * 2.0, WIN * 2.0
        best_mv, best = order[0], -WIN * 2.0
        scores = []
        for mv in order:
            b.push(mv)
            if b.state.winner == DRAW:
                v = 0.0
            elif b.state.winner is not None:
                v = float(WIN)
            else:
                v = -self._search(b, depth - 1, -beta, -alpha, 1)
            b.pop()
            scores.append((v, mv))
            if v > best:
                best, best_mv = v, mv
                alpha = max(alpha, v)
        return best_mv, best, scores

    def select_move(self, state: State) -> int:
        b = _Board(state, self.base, self.radius)
        player = state.to_move
        opp = 3 - player
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]
        wins = [mv for mv in legal if state.would_win(mv, player)]
        if wins:
            return self.rng.choice(wins)
        blocks = [mv for mv in legal if state.would_win(mv, opp)]
        if len(blocks) == 1:
            return blocks[0]
        if not state.history and not state.game.gravity:
            return state.game.center_order[0]

        self.tt.clear()
        self.nodes = 0
        moves = blocks or b.candidates()
        self.rng.shuffle(moves)
        order = self._ordered(b, moves, player, None, limit=False)
        if len(order) > 2 * self.max_candidates:
            order = order[: 2 * self.max_candidates]
        self._deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        best_mv = order[0]
        reached = 0
        try:
            for d in range(1, self.depth + 1):
                mv, val, scores = self._root(b, d, order)
                best_mv = mv
                reached = d
                if abs(val) >= WIN - 100:
                    break  # forced win or loss found
                # Search the best move first in the next iteration.
                order = [m for _, m in sorted(scores, key=lambda t: -t[0])]
        except _Timeout:
            pass
        self.add_stat("nodes", self.nodes)
        self.add_stat("depth_reached", reached)
        self.add_stat("moves", 1)
        return best_mv

    def memory_bytes(self) -> float:
        return len(self.tt) * 300.0

    # ------------------------------------------------------------ feasibility
    @staticmethod
    def _per_node(config: GameConfig) -> float:
        return 2.5e-5 + 1.2e-7 * config.size + 4e-7 * config.k * config.k

    def _nodes(self, config: GameConfig, depth: int) -> float:
        b = min(self.max_candidates, config.n if config.gravity else self.max_candidates, config.size)
        b_eff = max(1.5, b**0.6)
        return sum(b_eff**d for d in range(1, depth + 1)) * min(2 * self.max_candidates, config.size)

    def estimate(self, config: GameConfig) -> CostEstimate:
        t = self._nodes(config, self.depth) * self._per_node(config)
        if self.time_limit:
            t = min(t, self.time_limit)
        return CostEstimate(time_per_move=t, memory_bytes=self._nodes(config, self.depth) * 300 + 2e4,
                            notes=f"depth {self.depth}")

    def degrade(self, config: GameConfig, budget: Budget):
        for d in range(self.depth - 1, 0, -1):
            t = self._nodes(config, d) * self._per_node(config)
            if t <= budget.time_per_move:
                est = CostEstimate(time_per_move=t, memory_bytes=self._nodes(config, d) * 300, notes=f"depth {d}")
                return {"depth": d}, est
        return None


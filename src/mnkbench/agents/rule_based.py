"""A hand-written rule-based player, generalised from the classic
tic-tac-toe strategy (Newell & Simon, 1972) to any m,n,k board.

In priority order:

1. Win: complete k in a row if possible.
2. Block: stop the opponent's immediate win.
3. Fork: make a move that creates two or more immediate winning threats.
4. Block fork: occupy the cell where the opponent could fork.
5. Otherwise play the cell with the best "line potential": the sum, over
   every k-window through the cell that is still open, of a weight growing
   with the number of stones already in it (own lines count more than the
   opponent's). Ties are broken randomly (seeded).

Only cells near existing stones are considered on big boards, which keeps
the cost per move roughly linear in the number of stones.
"""

from __future__ import annotations

from ..game import EMPTY, GameConfig, State
from .base import Agent, CostEstimate


def threat_cells(state: State, cell: int, player: int) -> set[int]:
    """Cells where ``player`` would win next move, after placing a stone on
    ``cell`` (which must be empty; the stone is placed and removed)."""
    game = state.game
    cells = state.cells
    k = game.k
    found: set[int] = set()
    cells[cell] = player
    try:
        for wi in game.cell_windows[cell]:
            window = game.windows[wi]
            own = empties = 0
            gap = -1
            for c in window:
                v = cells[c]
                if v == player:
                    own += 1
                elif v == EMPTY:
                    empties += 1
                    gap = c
                else:
                    break
            else:
                if own == k - 1 and empties == 1 and gap not in found:
                    if state.would_win(gap, player) and _playable_after(state, gap, cell):
                        found.add(gap)
    finally:
        cells[cell] = EMPTY
    return found


def _playable_after(state: State, gap: int, placed: int) -> bool:
    """Under gravity a threat only counts if ``gap`` is directly playable once
    a stone is on ``placed``."""
    game = state.game
    if not game.gravity:
        return True
    col = gap % game.n
    top = state.col_top[col] - (1 if placed % game.n == col else 0)
    return gap // game.n == top


class RuleBasedAgent(Agent):
    name = "rule"

    def __init__(self, seed: int | None = None, use_forks: bool = True, radius: int = 2):
        super().__init__(seed)
        self.use_forks = use_forks
        self.radius = radius

    def params(self) -> dict:
        return {"use_forks": self.use_forks, "radius": self.radius}

    def candidates(self, state: State) -> list[int]:
        moves = state.legal_moves()
        game = state.game
        if game.gravity or game.size <= 25 or not state.history:
            return moves
        near = game.neighbors1 if self.radius <= 1 else game.neighbors2
        cells = state.cells
        cand = {c for mv in state.history for c in near[mv] if cells[c] == EMPTY}
        return sorted(cand) if cand else moves

    def select_move(self, state: State) -> int:
        me = state.to_move
        opp = 3 - me
        moves = state.legal_moves()
        rng = self.rng

        wins = [mv for mv in moves if state.would_win(mv, me)]
        if wins:
            return rng.choice(wins)
        blocks = [mv for mv in moves if state.would_win(mv, opp)]
        if blocks:
            return rng.choice(blocks)

        cands = self.candidates(state)
        if self.use_forks and state.game.k >= 2:
            forks = [mv for mv in cands if len(threat_cells(state, mv, me)) >= 2]
            if forks:
                return self._best_by_potential(state, forks, me)
            opp_forks = [mv for mv in cands if len(threat_cells(state, mv, opp)) >= 2]
            if len(opp_forks) == 1:
                return opp_forks[0]
            if opp_forks:
                # Several fork points: prefer making our own threat that forces
                # a reply, else take the fork point with the best potential.
                forcing = [mv for mv in cands if threat_cells(state, mv, me)]
                safe = [mv for mv in forcing if not (threat_cells(state, mv, me) & set(opp_forks))]
                if safe:
                    return self._best_by_potential(state, safe, me)
                return self._best_by_potential(state, opp_forks, me)
        return self._best_by_potential(state, cands, me)

    def _best_by_potential(self, state: State, moves: list[int], me: int) -> int:
        best_score, best = None, []
        for mv in moves:
            s = line_potential(state, mv, me)
            if best_score is None or s > best_score + 1e-9:
                best_score, best = s, [mv]
            elif abs(s - best_score) <= 1e-9:
                best.append(mv)
        return self.rng.choice(best)

    def estimate(self, config: GameConfig) -> CostEstimate:
        cands = min(config.size, 25 if config.size > 25 else config.size)
        per = 4 * config.k * config.k * 3e-7
        return CostEstimate(time_per_move=cands * cands * per * 0.2 + cands * per * 4, memory_bytes=1e4)


def line_potential(state: State, cell: int, me: int) -> float:
    """Sum of open-window weights through ``cell`` for both players."""
    game = state.game
    cells = state.cells
    score = 0.0
    for wi in game.cell_windows[cell]:
        own = opp = 0
        for c in game.windows[wi]:
            v = cells[c]
            if v == me:
                own += 1
            elif v != EMPTY:
                opp += 1
        if opp == 0:
            score += 1.0 + 4.0 ** own
        elif own == 0:
            score += 0.5 + 0.8 * 4.0 ** opp
    return score

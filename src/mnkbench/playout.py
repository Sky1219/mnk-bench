"""Fast random playouts (the inner loop of MCTS).

:func:`random_playout` plays a position out to the end with uniformly random
moves and returns the winner, without touching the input state. It works on
a bare copy of the cells (no history, no State bookkeeping) and inlines the
win check, which makes it several times faster than pushing moves through
:class:`~mnkbench.game.State`. :func:`reference_playout` is the simple version
kept for testing and benchmarking.
"""

from __future__ import annotations

import random

from .game import DRAW, EMPTY, State


def reference_playout(state: State, rng: random.Random) -> int:
    sim = state.copy()
    while sim.winner is None:
        sim.push(rng.choice(sim.legal_moves()))
    return sim.winner


def random_playout(state: State, rng: random.Random) -> int:
    if state.winner is not None:
        return state.winner
    game = state.game
    cells = state.cells[:]
    rays = game.rays
    k = game.k
    exact = game.exact_k
    player = state.to_move
    if game.gravity:
        n = game.n
        tops = state.col_top[:]
        open_cols = [c for c in range(n) if tops[c] >= 0]
        rand = rng.random
        while open_cols:
            i = int(rand() * len(open_cols))
            col = open_cols[i]
            mv = tops[col] * n + col
            tops[col] -= 1
            if tops[col] < 0:
                open_cols[i] = open_cols[-1]
                open_cols.pop()
            cells[mv] = player
            for fwd, bwd in rays[mv]:
                cnt = 1
                for c in fwd:
                    if cells[c] != player:
                        break
                    cnt += 1
                for c in bwd:
                    if cells[c] != player:
                        break
                    cnt += 1
                if cnt == k or (cnt > k and not exact):
                    return player
            player = 3 - player
        return DRAW
    empties = [i for i in range(len(cells)) if cells[i] == EMPTY]
    # Lazy Fisher-Yates: draw only as many moves as the game lasts, using the
    # C-level rng.random() (random.shuffle is pure Python and much slower).
    rand = rng.random
    remaining = len(empties)
    while remaining:
        i = int(rand() * remaining)
        mv = empties[i]
        remaining -= 1
        empties[i] = empties[remaining]
        cells[mv] = player
        for fwd, bwd in rays[mv]:
            cnt = 1
            for c in fwd:
                if cells[c] != player:
                    break
                cnt += 1
            if cnt >= k and not exact:
                return player
            for c in bwd:
                if cells[c] != player:
                    break
                cnt += 1
            if cnt == k or (cnt > k and not exact):
                return player
        player = 3 - player
    return DRAW

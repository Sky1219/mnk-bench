"""Rough state-space sizes used by the feasibility estimates."""

from __future__ import annotations

import math
from functools import lru_cache

from .game import GameConfig


@lru_cache(maxsize=256)
def position_bound(config: GameConfig) -> float:
    """Upper bound on the number of positions reachable in ``config``.

    Without gravity: every way to place ceil(i/2) X and floor(i/2) O stones on
    the board, summed over i. With gravity each column is a stack, so the
    bound is (2^(m+1) - 1)^n. Positions that contain a finished game and then
    continue are counted too, so this overestimates (6,046 vs 5,478 for
    tic-tac-toe), but it tracks the growth rate, which is what matters here.
    """
    n_cells = config.size
    free = sum(math.comb(n_cells, i) * math.comb(i, i // 2) for i in range(n_cells + 1))
    if config.gravity:
        stacked = float((2 ** (config.m + 1) - 1)) ** config.n
        return float(min(free, stacked))
    return float(free)


def log10_positions(config: GameConfig) -> float:
    return math.log10(position_bound(config))


def symmetry_count(config: GameConfig) -> int:
    if config.gravity:
        return 2
    return 8 if config.m == config.n else 4


def expected_game_length(config: GameConfig) -> float:
    """Crude typical game length (in plies) for random-ish play."""
    return min(config.size, max(2 * config.k - 1, config.size * 0.6))

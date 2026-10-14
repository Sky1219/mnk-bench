"""Micro-benchmarks for the engine and the search agents.

``run_benchmarks`` measures throughput (operations per second) of the hot
paths on several boards: make/unmake, random playouts, solver nodes,
alpha-beta nodes and MCTS iterations, plus time per move for each agent.
Results go to JSON so performance changes can be tracked over time.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

from .agents import make_agent
from .game import GameConfig, get_game

PERF_BOARDS = [GameConfig(3, 3, 3), GameConfig(6, 6, 4), GameConfig(6, 7, 4, gravity=True),
               GameConfig(9, 9, 5), GameConfig(15, 15, 5), GameConfig(19, 19, 5)]


def _rate(fn: Callable[[], int], min_time: float) -> float:
    """Call ``fn`` (which returns an operation count) until ``min_time`` has passed."""
    ops, t0 = 0, time.perf_counter()
    while True:
        ops += fn()
        dt = time.perf_counter() - t0
        if dt >= min_time:
            return ops / dt


def bench_push_pop(config: GameConfig, min_time: float = 0.3) -> float:
    game = get_game(config)
    rng = random.Random(0)
    state = game.new_state()
    order = list(range(game.size))

    def run() -> int:
        rng.shuffle(order)
        n = 0
        if game.gravity:
            while state.winner is None:
                state.push(rng.choice(state.legal_moves()))
                n += 1
        else:
            for mv in order:
                state.push(mv)
                n += 1
                if state.winner is not None:
                    break
        for _ in range(n):
            state.pop()
        return 2 * n

    return _rate(run, min_time)


def bench_playouts(config: GameConfig, min_time: float = 0.3) -> float:
    from .playout import random_playout

    game = get_game(config)
    rng = random.Random(0)
    state = game.new_state()
    return _rate(lambda: (random_playout(state, rng), 1)[1], min_time)


def bench_agent_move(spec: str, config: GameConfig, moves: int = 3) -> float:
    """Mean seconds per move for ``spec`` from a few early-game positions."""
    game = get_game(config)
    rng = random.Random(1)
    agent = make_agent(spec, seed=0)
    total = 0.0
    for i in range(moves):
        state = game.new_state()
        for _ in range(min(2 + 2 * i, game.size - 1)):
            state.push(rng.choice(state.legal_moves()))
            if state.winner is not None:
                break
        if state.winner is not None:
            continue
        t0 = time.perf_counter()
        agent.select_move(state)
        total += time.perf_counter() - t0
    return total / moves


def run_benchmarks(boards: list[GameConfig] | None = None, quick: bool = False,
                   log: Callable[[str], None] | None = None) -> list[dict]:
    boards = boards or PERF_BOARDS
    min_time = 0.05 if quick else 0.4
    rows = []
    for cfg in boards:
        row = {"config": cfg.label, "cells": cfg.size,
               "push_pop_per_s": bench_push_pop(cfg, min_time),
               "playouts_per_s": bench_playouts(cfg, min_time)}
        for spec in ("rule", "alphabeta:depth=2", "alphabeta:depth=4", "mcts:iterations=500"):
            row[f"sec_per_move[{spec}]"] = bench_agent_move(spec, cfg, 1 if quick else 3)
        if cfg.size <= 16:
            row["sec_per_move[minimax]"] = bench_agent_move("minimax", cfg, 1)
        if log:
            log(", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        rows.append(row)
    return rows

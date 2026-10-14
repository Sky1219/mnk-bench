"""Performance benchmarks and the gomoku (15,15,5) showdown."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from pathlib import Path

from ..agents import make_agent
from ..game import GOMOKU, GameConfig, get_game
from ..perf import PERF_BOARDS, run_benchmarks
from ..playout import random_playout, reference_playout
from ..plots import plot_scaling
from ..results import load_json, metadata, save_csv, save_json
from ..tournament import round_robin
from .sampling import elo_from_tournament

Log = Callable[[str], None]


def playout_speedup(boards: list[GameConfig], min_time: float) -> list[dict]:
    rows = []
    for cfg in boards:
        state = get_game(cfg).new_state()
        rates = {}
        for fn in (reference_playout, random_playout):
            rng = random.Random(0)
            n, t0 = 0, time.perf_counter()
            while time.perf_counter() - t0 < min_time:
                fn(state, rng)
                n += 1
            rates[fn.__name__] = n / (time.perf_counter() - t0)
        rows.append({"config": cfg.label, "cells": cfg.size, "reference_per_s": rates["reference_playout"],
                     "fast_per_s": rates["random_playout"],
                     "speedup": rates["random_playout"] / rates["reference_playout"]})
    return rows


def exp_perf(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    boards = PERF_BOARDS[:3] if quick else PERF_BOARDS
    rows = run_benchmarks(boards, quick=quick, log=log)
    speed = playout_speedup(boards, 0.05 if quick else 0.5)
    data = {"meta": metadata(seed=seed, quick=quick), "benchmarks": rows, "playouts": speed}
    save_json(data, out / "perf" / "perf.json")
    save_csv(rows, out / "perf" / "perf.csv")
    save_csv(speed, out / "perf" / "playout_speedup.csv")
    replot_perf(out)
    return data


def replot_perf(out: Path) -> None:
    data = load_json(out / "perf" / "perf.json")
    long = []
    for r in data["benchmarks"]:
        for key, v in r.items():
            if key.startswith("sec_per_move[") and v is not None:
                long.append({"agent": key[len("sec_per_move["):-1], "config": r["config"], "sec": v})
    plot_scaling(long, "config", "sec", out / "perf" / "time_per_move.png",
                 "Time per move as the board grows", "board", "seconds per move (log)")
    pl = []
    for r in data["playouts"]:
        pl.append({"agent": "reference playout", "config": r["config"], "rate": r["reference_per_s"]})
        pl.append({"agent": "fast playout", "config": r["config"], "rate": r["fast_per_s"]})
    plot_scaling(pl, "config", "rate", out / "perf" / "playouts.png",
                 "Random playouts per second (MCTS inner loop)", "board", "playouts/s (log)")


GOMOKU_AGENTS = ["random", "rule", "alphabeta:depth=2", "alphabeta:depth=4", "alphabeta:depth=6",
                 "mcts:iterations=1000"]


def exp_gomoku(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    cfg = GameConfig(9, 9, 5) if quick else GOMOKU
    specs = GOMOKU_AGENTS[:4] if quick else GOMOKU_AGENTS
    per_pair = 2 if quick else 10
    factories = {s: (lambda s=s: make_agent(s)) for s in specs}
    t = round_robin(cfg, factories, per_pair, seed=seed, progress=log).to_dict()
    elo = elo_from_tournament(t, seed, samples=20 if quick else 200)
    data = {"meta": metadata(seed=seed, quick=quick, games_per_pair=per_pair), "tournament": t, "elo": elo}
    save_json(data, out / "gomoku" / "gomoku.json")
    save_csv(t["wdl"], out / "gomoku" / "gomoku_wdl.csv")
    replot_gomoku(out)
    return data


def replot_gomoku(out: Path) -> None:
    from ..plots import plot_elo, plot_score_matrix

    data = load_json(out / "gomoku" / "gomoku.json")
    t = data["tournament"]
    plot_score_matrix(t, out / "gomoku" / "gomoku_matrix.png", f"Gomoku {t['config']}: score of row vs column")
    ci = {k: tuple(v) for k, v in data["elo"]["ci95"].items()}
    plot_elo(data["elo"]["ratings"], out / "gomoku" / "gomoku_elo.png",
             f"Elo on gomoku {t['config']} (random = 1000)", intervals=ci)

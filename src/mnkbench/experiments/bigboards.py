"""Bigger boards: where tabular learners choke and depth-limited search takes over."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ..agents import make_agent
from ..game import GameConfig
from ..metrics import frozen, wdl
from ..plots import plot_scaling, plot_score_matrix
from ..results import load_json, metadata, save_csv, save_json
from ..tournament import round_robin
from .learning import table_size, train

Log = Callable[[str], None]

BOARDS = [GameConfig(3, 3, 3), GameConfig(4, 4, 3), GameConfig(4, 4, 4), GameConfig(5, 5, 4), GameConfig(6, 6, 4)]
QUICK_BOARDS = [GameConfig(3, 3, 3), GameConfig(4, 4, 4)]


def tabular_scaling(boards: list[GameConfig], train_games: int, eval_games: int, seed: int,
                    log: Log) -> list[dict]:
    """Train each tabular learner for the same number of games on every board
    and measure strength, table size and coverage."""
    rows = []
    for cfg in boards:
        for spec in ("menace", "qlearning", "td"):
            agent = make_agent(spec, seed=seed)
            t0 = time.perf_counter()
            train(agent, cfg, train_games, opponent="random+rule")
            train_time = time.perf_counter() - t0
            with frozen(agent):
                agent.stats.clear()
                vs_random = wdl(agent, make_agent("random"), cfg, eval_games, seed)
                vs_rule = wdl(agent, make_agent("rule"), cfg, eval_games, seed + 1)
                lookups = agent.stats.get("lookups", 0)
                coverage = agent.stats.get("hits", 0) / lookups if lookups else None
            row = {
                "agent": spec,
                "config": cfg.label,
                "cells": cfg.size,
                "train_games": train_games,
                "train_time": train_time,
                "table_size": table_size(agent),
                "memory_bytes": agent.memory_bytes(),
                "coverage": coverage,
                "win_vs_random": vs_random["win"],
                "score_vs_random": vs_random["win"] + 0.5 * vs_random["draw"],
                "score_vs_rule": vs_rule["win"] + 0.5 * vs_rule["draw"],
                "loss_vs_rule": vs_rule["loss"],
            }
            log(f"{cfg.label} {spec}: table {row['table_size']}, coverage {coverage}, "
                f"score vs random {row['score_vs_random']:.2f}, vs rule {row['score_vs_rule']:.2f}")
            rows.append(row)
        for spec in ("rule", "alphabeta:depth=2", "alphabeta:depth=4"):
            agent = make_agent(spec, seed=seed)
            vs_random = wdl(agent, make_agent("random"), cfg, eval_games, seed)
            vs_rule = wdl(agent, make_agent("rule"), cfg, eval_games, seed + 1)
            rows.append({
                "agent": spec, "config": cfg.label, "cells": cfg.size, "train_games": 0, "train_time": 0.0,
                "table_size": 0, "memory_bytes": agent.memory_bytes(), "coverage": None,
                "win_vs_random": vs_random["win"],
                "score_vs_random": vs_random["win"] + 0.5 * vs_random["draw"],
                "score_vs_rule": vs_rule["win"] + 0.5 * vs_rule["draw"],
                "loss_vs_rule": vs_rule["loss"],
                "time_per_move": vs_random["time_per_move"],
            })
    return rows


def exp_bigboards(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    boards = QUICK_BOARDS if quick else BOARDS
    train_games, eval_games = (100, 6) if quick else (3000, 60)
    rows = tabular_scaling(boards, train_games, eval_games, seed, log)

    # A tournament on (6,6,4) (4,4,4 when quick), learners pre-trained the same way.
    cfg = GameConfig(4, 4, 4) if quick else GameConfig(6, 6, 4)
    per_pair = 4 if quick else 40
    trained: dict[str, object] = {}

    def learner(spec: str):
        def make():
            if spec not in trained:
                a = make_agent(spec, seed=seed)
                train(a, cfg, train_games, opponent="random+rule")
                a.training = False
                trained[spec] = a
            return trained[spec]
        return make

    factories = {
        "random": lambda: make_agent("random"),
        "rule": lambda: make_agent("rule"),
        "menace": learner("menace"),
        "qlearning": learner("qlearning"),
        "td": learner("td"),
        "alphabeta:depth=2": lambda: make_agent("alphabeta:depth=2"),
        "alphabeta:depth=4": lambda: make_agent("alphabeta:depth=4"),
    }
    tour = round_robin(cfg, factories, per_pair, seed=seed, progress=log).to_dict()
    data = {"meta": metadata(seed=seed, quick=quick, train_games=train_games, eval_games=eval_games),
            "scaling": rows, "tournament": tour}
    save_json(data, out / "bigboards" / "bigboards.json")
    save_csv(rows, out / "bigboards" / "tabular_scaling.csv")
    save_csv(tour["wdl"], out / "bigboards" / "tournament_wdl.csv")
    replot_bigboards(out)
    return data


def replot_bigboards(out: Path) -> None:
    data = load_json(out / "bigboards" / "bigboards.json")
    rows = data["scaling"]
    d = out / "bigboards"
    plot_scaling(rows, "config", "score_vs_rule", d / "score_vs_rule.png",
                 "Score vs the rule-based player as the board grows", "board",
                 "score (win + draw/2)", logy=False)
    tab = [r for r in rows if r["table_size"]]
    plot_scaling(tab, "config", "table_size", d / "table_size.png",
                 f"Table entries after {data['meta']['train_games']} training games", "board",
                 "entries (log)")
    plot_scaling([r for r in tab if r["coverage"] is not None], "config", "coverage", d / "coverage.png",
                 "Share of evaluation positions the learner had seen before", "board",
                 "coverage", logy=False)
    plot_score_matrix(data["tournament"], d / "tournament_matrix.png",
                      f"Head-to-head on {data['tournament']['config']}, row vs column")

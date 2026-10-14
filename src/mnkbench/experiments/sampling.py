"""MCTS strength vs simulation budget, and Elo ratings from round robins."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..agents import make_agent
from ..agents.base import Agent
from ..elo import bootstrap_elo, fit_elo
from ..game import TIC_TAC_TOE, GameConfig
from ..metrics import optimal_move_rate, wdl
from ..plots import plot_elo, plot_scaling, plot_score_matrix
from ..results import load_json, metadata, save_csv, save_json
from ..tournament import round_robin
from .learning import train

Log = Callable[[str], None]


def exp_mcts(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    """How strong is MCTS as a function of playouts per move?"""
    budgets = [25, 100, 400] if quick else [25, 50, 100, 200, 400, 800, 1600, 3200]
    n = 6 if quick else 40
    rows = []
    for it in budgets:
        spec = f"mcts:iterations={it}"
        opt = optimal_move_rate(make_agent(spec), TIC_TAC_TOE, n, seed=seed)
        rows.append({"agent": "mcts", "iterations": it, "config": "(3,3,3)", "metric": "optimal_rate",
                     "value": opt["optimal_rate"]})
        for cfg in (GameConfig(4, 4, 4), GameConfig(6, 6, 4), GameConfig(6, 7, 4, gravity=True)):
            if quick and cfg.size > 16:
                continue
            r = wdl(make_agent(spec), make_agent("rule"), cfg, n, seed)
            rows.append({"agent": "mcts", "iterations": it, "config": cfg.label, "metric": "score_vs_rule",
                         "value": r["win"] + 0.5 * r["draw"], "time_per_move": r["time_per_move"]})
        log(f"mcts {it}: optimal {opt['optimal_rate']:.3f}")
    data = {"meta": metadata(seed=seed, quick=quick, games=n), "rows": rows}
    save_json(data, out / "mcts" / "mcts.json")
    save_csv(rows, out / "mcts" / "mcts.csv")
    replot_mcts(out)
    return data


def replot_mcts(out: Path) -> None:
    data = load_json(out / "mcts" / "mcts.json")
    rows = []
    for r in data["rows"]:
        name = "optimal-move rate (3,3,3)" if r["metric"] == "optimal_rate" else f"score vs rule {r['config']}"
        rows.append({"series": name, "iterations": r["iterations"], "value": r["value"]})
    plot_scaling(rows, "iterations", "value", out / "mcts" / "mcts_budget.png",
                 "MCTS strength vs playouts per move", "playouts per move (log)", "rate",
                 logy=False, group="series", logx=True)


# ---------------------------------------------------------------------- Elo
def elo_agents(cfg: GameConfig, quick: bool, seed: int) -> dict[str, Callable[[], Agent]]:
    train_games = 300 if quick else 5000
    trained: dict[str, Agent] = {}

    def learner(spec: str) -> Callable[[], Agent]:
        def make() -> Agent:
            if spec not in trained:
                a = make_agent(spec, seed=seed)
                train(a, cfg, train_games, opponent="random+minimax" if cfg.size <= 9 else "random+rule")
                a.training = False
                trained[spec] = a
            return trained[spec]
        return make

    agents: dict[str, Callable[[], Agent]] = {
        "random": lambda: make_agent("random"),
        "rule": lambda: make_agent("rule"),
    }
    if cfg.size <= 16:
        agents["minimax"] = lambda: make_agent("minimax")
    if cfg.size <= 9:
        for spec in ("menace", "qlearning", "td"):
            agents[spec] = learner(spec)
    agents["alphabeta:depth=2"] = lambda: make_agent("alphabeta:depth=2")
    agents["alphabeta:depth=4"] = lambda: make_agent("alphabeta:depth=4")
    agents["mcts:iterations=200"] = lambda: make_agent("mcts:iterations=200")
    agents["mcts:iterations=1000"] = lambda: make_agent("mcts:iterations=1000")
    from .suite import _alphazero_entry

    agents.update(_alphazero_entry(cfg))
    return agents


def elo_from_tournament(t: dict, seed: int = 0, samples: int = 100) -> dict:
    games = []
    for m in t["matches"]:
        games += [(m["a"], m["b"], 1.0)] * m["wins_a"]
        games += [(m["a"], m["b"], 0.0)] * m["wins_b"]
        games += [(m["a"], m["b"], 0.5)] * m["draws"]
    ratings = fit_elo(games, anchor="random", anchor_rating=1000.0)
    ci = bootstrap_elo(games, samples=samples, seed=seed, anchor="random", anchor_rating=1000.0)
    return {"ratings": ratings, "ci95": {k: list(v) for k, v in ci.items()}, "games": len(games)}


def exp_elo(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    boards = [TIC_TAC_TOE, GameConfig(4, 4, 4)] if quick else \
        [TIC_TAC_TOE, GameConfig(6, 6, 4), GameConfig(6, 7, 4, gravity=True)]
    per_pair = 4 if quick else 40
    results = []
    for cfg in boards:
        log(f"Elo round robin on {cfg.label}")
        t = round_robin(cfg, elo_agents(cfg, quick, seed), per_pair, seed=seed, progress=log).to_dict()
        elo = elo_from_tournament(t, seed, samples=20 if quick else 200)
        results.append({"config": cfg.label, "tournament": t, "elo": elo})
    rows = [{"config": r["config"], "agent": a, "elo": e, "ci_low": r["elo"]["ci95"][a][0],
             "ci_high": r["elo"]["ci95"][a][1]} for r in results for a, e in r["elo"]["ratings"].items()]
    data = {"meta": metadata(seed=seed, quick=quick, games_per_pair=per_pair), "boards": results}
    save_json(data, out / "elo" / "elo.json")
    save_csv(rows, out / "elo" / "elo.csv")
    replot_elo(out)
    return data


def _slug(label: str) -> str:
    return label.strip("()").replace(",", "-").replace(")+", "-").replace("+", "-")


def replot_elo(out: Path) -> None:
    data = load_json(out / "elo" / "elo.json")
    for r in data["boards"]:
        slug = _slug(r["config"])
        ci = {k: tuple(v) for k, v in r["elo"]["ci95"].items()}
        plot_elo(r["elo"]["ratings"], out / "elo" / f"elo_{slug}.png",
                 f"Elo on {r['config']} (random = 1000, 95% bootstrap CI)", intervals=ci)
        plot_score_matrix(r["tournament"], out / "elo" / f"matrix_{slug}.png",
                          f"Score of row vs column on {r['config']}")

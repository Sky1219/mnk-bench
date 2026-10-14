"""The benchmark suite: named experiments that write data and plots.

Each experiment takes an output directory and a ``quick`` flag (tiny sizes,
for CI smoke runs) and writes ``<name>.json`` (plus CSV where tabular) and
PNG plots into ``out_dir/<name>/``. ``replot`` redraws every plot from the
saved JSON alone.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..agents import make_agent
from ..agents.base import Agent
from ..game import TIC_TAC_TOE, GameConfig
from ..plots import plot_learning_curves, plot_score_matrix, plot_table_growth
from ..results import load_json, metadata, save_csv, save_json
from ..tournament import round_robin
from .learning import learning_curve, train

Log = Callable[[str], None]


def _noop(_: str) -> None:
    pass


# ------------------------------------------------------------------ MENACE
def exp_menace(out: Path, quick: bool = False, seed: int = 0, log: Log = _noop) -> dict:
    """MENACE on tic-tac-toe trained against random, perfect and mixed opponents."""
    total, every, evals = (200, 100, 10) if quick else (5000, 250, 200)
    runs = []
    for opp in ("random", "minimax", "random+minimax", "self"):
        log(f"menace vs {opp}")
        runs.append(learning_curve("menace", TIC_TAC_TOE, total, every, evals, opponent=opp, seed=seed, progress=log))
    data = {"meta": metadata(seed=seed, quick=quick), "runs": runs}
    save_json(data, out / "menace" / "menace.json")
    _menace_csv(runs, out / "menace" / "menace.csv")
    replot_menace(out)
    return data


def _menace_csv(runs: list[dict], path: Path) -> None:
    rows = []
    for run in runs:
        for cp in run["checkpoints"]:
            rows.append(
                {
                    "agent": run["agent"],
                    "opponent": run["opponent"],
                    "games": cp["games"],
                    "win_vs_random": cp["vs_random"]["win"],
                    "loss_vs_random": cp["vs_random"]["loss"],
                    "nonloss_vs_minimax": 1 - cp["vs_minimax"]["loss"],
                    "optimal_rate": cp.get("optimal_rate"),
                    "table_size": cp["table_size"],
                    "train_time": cp["train_time"],
                }
            )
    save_csv(rows, path)


def replot_menace(out: Path) -> None:
    data = load_json(out / "menace" / "menace.json")
    runs = data["runs"]
    plot_learning_curves(runs, out / "menace" / "menace_learning.png", "MENACE learning on tic-tac-toe (3,3,3)")
    plot_table_growth(runs, out / "menace" / "menace_boxes.png", "MENACE matchboxes in use")


# --------------------------------------------------------------- symmetry
def exp_symmetry(out: Path, quick: bool = False, seed: int = 0, log: Log = _noop) -> dict:
    """How much do board symmetries shrink MENACE, and does it learn faster?"""
    from ..agents.menace import count_matchboxes

    counts = []
    for cfg in (TIC_TAC_TOE,) if quick else (TIC_TAC_TOE, GameConfig(3, 4, 3)):
        for player in (1, 2):
            for sym in (True, False):
                for forced in (True, False):
                    n = count_matchboxes(cfg, player, use_symmetry=sym, skip_forced=forced)
                    counts.append({"config": cfg.label, "player": "XO"[player - 1], "symmetry": sym,
                                   "skip_forced": forced, "matchboxes": n})
                    log(f"{cfg.label} {'XO'[player - 1]} symmetry={sym} skip_forced={forced}: {n}")
    total, every, evals = (200, 100, 10) if quick else (4000, 250, 200)
    runs = []
    for sym in (True, False):
        spec = f"menace:use_symmetry={sym}"
        log(spec)
        run = learning_curve(spec, TIC_TAC_TOE, total, every, evals, opponent="random+minimax", seed=seed)
        run["agent"] = "menace" if sym else "menace (no symmetry)"
        runs.append(run)
    data = {"meta": metadata(seed=seed, quick=quick), "matchbox_counts": counts, "runs": runs}
    save_json(data, out / "symmetry" / "symmetry.json")
    save_csv(counts, out / "symmetry" / "matchbox_counts.csv")
    replot_symmetry(out)
    return data


def replot_symmetry(out: Path) -> None:
    data = load_json(out / "symmetry" / "symmetry.json")
    plot_learning_curves(data["runs"], out / "symmetry" / "symmetry_learning.png",
                         "MENACE with and without board symmetries (3,3,3)")
    plot_table_growth(data["runs"], out / "symmetry" / "symmetry_boxes.png",
                      "Matchboxes used, with and without symmetries")


# ------------------------------------------------------------ head to head
def trained_factory(spec: str, config: GameConfig, games: int, opponent: str = "random+minimax",
                    seed: int = 0) -> Callable[[], Agent]:
    """Factory for a learner pre-trained for ``games`` games, then frozen.
    The trained agent is built once and reused by every match."""
    cache: dict[str, Agent] = {}

    def make() -> Agent:
        if "agent" not in cache:
            agent = make_agent(spec, seed=seed)
            train(agent, config, games, opponent)
            agent.training = False
            cache["agent"] = agent
        return cache["agent"]

    return make


def head_to_head(config: GameConfig, factories: dict[str, Callable[[], Agent]], games_per_pair: int,
                 seed: int = 0, log: Log = _noop) -> dict:
    t = round_robin(config, factories, games_per_pair, seed=seed, progress=log)
    return t.to_dict()


def wdl_markdown(tournament: dict) -> str:
    """Markdown table of total W/D/L per agent, best score first."""
    rows = []
    for r in tournament["wdl"]:
        n = r["win"] + r["draw"] + r["loss"]
        score = (r["win"] + 0.5 * r["draw"]) / n if n else float("nan")
        rows.append((score, r, n))
    rows.sort(key=lambda t: -t[0])
    lines = ["| agent | games | win | draw | loss | score |", "|---|---:|---:|---:|---:|---:|"]
    for score, r, n in rows:
        lines.append(f"| {r['agent']} | {n} | {r['win']} | {r['draw']} | {r['loss']} | {score:.3f} |")
    return "\n".join(lines) + "\n"


def exp_h2h(out: Path, quick: bool = False, seed: int = 0, log: Log = _noop) -> dict:
    """Round robin on tic-tac-toe between every baseline and trained learner."""
    games = 10 if quick else 100
    train_games = 200 if quick else 5000
    cfg = TIC_TAC_TOE
    factories = h2h_agents(cfg, train_games, seed, quick)
    data = head_to_head(cfg, factories, games, seed, log)
    data["meta"] = metadata(seed=seed, quick=quick, train_games=train_games)
    save_json(data, out / "h2h" / "h2h.json")
    save_csv(data["wdl"], out / "h2h" / "h2h_wdl.csv")
    (out / "h2h" / "h2h_wdl.md").write_text(wdl_markdown(data))
    replot_h2h(out)
    return data


def h2h_agents(cfg: GameConfig, train_games: int, seed: int, quick: bool) -> dict[str, Callable[[], Agent]]:
    return {
        "random": lambda: make_agent("random"),
        "rule": lambda: make_agent("rule"),
        "minimax": lambda: make_agent("minimax"),
        "menace": trained_factory("menace", cfg, train_games, seed=seed),
    }


def replot_h2h(out: Path) -> None:
    data = load_json(out / "h2h" / "h2h.json")
    plot_score_matrix(data, out / "h2h" / "h2h_matrix.png", "Head-to-head score on (3,3,3), row vs column")


EXPERIMENTS: dict[str, Callable[..., dict]] = {
    "menace": exp_menace,
    "symmetry": exp_symmetry,
    "h2h": exp_h2h,
}
REPLOTTERS: dict[str, Callable[[Path], None]] = {
    "menace": replot_menace,
    "symmetry": replot_symmetry,
    "h2h": replot_h2h,
}


def run(names: list[str] | None, out: str | Path, quick: bool = False, seed: int = 0, log: Log = print) -> None:
    out = Path(out)
    for name in names or list(EXPERIMENTS):
        log(f"== experiment {name} ==")
        EXPERIMENTS[name](out, quick=quick, seed=seed, log=log)


def replot(out: str | Path, names: list[str] | None = None) -> list[str]:
    out = Path(out)
    done = []
    for name in names or list(REPLOTTERS):
        if (out / name / f"{name}.json").exists():
            REPLOTTERS[name](out)
            done.append(name)
    return done

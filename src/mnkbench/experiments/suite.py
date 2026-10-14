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
from .az_exp import exp_alphazero, replot_alphazero
from .bigboards import exp_bigboards, replot_bigboards
from .feasible import exp_feasibility, replot_feasibility
from .gomoku import exp_gomoku, exp_perf, replot_gomoku, replot_perf
from .learning import learning_curve, train
from .llm_exp import exp_llm, replot_llm
from .sampling import exp_elo, exp_mcts, replot_elo, replot_mcts
from .targeted import exp_targeted, replot_targeted

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


# ---------------------------------------------------------------- tabular
TABULAR = ("menace", "qlearning", "td")


def exp_tabular(out: Path, quick: bool = False, seed: int = 0, log: Log = _noop) -> dict:
    """MENACE vs Q-learning vs TD(0) on tic-tac-toe, against two training regimes."""
    total, every, evals = (200, 100, 10) if quick else (6000, 500, 200)
    seeds = [seed] if quick else [seed, seed + 1, seed + 2]
    runs = []
    for spec in TABULAR:
        for opp in ("random+minimax", "self"):
            for sd in seeds:
                log(f"{spec} vs {opp} seed {sd}")
                runs.append(learning_curve(spec, TIC_TAC_TOE, total, every, evals, opponent=opp, seed=sd))
    summary = []
    for spec in TABULAR:
        for opp in ("random+minimax", "self"):
            rs = [r for r in runs if r["agent"] == spec and r["opponent"] == opp]
            learned = [r["games_to_learn"] for r in rs]
            last = [r["checkpoints"][-1] for r in rs]
            summary.append({
                "agent": spec,
                "opponent": opp,
                "seeds": len(rs),
                "games_to_learn": learned,
                "learned_runs": sum(g is not None for g in learned),
                "final_nonloss_vs_minimax": sum(1 - c["vs_minimax"]["loss"] for c in last) / len(last),
                "final_win_vs_random": sum(c["vs_random"]["win"] for c in last) / len(last),
                "final_optimal_rate": sum(c["optimal_rate"] for c in last) / len(last),
                "table_size": sum(c["table_size"] for c in last) / len(last),
                "train_time": sum(c["train_time"] for c in last) / len(last),
            })
    data = {"meta": metadata(seed=seed, quick=quick, seeds=seeds), "runs": runs, "summary": summary}
    save_json(data, out / "tabular" / "tabular.json")
    save_csv(summary, out / "tabular" / "tabular_summary.csv")
    replot_tabular(out)
    return data


def replot_tabular(out: Path) -> None:
    data = load_json(out / "tabular" / "tabular.json")
    first_seed = data["meta"]["seeds"][0]
    for opp, tag in (("random+minimax", "mixed"), ("self", "selfplay")):
        runs = [r for r in data["runs"] if r["opponent"] == opp and r["seed"] == first_seed]
        plot_learning_curves(runs, out / "tabular" / f"tabular_{tag}.png",
                             f"Tabular learners on (3,3,3), trained vs {opp}")
    runs = [r for r in data["runs"] if r["opponent"] == "random+minimax" and r["seed"] == first_seed]
    plot_table_growth(runs, out / "tabular" / "tabular_tables.png", "Table entries during training")


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
        "qlearning": trained_factory("qlearning", cfg, train_games, seed=seed),
        "td": trained_factory("td", cfg, train_games, seed=seed),
        "alphabeta:depth=4": lambda: make_agent("alphabeta:depth=4"),
        "mcts:iterations=1000": lambda: make_agent("mcts:iterations=1000"),
        **_alphazero_entry(cfg),
    }


def _alphazero_entry(cfg: GameConfig) -> dict[str, Callable[[], Agent]]:
    """Include the AlphaZero agent when PyTorch and a trained network are available."""
    try:
        from ..agents.alphazero import has_bundled_checkpoint
    except ImportError:
        return {}
    if not has_bundled_checkpoint(cfg):
        return {}
    return {"alphazero": lambda: make_agent("alphazero:simulations=100")}


def replot_h2h(out: Path) -> None:
    data = load_json(out / "h2h" / "h2h.json")
    plot_score_matrix(data, out / "h2h" / "h2h_matrix.png", "Head-to-head score on (3,3,3), row vs column")


EXPERIMENTS: dict[str, Callable[..., dict]] = {
    "menace": exp_menace,
    "symmetry": exp_symmetry,
    "tabular": exp_tabular,
    "h2h": exp_h2h,
    "bigboards": exp_bigboards,
    "mcts": exp_mcts,
    "elo": exp_elo,
    "feasibility": exp_feasibility,
    "perf": exp_perf,
    "gomoku": exp_gomoku,
    "alphazero": exp_alphazero,
    "llm": exp_llm,
    "targeted": exp_targeted,
}
REPLOTTERS: dict[str, Callable[[Path], None]] = {
    "menace": replot_menace,
    "symmetry": replot_symmetry,
    "tabular": replot_tabular,
    "h2h": replot_h2h,
    "bigboards": replot_bigboards,
    "mcts": replot_mcts,
    "elo": replot_elo,
    "feasibility": replot_feasibility,
    "perf": replot_perf,
    "gomoku": replot_gomoku,
    "alphazero": replot_alphazero,
    "llm": replot_llm,
    "targeted": replot_targeted,
}


def run(names: list[str] | None, out: str | Path, quick: bool = False, seed: int = 0, log: Log = print) -> None:
    """Run experiments. When running everything (``names`` empty), experiments
    whose optional dependency is missing (AlphaZero without PyTorch) are
    skipped; explicitly requested ones raise."""
    out = Path(out)
    explicit = bool(names)
    for name in names or list(EXPERIMENTS):
        log(f"== experiment {name} ==")
        try:
            EXPERIMENTS[name](out, quick=quick, seed=seed, log=log)
        except ImportError as exc:
            if explicit:
                raise
            log(f"skipping {name}: {exc}")


def replot(out: str | Path, names: list[str] | None = None) -> list[str]:
    out = Path(out)
    done = []
    for name in names or list(REPLOTTERS):
        if (out / name / f"{name}.json").exists():
            REPLOTTERS[name](out)
            done.append(name)
    return done

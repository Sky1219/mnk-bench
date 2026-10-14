"""The benchmark suite: named experiments that write data and plots.

Each experiment takes an output directory and a ``quick`` flag (tiny sizes,
for CI smoke runs) and writes ``<name>.json`` (plus CSV where tabular) and
PNG plots into ``out_dir/<name>/``. ``replot`` redraws every plot from the
saved JSON alone.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..game import TIC_TAC_TOE
from ..plots import plot_learning_curves, plot_table_growth
from ..results import load_json, metadata, save_csv, save_json
from .learning import learning_curve

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


EXPERIMENTS: dict[str, Callable[..., dict]] = {
    "menace": exp_menace,
}
REPLOTTERS: dict[str, Callable[[Path], None]] = {
    "menace": replot_menace,
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

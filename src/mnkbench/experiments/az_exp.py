"""AlphaZero-style self-play training on small boards (needs PyTorch)."""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
from collections.abc import Callable
from pathlib import Path

from ..game import GameConfig
from ..results import load_json, metadata, save_json

Log = Callable[[str], None]

# (board, iterations, self-play games per iteration, simulations per move)
FULL_PLAN = [("3,3,3", 25, 50, 50), ("4,4,4", 25, 40, 64), ("6,7,4+gravity", 25, 40, 64)]
QUICK_PLAN = [("3,3,3", 2, 6, 12)]


def _train_one(job: tuple) -> dict:
    board, iterations, games, sims, eval_games, eval_sims, seed, ckpt_dir = job
    from ..agents.alphazero import train

    return train(GameConfig.parse(board), iterations=iterations, games_per_iter=games, simulations=sims,
                 eval_games=eval_games, eval_simulations=eval_sims, seed=seed, out_dir=ckpt_dir,
                 log=lambda s: None)


def exp_alphazero(out: Path, quick: bool = False, seed: int = 0, log: Log = print,
                  jobs: int | None = None, bundle: bool = False) -> dict:
    from ..agents.alphazero import DATA_DIR, checkpoint_name, require_torch

    require_torch()
    plan = QUICK_PLAN if quick else FULL_PLAN
    eval_games, eval_sims = (4, 12) if quick else (40, 100)
    ckpt_dir = out / "alphazero" / "checkpoints"
    work = [(b, it, g, s, eval_games, eval_sims, seed, str(ckpt_dir)) for b, it, g, s in plan]
    jobs = jobs or min(len(work), os.cpu_count() or 1)
    log(f"training AlphaZero on {', '.join(b for b, *_ in plan)} ({jobs} parallel jobs)")
    if jobs > 1 and len(work) > 1:
        with mp.get_context("spawn").Pool(jobs) as pool:
            runs = pool.map(_train_one, work)
    else:
        runs = [_train_one(w) for w in work]
    data = {"meta": metadata(seed=seed, quick=quick, plan=plan), "runs": runs}
    save_json(data, out / "alphazero" / "alphazero.json")
    if bundle:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for b, *_ in plan:
            name = checkpoint_name(GameConfig.parse(b))
            shutil.copy(ckpt_dir / name, DATA_DIR / name)
    replot_alphazero(out)
    return data


def replot_alphazero(out: Path) -> None:
    from ..plots import plot_alphazero

    data = load_json(out / "alphazero" / "alphazero.json")
    plot_alphazero(data["runs"], out / "alphazero" / "alphazero_training.png",
                   "AlphaZero-style agent: evaluation after each self-play iteration")

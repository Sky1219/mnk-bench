"""LLM players: strength and cost per move.

With credentials (``ANTHROPIC_API_KEY``) this plays real games against
Claude; without them it runs the identical pipeline against the offline mock
client so the code path is exercised, and says so in the saved note.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from ..agents import make_agent
from ..game import TIC_TAC_TOE, GameConfig
from ..metrics import optimal_move_rate
from ..results import load_json, metadata, save_csv, save_json
from ..tournament import play_match

Log = Callable[[str], None]


def exp_llm(out: Path, quick: bool = False, seed: int = 0, log: Log = print,
            spec: str | None = None) -> dict:
    if spec is None:
        live = bool(os.environ.get("ANTHROPIC_API_KEY"))
        spec = "llm" if live else "llm:provider=mock"
    else:
        live = "mock" not in spec
    plan = [(TIC_TAC_TOE, "random", 4 if quick else 10), (TIC_TAC_TOE, "rule", 2 if quick else 10)]
    if not quick:
        plan += [(GameConfig(6, 7, 4, gravity=True), "rule", 4), (GameConfig(9, 9, 5), "rule", 2)]
    rows = []
    for cfg, opp, games in plan:
        agent = make_agent(spec, seed=seed)
        log(f"{spec} vs {opp} on {cfg.label} ({games} games)")
        r = play_match(cfg, agent, make_agent(opp), games, seed=seed, a_name="llm", b_name=opp)
        s = agent.summary()
        rows.append({
            "agent": spec, "config": cfg.label, "opponent": opp, "games": r.n_games,
            "win": r.wins_a, "draw": r.draws, "loss": r.wins_b, "score": r.score_a,
            "sec_per_move": r.time_per_move.get("llm"),
            "input_tokens_per_move": s["input_tokens_per_move"],
            "output_tokens_per_move": s["output_tokens_per_move"],
            "cost_per_move_usd": s["cost_per_move_usd"],
            "retry_rate": s["retry_rate"], "fallback_rate": s["fallback_rate"],
        })
    agent = make_agent(spec, seed=seed)
    opt = optimal_move_rate(agent, TIC_TAC_TOE, 4 if quick else 10, seed=seed)
    note = ("Live games against Claude via the Anthropic API." if live else
            "No ANTHROPIC_API_KEY was available when these results were generated, so this table comes "
            "from the offline mock client (random legal moves with simulated token counts). It checks the "
            "pipeline, not a model. Set ANTHROPIC_API_KEY and run `mnk-bench experiment llm` for real numbers.")
    data = {"meta": metadata(seed=seed, quick=quick, live=live, spec=spec), "note": note, "rows": rows,
            "optimal_move_rate_333": opt}
    save_json(data, out / "llm" / "llm.json")
    save_csv(rows, out / "llm" / "llm.csv")
    return data


def replot_llm(out: Path) -> None:
    load_json(out / "llm" / "llm.json")  # tables only; nothing to redraw

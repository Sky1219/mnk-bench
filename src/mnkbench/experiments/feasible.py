"""The feasibility map: which methods survive at which board sizes."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..agents import REGISTRY
from ..agents.base import GB, MB, Budget
from ..feasibility import DEFAULT_BOARDS, feasibility_map
from ..game import GameConfig
from ..plots import plot_feasibility_map
from ..results import load_json, metadata, save_csv, save_json

Log = Callable[[str], None]

BUDGETS = {
    "standard": Budget(time_per_move=1.0, memory_bytes=1 * GB, train_time=600.0, cost_per_move=0.05),
    "tight": Budget(time_per_move=0.05, memory_bytes=128 * MB, train_time=30.0, cost_per_move=0.005),
}


def map_specs() -> list[str]:
    specs = ["random", "rule", "minimax", "menace", "qlearning", "td", "alphabeta:depth=4",
             "mcts:iterations=1000"]
    if "alphazero" in REGISTRY:
        specs.append("alphazero")
    if "llm" in REGISTRY:
        specs.append("llm")
    return specs


def frontier(data: dict) -> list[dict]:
    """Largest board (in the map's order) each agent can still play."""
    rows = []
    for agent in data["agents"]:
        cells = [c for c in data["cells"] if c["agent"] == agent]
        ok = [c for c in cells if c["status"] == "ok"]
        usable = [c for c in cells if c["status"] in ("ok", "degraded")]
        rows.append({
            "agent": agent,
            "largest_ok": ok[-1]["config"] if ok else None,
            "largest_usable": usable[-1]["config"] if usable else None,
            "first_refused": next((c["config"] for c in cells if c["status"] == "refused"), None),
            "aborted": [c["config"] for c in cells if c["status"] == "aborted"],
        })
    return rows


def exp_feasibility(out: Path, quick: bool = False, seed: int = 0, log: Log = print) -> dict:
    boards: list[GameConfig] = DEFAULT_BOARDS[:4] if quick else DEFAULT_BOARDS
    specs = map_specs()
    if quick:
        specs = [s for s in specs if s not in ("alphazero", "llm")]
    maps = {}
    for name, budget in BUDGETS.items():
        log(f"feasibility map, {name} budget")
        maps[name] = feasibility_map(specs, boards, budget, measure=not quick, probe_moves=2,
                                     verify_frontier=not quick, log=log)
        maps[name]["frontier"] = frontier(maps[name])
    data = {"meta": metadata(seed=seed, quick=quick), "maps": maps}
    save_json(data, out / "feasibility" / "feasibility.json")
    rows = []
    for name, m in maps.items():
        for c in m["cells"]:
            rows.append({"budget": name, "agent": c["agent"], "config": c["config"], "status": c["status"],
                         "estimated_status": c["estimated_status"],
                         "est_time_per_move": c["estimate"]["time_per_move"],
                         "est_memory_bytes": c["estimate"]["memory_bytes"],
                         "est_train_time": c["estimate"]["train_time"],
                         "measured_time_per_move": (c.get("probe") or {}).get("seconds_per_move"),
                         "reason": c["reason"]})
    save_csv(rows, out / "feasibility" / "feasibility.csv")
    replot_feasibility(out)
    return data


def replot_feasibility(out: Path) -> None:
    data = load_json(out / "feasibility" / "feasibility.json")
    for name, m in data["maps"].items():
        plot_feasibility_map(m, out / "feasibility" / f"feasibility_{name}.png", f"Feasibility map, {name} budget")

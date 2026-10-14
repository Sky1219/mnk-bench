"""Know when a method won't fit.

Two levels of checking:

* :func:`assess` asks the agent for its own *estimate* (cheap, no games
  played) and classifies it as ok / degraded / refused against a
  :class:`Budget`. This is what an agent uses to refuse a board up front.
* :func:`probe` actually runs the (possibly degraded) agent for a few moves
  in a child process with a hard wall-clock timeout and an address-space
  limit. An agent that overruns is killed and reported as ``aborted``, so a
  bad estimate can never hang or crash the benchmark.

:func:`feasibility_map` sweeps agents x boards and returns data for the
"feasibility map" plot.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .agents import make_agent
from .agents.base import AgentRefused, Budget, BudgetExceeded, FeasibilityReport, Status
from .game import GameConfig, get_game


def assess(spec: str, config: GameConfig, budget: Budget) -> FeasibilityReport:
    """Estimate-only feasibility of agent ``spec`` on ``config``."""
    try:
        agent = make_agent(spec)
    except ImportError as exc:
        from .agents.base import CostEstimate

        return FeasibilityReport(spec, config.label, Status.UNAVAILABLE,
                                 CostEstimate(float("nan"), float("nan")), reason=str(exc))
    report = agent.feasibility(config, budget)
    report.agent = spec
    return report


@dataclass
class ProbeResult:
    status: Status
    seconds_per_move: float | None
    max_seconds: float | None
    moves: int
    peak_rss: float | None
    reason: str = ""

    def to_dict(self) -> dict:
        return {"status": self.status.value, "seconds_per_move": self.seconds_per_move,
                "max_seconds": self.max_seconds, "moves": self.moves, "peak_rss": self.peak_rss,
                "reason": self.reason}


def _probe_child(spec: str, config: GameConfig, budget: Budget, moves: int, memory_limit: float | None,
                 force: bool, conn) -> None:  # pragma: no cover - runs in a subprocess
    try:
        if memory_limit:
            try:
                import resource

                resource.setrlimit(resource.RLIMIT_AS, (int(memory_limit), int(memory_limit)))
            except (ImportError, ValueError, OSError):
                pass
        if force:
            agent = make_agent(spec, seed=0)
            agent.apply_budget(budget)
        else:
            agent = make_agent(spec, seed=0).adapt(config, budget)
        game = get_game(config)
        state = game.new_state()
        opponent = make_agent("random", seed=1)
        agent.new_game(game, 1)
        times = []
        while state.winner is None and len(times) < moves:
            t0 = time.perf_counter()
            mv = agent.select_move(state)
            times.append(time.perf_counter() - t0)
            conn.send(("tick", times[-1]))
            state.push(mv)
            if state.winner is None:
                state.push(opponent.select_move(state))
        peak = None
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0
        except ImportError:
            pass
        conn.send(("done", times, peak))
    except AgentRefused as exc:
        conn.send(("refused", str(exc)))
    except (BudgetExceeded, MemoryError) as exc:
        conn.send(("aborted", f"{type(exc).__name__}: {exc}"))
    except ImportError as exc:
        conn.send(("unavailable", str(exc)))
    except Exception as exc:  # noqa: BLE001 - report anything else as an abort
        conn.send(("aborted", f"{type(exc).__name__}: {exc}"))
    finally:
        conn.close()


def probe(spec: str, config: GameConfig, budget: Budget, moves: int = 3, slack: float = 3.0,
          startup: float = 5.0, enforce_memory: bool = True, force: bool = False) -> ProbeResult:
    """Run the agent for up to ``moves`` of its moves in a child process.

    The child is killed if any single move takes longer than ``slack`` times
    the per-move budget (plus ``startup`` seconds for the first move, which
    includes imports and up-front work). The address space is capped at the
    memory budget plus a fixed allowance for the interpreter. With
    ``force=True`` the agent plays even if it would refuse (with its run-time
    guards installed), to check that a refusal was justified.
    """
    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context()
    parent, child = ctx.Pipe(duplex=False)
    mem = budget.memory_bytes + 1.5 * 2**30 if enforce_memory else None
    proc = ctx.Process(target=_probe_child, args=(spec, config, budget, moves, mem, force, child), daemon=True)
    proc.start()
    child.close()
    times: list[float] = []
    limit = budget.time_per_move * slack
    deadline = time.perf_counter() + limit + startup
    result: ProbeResult | None = None
    while result is None:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            result = ProbeResult(Status.ABORTED, None, None, len(times), None,
                                 f"move took longer than {limit:.2f}s; killed")
            break
        if parent.poll(min(remaining, 0.5)):
            try:
                msg = parent.recv()
            except EOFError:
                result = ProbeResult(Status.ABORTED, None, None, len(times), None, "process died")
                break
            kind = msg[0]
            if kind == "tick":
                times.append(msg[1])
                deadline = time.perf_counter() + limit
            elif kind == "done":
                ts, peak = msg[1], msg[2]
                status = Status.OK if max(ts) <= limit else Status.ABORTED
                result = ProbeResult(status, sum(ts) / len(ts), max(ts), len(ts), peak)
            else:
                status = {"refused": Status.REFUSED, "aborted": Status.ABORTED,
                          "unavailable": Status.UNAVAILABLE}[kind]
                result = ProbeResult(status, None, None, len(times), None, msg[1])
        elif not proc.is_alive():
            result = ProbeResult(Status.ABORTED, None, None, len(times), None,
                                 f"process exited with code {proc.exitcode}")
    if proc.is_alive():
        proc.kill()
    proc.join(timeout=5)
    parent.close()
    return result


DEFAULT_BOARDS = [
    GameConfig(3, 3, 3),
    GameConfig(4, 4, 4),
    GameConfig(5, 5, 4),
    GameConfig(6, 6, 4),
    GameConfig(6, 7, 4, gravity=True),
    GameConfig(7, 7, 5),
    GameConfig(9, 9, 5),
    GameConfig(11, 11, 5),
    GameConfig(15, 15, 5),
    GameConfig(19, 19, 5),
]


def feasibility_map(
    specs: Sequence[str],
    boards: Sequence[GameConfig] = DEFAULT_BOARDS,
    budget: Budget | None = None,
    measure: bool = False,
    probe_moves: int = 2,
    verify_frontier: bool = False,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Assess every agent on every board; optionally confirm by probing.

    With ``measure``, cells the estimate accepts (ok or degraded) are probed,
    and a probe that overruns turns the cell into ``aborted``. With
    ``verify_frontier``, the first board each agent refuses for *time* is
    probed anyway (``force``) to check the refusal; the outcome is stored
    under ``frontier_probe`` without changing the cell's status.
    """
    budget = budget or Budget()
    cells = []
    for spec in specs:
        for cfg in boards:
            rep = assess(spec, cfg, budget)
            cell = {
                "agent": spec,
                "config": cfg.label,
                "cells": cfg.size,
                "status": rep.status.value,
                "estimated_status": rep.status.value,
                "reason": rep.reason,
                "estimate": rep.estimate.to_dict(),
                "degraded_params": rep.degraded_params,
                "measured": False,
            }
            if measure and rep.status in (Status.OK, Status.DEGRADED):
                pr = probe(spec, cfg, budget, moves=probe_moves)
                cell["measured"] = True
                cell["probe"] = pr.to_dict()
                if pr.status in (Status.ABORTED, Status.REFUSED, Status.UNAVAILABLE):
                    cell["status"] = pr.status.value
                    cell["reason"] = pr.reason
            if (verify_frontier and rep.status == Status.REFUSED and "time/move" in rep.reason
                    and not any(c["agent"] == spec and "frontier_probe" in c for c in cells)):
                pr = probe(spec, cfg, budget, moves=1, force=True)
                cell["frontier_probe"] = pr.to_dict()
                cell["measured"] = True
            if log:
                extra = f" (forced probe: {cell['frontier_probe']['status']})" if "frontier_probe" in cell else ""
                log(f"{spec:24s} {cfg.label:18s} {cell['status']:9s} {cell['reason']}{extra}")
            cells.append(cell)
    return {
        "agents": list(specs),
        "boards": [b.label for b in boards],
        "budget": budget.to_dict(),
        "cells": cells,
    }

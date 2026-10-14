"""Learning curves for agents that improve with experience."""

from __future__ import annotations

import time
from collections.abc import Callable

from ..agents import make_agent
from ..agents.base import Agent
from ..game import GameConfig, get_game
from ..metrics import standard_evaluation
from ..tournament import play_game


def train(
    agent: Agent,
    config: GameConfig,
    n_games: int,
    opponent: Agent | str = "random",
    sides: str = "both",
    game_offset: int = 0,
) -> None:
    """Train ``agent`` for ``n_games`` games.

    ``opponent`` is an agent, an agent spec string, or ``"self"`` for self-play;
    several specs joined by ``+`` (e.g. ``"random+minimax"``) take turns.
    ``sides`` is ``"both"`` (alternate X and O), ``"x"`` or ``"o"``.
    """
    game = get_game(config)
    if isinstance(opponent, str):
        opponents = [
            agent if spec == "self" else make_agent(spec, seed=agent.rng.randrange(2**31))
            for spec in opponent.split("+")
        ]
    else:
        opponents = [opponent]
    agent.training = True
    for i in range(n_games):
        idx = game_offset + i
        opponent = opponents[idx % len(opponents)]
        if len(opponents) > 1:
            idx //= len(opponents)
        as_x = sides == "x" or (sides == "both" and idx % 2 == 0)
        if opponent is agent:
            play_game(game, agent, agent, "self#x", "self#o")
        elif as_x:
            play_game(game, agent, opponent, "agent", "opp")
        else:
            play_game(game, opponent, agent, "opp", "agent")


def table_size(agent: Agent) -> int:
    for attr in ("boxes", "q", "values"):
        table = getattr(agent, attr, None)
        if isinstance(table, dict):
            return len(table)
    return 0


def learning_curve(
    agent_spec: str,
    config: GameConfig,
    total_games: int = 2000,
    eval_every: int = 200,
    eval_games: int = 40,
    opponent: str = "random",
    sides: str = "both",
    seed: int = 0,
    agent_factory: Callable[[], Agent] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Train an agent and evaluate it at regular checkpoints.

    Each checkpoint records W/D/L vs random and vs perfect play, the
    optimal-move rate, the table size and the wall time spent training.
    """
    agent = agent_factory() if agent_factory else make_agent(agent_spec, seed=seed)
    checkpoints = []
    trained = 0
    train_time = 0.0
    eval_points = [0] + list(range(eval_every, total_games + 1, eval_every))
    if eval_points[-1] != total_games:
        eval_points.append(total_games)
    for point in eval_points:
        if point > trained:
            t0 = time.perf_counter()
            train(agent, config, point - trained, opponent, sides, game_offset=trained)
            train_time += time.perf_counter() - t0
            trained = point
        ev = standard_evaluation(agent, config, eval_games, seed=seed * 1000 + point)
        ev.update(
            {
                "games": trained,
                "train_time": train_time,
                "table_size": table_size(agent),
                "memory_bytes": agent.memory_bytes(),
            }
        )
        checkpoints.append(ev)
        if progress:
            vm = ev.get("vs_minimax", {})
            progress(
                f"{agent_spec} {trained:>6} games: vs random win {ev['vs_random']['win']:.2f}, "
                f"vs minimax non-loss {1 - vm.get('loss', float('nan')):.2f}, "
                f"optimal {ev.get('optimal_rate', float('nan')):.2f}, table {ev['table_size']}"
            )
    return {
        "agent": agent_spec,
        "config": config.to_dict(),
        "label": config.label,
        "opponent": opponent,
        "sides": sides,
        "seed": seed,
        "eval_games": eval_games,
        "checkpoints": checkpoints,
        "games_to_learn": games_to_learn(checkpoints),
    }


def games_to_learn(checkpoints: list[dict], threshold: float = 0.95) -> int | None:
    """First checkpoint from which the agent never loses more than
    ``1 - threshold`` of its games against perfect play (None if never)."""
    ok_from = None
    for cp in checkpoints:
        vm = cp.get("vs_minimax")
        if vm is None:
            return None
        if 1 - vm["loss"] >= threshold:
            if ok_from is None:
                ok_from = cp["games"]
        else:
            ok_from = None
    return ok_from

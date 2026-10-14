"""Strength metrics: W/D/L against reference opponents and the optimal-move
rate against perfect play."""

from __future__ import annotations

from contextlib import contextmanager

from .agents.base import Agent
from .agents.minimax import MinimaxAgent, Solver, shared_solver
from .agents.random_agent import RandomAgent
from .game import Game, GameConfig, State, get_game
from .tournament import play_game, play_match


@contextmanager
def frozen(agent: Agent):
    """Temporarily switch a learning agent to evaluation mode."""
    was = agent.training
    agent.training = False
    try:
        yield agent
    finally:
        agent.training = was


def wdl(agent: Agent, opponent: Agent, config: GameConfig, n_games: int, seed: int = 0) -> dict:
    """Win/draw/loss rates of ``agent`` vs ``opponent`` (alternating sides)."""
    r = play_match(config, agent, opponent, n_games, seed=seed, a_name="agent", b_name="opp")
    n = max(1, r.n_games)
    return {
        "win": r.wins_a / n,
        "draw": r.draws / n,
        "loss": r.wins_b / n,
        "games": r.n_games,
        "aborted": r.aborted,
        "as_x": r.a_as_x,
        "as_o": r.a_as_o,
        "time_per_move": r.time_per_move.get("agent", 0.0),
    }


def optimal_move_rate(
    agent: Agent,
    config: GameConfig,
    n_games: int = 50,
    opponent: Agent | None = None,
    seed: int = 0,
    solver: Solver | None = None,
) -> dict:
    """Fraction of ``agent``'s moves that preserve the game-theoretic outcome.

    Positions come from games against ``opponent`` (random by default, so a
    wide range of positions is visited), with the agent alternating sides.
    Needs a board small enough for :class:`Solver`.
    """
    game = get_game(config)
    solver = solver or shared_solver(config)
    opponent = opponent or RandomAgent()
    agent.seed(seed * 7919 + 11)
    opponent.seed(seed * 7919 + 13)
    was_training, agent.training = agent.training, False
    good = total = 0
    # Positions where the choice matters (some moves would change the outcome).
    critical_good = critical_total = 0

    for i in range(n_games):
        agent_side = 1 if i % 2 == 0 else 2

        def check(state: State, move: int, side=agent_side) -> None:
            nonlocal good, total, critical_good, critical_total
            if state.to_move != side:
                return
            ok_moves = solver.outcome_preserving_moves(state)
            hit = move in ok_moves
            total += 1
            good += hit
            if len(ok_moves) < len(state.legal_moves()):
                critical_total += 1
                critical_good += hit

        if agent_side == 1:
            play_game(game, agent, opponent, "agent", "opp", on_move=check)
        else:
            play_game(game, opponent, agent, "opp", "agent", on_move=check)
    agent.training = was_training
    return {
        "optimal_rate": good / total if total else float("nan"),
        "critical_optimal_rate": critical_good / critical_total if critical_total else float("nan"),
        "moves": total,
        "critical_moves": critical_total,
    }


def standard_evaluation(
    agent: Agent,
    config: GameConfig,
    n_games: int = 50,
    seed: int = 0,
    perfect: bool | None = None,
) -> dict:
    """W/D/L vs random and (when solvable) vs perfect play, plus optimal-move rate."""
    game: Game = get_game(config)
    out: dict = {}
    with frozen(agent):
        out["vs_random"] = wdl(agent, RandomAgent(), config, n_games, seed)
        if perfect is None:
            perfect = game.size <= 16
        if perfect:
            out["vs_minimax"] = wdl(agent, MinimaxAgent(), config, n_games, seed + 1)
            out.update(optimal_move_rate(agent, config, max(10, n_games // 2), seed=seed + 2))
    return out

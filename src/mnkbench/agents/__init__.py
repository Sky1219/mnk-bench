"""Agent registry: build agents from short specs like ``"mcts:iterations=500"``."""

from __future__ import annotations

import importlib
from typing import Any

from .base import (
    Agent,
    AgentRefused,
    Budget,
    BudgetExceeded,
    CostEstimate,
    FeasibilityReport,
    Status,
)

# name -> (module, class); imported lazily so optional deps stay optional.
REGISTRY: dict[str, tuple[str, str]] = {
    "random": ("mnkbench.agents.random_agent", "RandomAgent"),
    "minimax": ("mnkbench.agents.minimax", "MinimaxAgent"),
    "menace": ("mnkbench.agents.menace", "MenaceAgent"),
    "rule": ("mnkbench.agents.rule_based", "RuleBasedAgent"),
    "qlearning": ("mnkbench.agents.tabular", "QLearningAgent"),
    "td": ("mnkbench.agents.tabular", "TDAgent"),
    "alphabeta": ("mnkbench.agents.alphabeta", "AlphaBetaAgent"),
    "mcts": ("mnkbench.agents.mcts", "MCTSAgent"),
    "human": ("mnkbench.agents.human", "HumanAgent"),
}

DESCRIPTIONS: dict[str, str] = {
    "random": "uniformly random legal moves",
    "minimax": "perfect play: full alpha-beta search with transposition table",
    "menace": "Michie's matchbox learner (bead reinforcement)",
    "rule": "win / block / fork / block-fork rules, then line potential",
    "qlearning": "tabular Q-learning with epsilon-greedy exploration",
    "td": "TD(0) afterstate value learning (Sutton & Barto)",
    "alphabeta": "depth-limited alpha-beta with a threat-based evaluation",
    "mcts": "Monte Carlo Tree Search (UCT, random playouts)",
    "human": "you, at the terminal",
}


def agent_class(name: str) -> type[Agent]:
    if name not in REGISTRY:
        raise KeyError(f"unknown agent {name!r}; choose from {', '.join(sorted(REGISTRY))}")
    module, cls = REGISTRY[name]
    return getattr(importlib.import_module(module), cls)


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def parse_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``"mcts:iterations=500,c=1.4"`` -> ``("mcts", {"iterations": 500, "c": 1.4})``."""
    name, _, rest = spec.partition(":")
    kwargs: dict[str, Any] = {}
    for part in filter(None, (p.strip() for p in rest.split(","))):
        key, eq, value = part.partition("=")
        if not eq:
            raise ValueError(f"bad parameter {part!r} in agent spec {spec!r}")
        kwargs[key.strip()] = _coerce(value.strip())
    return name.strip(), kwargs


def make_agent(spec: str, seed: int | None = None, **overrides: Any) -> Agent:
    """Build an agent from a spec string, e.g. ``make_agent("alphabeta:depth=3")``."""
    name, kwargs = parse_spec(spec)
    kwargs.update(overrides)
    agent = agent_class(name)(seed=seed, **kwargs)
    if ":" in spec:
        agent.name = spec
    return agent


__all__ = [
    "Agent",
    "AgentRefused",
    "Budget",
    "BudgetExceeded",
    "CostEstimate",
    "FeasibilityReport",
    "Status",
    "REGISTRY",
    "DESCRIPTIONS",
    "agent_class",
    "make_agent",
    "parse_spec",
]

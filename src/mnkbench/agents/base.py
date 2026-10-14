"""The common Agent interface, budgets and feasibility reports.

Every agent, from random play to an LLM, implements :meth:`Agent.select_move`.
Learning agents additionally use :meth:`Agent.new_game` and
:meth:`Agent.end_game` to see whole games.

Every agent can also *estimate* what it would cost to play a given board
(:meth:`Agent.estimate`) and compare that against a :class:`Budget`
(:meth:`Agent.feasibility`). The result says whether the agent can run as
configured (``ok``), can run only in a cheaper configuration (``degraded``),
or should not run at all (``refused``). If an agent runs anyway and blows its
budget at play time it raises :class:`BudgetExceeded`, which the tournament
runner records as an aborted game instead of crashing.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from ..game import Game, GameConfig, State, get_game

MB = 1024 * 1024
GB = 1024 * MB


class BudgetExceeded(RuntimeError):
    """Raised by an agent that runs out of time or memory mid-game."""


class AgentRefused(RuntimeError):
    """Raised when an agent is asked to play a board it has judged infeasible."""


@dataclass
class Budget:
    """Resource limits an agent must respect.

    ``time_per_move`` is in seconds and ``memory_bytes`` in bytes.
    ``train_time`` bounds up-front learning (MENACE, Q-learning, AlphaZero...).
    ``cost_per_move`` bounds money (USD) spent per move, for LLM agents.
    """

    time_per_move: float = 1.0
    memory_bytes: float = 1 * GB
    train_time: float = 600.0
    cost_per_move: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)


class Status(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    REFUSED = "refused"
    ABORTED = "aborted"  # ran, but blew its budget at play time
    UNAVAILABLE = "unavailable"  # missing dependency or credentials


@dataclass
class CostEstimate:
    """Predicted resource use for one agent on one board."""

    time_per_move: float  # seconds
    memory_bytes: float
    train_time: float = 0.0  # seconds of learning before play
    cost_per_move: float = 0.0  # USD
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FeasibilityReport:
    agent: str
    config: str
    status: Status
    estimate: CostEstimate
    reason: str = ""
    degraded_params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class Agent:
    """Base class for all agents.

    Subclasses must implement :meth:`select_move` and usually override
    :meth:`estimate`. ``name`` is a short identifier used in tables and plots.
    """

    name = "agent"
    #: Whether the agent improves through :meth:`end_game` (MENACE, Q, TD...).
    learns = False

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.training = False
        self.stats: dict[str, float] = {}

    # ------------------------------------------------------------ life cycle
    def seed(self, seed: int | None) -> None:
        self.rng.seed(seed)

    def new_game(self, game: Game, player: int) -> None:
        """Called before each game with the side this agent plays."""

    def select_move(self, state: State) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def end_game(self, state: State, player: int) -> None:
        """Called after each game with the final position (learning hook)."""

    # ---------------------------------------------------------- introspection
    def params(self) -> dict[str, Any]:
        """Constructor parameters, for reports and for rebuilding degraded variants."""
        return {}

    def memory_bytes(self) -> float:
        """Approximate memory currently held (tables, trees, networks)."""
        return 0.0

    def describe(self) -> str:
        p = ", ".join(f"{k}={v}" for k, v in self.params().items())
        return f"{self.name}({p})" if p else self.name

    def add_stat(self, key: str, value: float) -> None:
        self.stats[key] = self.stats.get(key, 0.0) + value

    # ------------------------------------------------------------ feasibility
    def estimate(self, config: GameConfig) -> CostEstimate:
        """Predict time per move and memory for ``config``. Cheap by default."""
        size = config.size
        return CostEstimate(time_per_move=2e-6 * size, memory_bytes=64.0 * size)

    def feasibility(self, config: GameConfig, budget: Budget) -> FeasibilityReport:
        """Compare :meth:`estimate` against ``budget``.

        Subclasses that can degrade (shallower search, fewer simulations)
        override :meth:`degrade` to propose cheaper parameters.
        """
        est = self.estimate(config)
        problems = budget_problems(est, budget)
        if not problems:
            return FeasibilityReport(self.name, config.label, Status.OK, est)
        cheaper = self.degrade(config, budget)
        if cheaper is not None:
            params, est2 = cheaper
            return FeasibilityReport(
                self.name,
                config.label,
                Status.DEGRADED,
                est2,
                reason="; ".join(problems),
                degraded_params=params,
            )
        return FeasibilityReport(self.name, config.label, Status.REFUSED, est, reason="; ".join(problems))

    def degrade(self, config: GameConfig, budget: Budget) -> tuple[dict, CostEstimate] | None:
        """Return ``(params, estimate)`` for a cheaper configuration that fits the
        budget, or None if there is none."""
        return None

    def adapt(self, config: GameConfig, budget: Budget) -> Agent:
        """Return an agent that fits ``budget`` on ``config``: ``self`` if it
        already fits, a degraded copy if one exists, else raise AgentRefused."""
        report = self.feasibility(config, budget)
        if report.status == Status.OK:
            return self
        if report.status == Status.DEGRADED:
            params = {**self.params(), **report.degraded_params}
            clone = type(self)(**params)
            clone.name = self.name
            return clone
        raise AgentRefused(f"{self.name} refuses {config.label}: {report.reason}")

    def __repr__(self) -> str:
        return self.describe()


def budget_problems(est: CostEstimate, budget: Budget) -> list[str]:
    """Human-readable list of the ways ``est`` breaks ``budget``."""
    problems = []
    if est.time_per_move > budget.time_per_move:
        problems.append(f"time/move {fmt_seconds(est.time_per_move)} > {fmt_seconds(budget.time_per_move)}")
    if est.memory_bytes > budget.memory_bytes:
        problems.append(f"memory {fmt_bytes(est.memory_bytes)} > {fmt_bytes(budget.memory_bytes)}")
    if est.train_time > budget.train_time:
        problems.append(f"training {fmt_seconds(est.train_time)} > {fmt_seconds(budget.train_time)}")
    if est.cost_per_move > budget.cost_per_move:
        problems.append(f"cost/move ${est.cost_per_move:.4f} > ${budget.cost_per_move:.4f}")
    return problems


def fmt_seconds(s: float) -> str:
    if not math.isfinite(s):
        return "inf"
    if s < 1e-3:
        return f"{s * 1e6:.0f}us"
    if s < 1:
        return f"{s * 1e3:.0f}ms"
    if s < 120:
        return f"{s:.1f}s"
    if s < 7200:
        return f"{s / 60:.0f}min"
    if s < 86400 * 365:
        return f"{s / 3600:.0f}h"
    return f"{s / (86400 * 365):.1e}y"


def fmt_bytes(b: float) -> str:
    if not math.isfinite(b):
        return "inf"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024:
            return f"{b:.0f}{unit}"
        b /= 1024
    return f"{b:.1e}PB"


def as_game(config_or_game: Game | GameConfig) -> Game:
    return config_or_game if isinstance(config_or_game, Game) else get_game(config_or_game)

"""Tabular reinforcement learners: Q-learning and TD(0) afterstate learning.

Both store one entry per (canonical) position, like MENACE, so they share its
scaling problem: the table grows with the number of reachable positions.

* :class:`QLearningAgent` learns action values Q(s, a) with the standard
  off-policy update. A transition runs from one of the agent's decision
  points to its next one (the opponent's reply is part of the environment),
  so the same code works against a fixed opponent and in self-play.
* :class:`TDAgent` learns state values V(s') of *afterstates* (the position
  right after the agent moves), as in Sutton & Barto's tic-tac-toe example.
  V is the probability of winning (draw = 0.5) from the mover's view.
"""

from __future__ import annotations

from ..complexity import position_bound, symmetry_count
from ..game import DRAW, Game, GameConfig, State
from ..symmetry import Canonicalizer
from .base import Agent, CostEstimate

BYTES_PER_ENTRY = 350


class _Tabular(Agent):
    learns = True

    def __init__(self, seed: int | None = None, use_symmetry: bool = True, alpha: float = 0.3,
                 epsilon: float = 0.1, epsilon_decay: float = 1.0, min_epsilon: float = 0.01):
        super().__init__(seed)
        self.use_symmetry = use_symmetry
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.training = True
        self._game: Game | None = None
        self._canon: Canonicalizer | None = None
        self.games_played = 0

    def params(self) -> dict:
        return {"use_symmetry": self.use_symmetry, "alpha": self.alpha, "epsilon": self.epsilon,
                "epsilon_decay": self.epsilon_decay, "min_epsilon": self.min_epsilon}

    def _setup(self, game: Game) -> None:
        if self._game is None or self._game.config != game.config:
            self._game = game
            self._canon = Canonicalizer(game, self.use_symmetry)
            self._reset_table()

    def _reset_table(self) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _explore(self) -> bool:
        return self.training and self.rng.random() < self.epsilon

    def _decay(self) -> None:
        self.games_played += 1
        if self.training and self.epsilon_decay < 1.0:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def estimate(self, config: GameConfig) -> CostEstimate:
        entries = position_bound(config) / (symmetry_count(config) if self.use_symmetry else 1)
        games = 20 * entries / max(1.0, config.size / 2)
        return CostEstimate(
            time_per_move=config.size * 3e-6 * (1 + 8 * self.use_symmetry),
            memory_bytes=entries * (BYTES_PER_ENTRY + config.size * 4),
            train_time=games * config.size * 3e-5,
            notes=f"up to {entries:.2g} table entries",
        )


def _reward(state: State, player: int) -> float:
    if state.winner == DRAW:
        return 0.0
    return 1.0 if state.winner == player else -1.0


class QLearningAgent(_Tabular):
    """Q-learning with epsilon-greedy exploration; rewards +1 / 0 / -1."""

    name = "qlearning"

    def __init__(self, seed: int | None = None, use_symmetry: bool = True, alpha: float = 0.3,
                 gamma: float = 0.95, epsilon: float = 0.2, epsilon_decay: float = 0.9995,
                 min_epsilon: float = 0.02, init_q: float = 0.0):
        super().__init__(seed, use_symmetry, alpha, epsilon, epsilon_decay, min_epsilon)
        self.gamma = gamma
        self.init_q = init_q
        self.q: dict[bytes, dict[int, float]] = {}
        self._pending: dict[int, tuple[bytes, int] | None] = {1: None, 2: None}

    def params(self) -> dict:
        return {**super().params(), "gamma": self.gamma, "init_q": self.init_q}

    def _reset_table(self) -> None:
        self.q = {}

    def new_game(self, game: Game, player: int) -> None:
        self._setup(game)
        self._pending = {1: None, 2: None}

    def _row(self, key: bytes, state: State, sym: int) -> dict[int, float]:
        row = self.q.get(key)
        if row is None:
            canon = self._canon
            moves = [canon.to_canon(sym, mv) for mv in state.legal_moves()]
            if self.use_symmetry:
                moves = canon.move_orbit_reps(key, moves)
            row = {mv: self.init_q for mv in moves}
            if self.training:
                self.q[key] = row
        return row

    def select_move(self, state: State) -> int:
        self._setup(state.game)
        canon = self._canon
        key, sym = canon.canonical(state.cells)
        row = self._row(key, state, sym)
        player = state.to_move
        best = max(row.values())
        if self.training:
            pending = self._pending.get(player)
            if pending is not None:
                pk, pm = pending
                prow = self.q[pk]
                prow[pm] += self.alpha * (self.gamma * best - prow[pm])
        if self._explore():
            cm = self.rng.choice(list(row))
        else:
            cm = self.rng.choice([m for m, v in row.items() if v >= best - 1e-12])
        if self.training:
            self._pending[player] = (key, cm)
        return canon.from_canon(sym, cm)

    def end_game(self, state: State, player: int) -> None:
        pending = self._pending.get(player)
        if self.training and pending is not None:
            pk, pm = pending
            prow = self.q[pk]
            prow[pm] += self.alpha * (_reward(state, player) - prow[pm])
        self._pending[player] = None
        if player == 2:  # end_game runs for X then O: decay once per game
            self._decay()

    def memory_bytes(self) -> float:
        size = self._game.size if self._game else 9
        return sum(BYTES_PER_ENTRY + size + 40 * len(r) for r in self.q.values())


class TDAgent(_Tabular):
    """TD(0) on afterstate values, greedy with epsilon exploration."""

    name = "td"

    def __init__(self, seed: int | None = None, use_symmetry: bool = True, alpha: float = 0.3,
                 epsilon: float = 0.2, epsilon_decay: float = 0.9995, min_epsilon: float = 0.02,
                 init_value: float = 0.5):
        super().__init__(seed, use_symmetry, alpha, epsilon, epsilon_decay, min_epsilon)
        self.init_value = init_value
        self.values: dict[bytes, float] = {}
        self._pending: dict[int, bytes | None] = {1: None, 2: None}

    def params(self) -> dict:
        return {**super().params(), "init_value": self.init_value}

    def _reset_table(self) -> None:
        self.values = {}

    def new_game(self, game: Game, player: int) -> None:
        self._setup(game)
        self._pending = {1: None, 2: None}

    def _afterstate_value(self, state: State, player: int) -> tuple[float, bytes | None]:
        """Value (for ``player``, who just moved) of the position in ``state``."""
        if state.winner is not None:
            return (0.5 if state.winner == DRAW else 1.0 if state.winner == player else 0.0), None
        key, _ = self._canon.canonical(state.cells)
        return self.values.get(key, self.init_value), key

    def select_move(self, state: State) -> int:
        self._setup(state.game)
        player = state.to_move
        work = state.copy()
        scored = []
        for mv in state.legal_moves():
            work.push(mv)
            v, key = self._afterstate_value(work, player)
            work.pop()
            scored.append((v, mv, key))
        if self._explore():
            v, mv, key = self.rng.choice(scored)
        else:
            best = max(s[0] for s in scored)
            v, mv, key = self.rng.choice([s for s in scored if s[0] >= best - 1e-12])
        if self.training:
            prev = self._pending.get(player)
            if prev is not None:
                old = self.values.get(prev, self.init_value)
                self.values[prev] = old + self.alpha * (v - old)
            self._pending[player] = key
            if key is not None and key not in self.values:
                self.values[key] = self.init_value
        return mv

    def end_game(self, state: State, player: int) -> None:
        prev = self._pending.get(player)
        if self.training and prev is not None:
            r = 0.5 if state.winner == DRAW else 1.0 if state.winner == player else 0.0
            old = self.values.get(prev, self.init_value)
            self.values[prev] = old + self.alpha * (r - old)
        self._pending[player] = None
        if player == 2:  # end_game runs for X then O: decay once per game
            self._decay()

    def memory_bytes(self) -> float:
        size = self._game.size if self._game else 9
        return len(self.values) * (BYTES_PER_ENTRY + size)

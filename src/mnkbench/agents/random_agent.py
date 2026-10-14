"""Uniformly random play: the weakest baseline."""

from __future__ import annotations

from ..game import GameConfig, State
from .base import Agent, CostEstimate


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None):
        super().__init__(seed)

    def select_move(self, state: State) -> int:
        return self.rng.choice(state.legal_moves())

    def estimate(self, config: GameConfig) -> CostEstimate:
        return CostEstimate(time_per_move=1e-6 * config.size + 5e-6, memory_bytes=16.0 * config.size)

"""A human player at the terminal (for ``mnk-bench play``)."""

from __future__ import annotations

import sys
from collections.abc import Callable

from ..game import GameConfig, State
from .base import Agent, CostEstimate


class HumanAgent(Agent):
    name = "human"

    def __init__(self, seed: int | None = None, input_fn: Callable[[str], str] | None = None,
                 output_fn: Callable[[str], None] | None = None):
        super().__init__(seed)
        self.input_fn = input_fn or input
        self.output_fn = output_fn or (lambda s: print(s, file=sys.stderr))

    def select_move(self, state: State) -> int:
        game = state.game
        self.output_fn(state.render())
        while True:
            text = self.input_fn(f"{'XO'[state.to_move - 1]} to move (e.g. {game.cell_name(state.legal_moves()[0])}): ")
            try:
                mv = game.parse_cell(text)
            except ValueError as exc:
                self.output_fn(str(exc))
                continue
            if game.gravity and not state.is_legal(mv):
                # Under gravity, accept any cell in the column.
                col = mv % game.n
                legal = [m for m in state.legal_moves() if m % game.n == col]
                if legal:
                    return legal[0]
            if state.is_legal(mv):
                return mv
            self.output_fn("that cell is not a legal move")

    def estimate(self, config: GameConfig) -> CostEstimate:
        return CostEstimate(time_per_move=10.0, memory_bytes=0.0, notes="human thinking time")

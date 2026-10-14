"""MENACE: Donald Michie's Machine Educable Noughts And Crosses Engine (1961).

One "matchbox" per position the agent has to move from, holding coloured
"beads", one colour per legal move. To move, MENACE draws a bead at random
(so moves with more beads are likelier). After the game it reinforces every
move it made: beads are added after a win or draw and removed after a loss.

With ``use_symmetry=True`` positions are stored in canonical form and a box
only holds one bead colour per *distinct* move (moves that are equivalent
under the position's own symmetries share a colour), exactly as Michie did.
With ``skip_forced=True`` positions with a single distinct move get no box.
For X on tic-tac-toe that leaves 304 possible matchboxes (see
:func:`count_matchboxes`), matching Michie's machine.

This generalises to any m,n,k board and to playing either side; the bead
schedule ``initial_beads`` gives the starting beads per move as a function of
how many stones are on the board (Michie used 4, 3, 2, 1 for X's moves 1-4).
"""

from __future__ import annotations

from ..complexity import position_bound, symmetry_count
from ..game import DRAW, Game, GameConfig, State, get_game
from ..symmetry import Canonicalizer
from .base import Agent, CostEstimate

BYTES_PER_BOX = 400  # dict entry + bytes key + small dict of beads


class MenaceAgent(Agent):
    name = "menace"
    learns = True

    def __init__(
        self,
        seed: int | None = None,
        use_symmetry: bool = True,
        skip_forced: bool = True,
        max_beads: int = 4,
        win_reward: int = 3,
        draw_reward: int = 1,
        loss_penalty: int = 1,
        refill_empty: bool = True,
    ):
        super().__init__(seed)
        self.use_symmetry = use_symmetry
        self.skip_forced = skip_forced
        self.max_beads = max_beads
        self.win_reward = win_reward
        self.draw_reward = draw_reward
        self.loss_penalty = loss_penalty
        self.refill_empty = refill_empty
        self.training = True
        self.boxes: dict[bytes, dict[int, int]] = {}
        self._canon: Canonicalizer | None = None
        self._game: Game | None = None
        # Moves made this game, per player: list of (box key, canonical move).
        self._trail: dict[int, list[tuple[bytes, int]]] = {1: [], 2: []}

    def params(self) -> dict:
        return {
            "use_symmetry": self.use_symmetry,
            "skip_forced": self.skip_forced,
            "max_beads": self.max_beads,
            "win_reward": self.win_reward,
            "draw_reward": self.draw_reward,
            "loss_penalty": self.loss_penalty,
            "refill_empty": self.refill_empty,
        }

    def _setup(self, game: Game) -> None:
        if self._game is None or self._game.config != game.config:
            self._game = game
            self._canon = Canonicalizer(game, self.use_symmetry)
            self.boxes.clear()

    def new_game(self, game: Game, player: int) -> None:
        self._setup(game)
        self._trail = {1: [], 2: []}

    def initial_beads(self, stones: int) -> int:
        return max(1, self.max_beads - stones // 2)

    def _box_moves(self, key: bytes, state: State, sym: int) -> list[int]:
        canon = self._canon
        moves = [canon.to_canon(sym, mv) for mv in state.legal_moves()]
        return canon.move_orbit_reps(key, moves) if self.use_symmetry else moves

    def select_move(self, state: State) -> int:
        self._setup(state.game)
        canon = self._canon
        key, sym = canon.canonical(state.cells)
        box = self.boxes.get(key)
        if not self.training:
            self.add_stat("lookups", 1)
            self.add_stat("hits", box is not None)
        if box is None:
            moves = self._box_moves(key, state, sym)
            if self.skip_forced and len(moves) == 1:
                return canon.from_canon(sym, moves[0])
            if not self.training:
                # Unseen position at evaluation time: no box, so play uniformly.
                return canon.from_canon(sym, self.rng.choice(moves))
            beads = self.initial_beads(state.move_count)
            box = {mv: beads for mv in moves}
            self.boxes[key] = box
        total = sum(box.values())
        if total <= 0:
            if self.refill_empty:
                beads = self.initial_beads(state.move_count)
                for mv in box:
                    box[mv] = beads
                total = beads * len(box)
            else:
                # A resigning MENACE would give up here; we play uniformly instead.
                cm = self.rng.choice(list(box))
                self._trail[state.to_move].append((key, cm))
                return canon.from_canon(sym, cm)
        if self.training:
            pick = self.rng.random() * total
            for cm, beads in box.items():
                pick -= beads
                if pick < 0:
                    break
        else:
            # Greedy at evaluation time: most beads, random tie-break.
            top = max(box.values())
            cm = self.rng.choice([m for m, b in box.items() if b == top])
        self._trail[state.to_move].append((key, cm))
        return canon.from_canon(sym, cm)

    def end_game(self, state: State, player: int) -> None:
        trail = self._trail.get(player, [])
        if self.training and trail:
            if state.winner == DRAW:
                delta = self.draw_reward
            elif state.winner == player:
                delta = self.win_reward
            else:
                delta = -self.loss_penalty
            for key, cm in trail:
                box = self.boxes[key]
                box[cm] = max(0, box[cm] + delta)
        self._trail[player] = []

    @property
    def num_boxes(self) -> int:
        return len(self.boxes)

    def memory_bytes(self) -> float:
        size = self._game.size if self._game else 9
        return sum(BYTES_PER_BOX + size + 16 * len(b) for b in self.boxes.values())

    def estimate(self, config: GameConfig) -> CostEstimate:
        boxes = position_bound(config) / (symmetry_count(config) if self.use_symmetry else 1)
        # Learning needs every box visited many times; assume ~10 visits per box
        # at ~2 boxes per game per player and ~50us per game move.
        games = 10 * boxes / max(1.0, config.size / 2)
        return CostEstimate(
            time_per_move=1e-5 * (1 + 8 * self.use_symmetry) * (1 + config.size / 50),
            memory_bytes=boxes * (BYTES_PER_BOX + config.size * 5),
            train_time=games * config.size * 2e-5,
            notes=f"up to {boxes:.2g} matchboxes",
        )


def count_matchboxes(config: GameConfig, player: int = 1, use_symmetry: bool = True, skip_forced: bool = True) -> int:
    """Count the matchboxes MENACE would need to play ``player`` against every
    possible opponent: all non-terminal positions with ``player`` to move,
    reachable in legal play, up to symmetry, excluding forced moves."""
    game = get_game(config)
    canon = Canonicalizer(game, use_symmetry)
    seen: set[bytes] = set()
    boxes: set[bytes] = set()
    state = game.new_state()

    def visit() -> None:
        key, sym = canon.canonical(state.cells)
        if key in seen:
            return
        seen.add(key)
        if state.winner is not None:
            return
        if state.to_move == player:
            moves = [canon.to_canon(sym, mv) for mv in state.legal_moves()]
            reps = canon.move_orbit_reps(key, moves) if use_symmetry else moves
            if not (skip_forced and len(reps) == 1):
                boxes.add(key)
        for mv in state.legal_moves():
            state.push(mv)
            visit()
            state.pop()

    visit()
    return len(boxes)


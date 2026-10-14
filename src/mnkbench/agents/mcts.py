"""Sampling: Monte Carlo Tree Search with UCT and random playouts.

Each iteration walks down the tree choosing children by the UCB1 formula,
adds one new node, plays the game out at random, and backs the result up.
After ``iterations`` iterations or ``time_limit`` seconds, whichever comes
first (``iterations=None`` means time only), it plays the most visited move. No knowledge of the game beyond its rules is needed, which is
what makes MCTS general, and random playouts are also why it weakens on big
boards: a random game of gomoku says very little about who is better.

Two optional, cheap pieces of game knowledge (on by default) make it much
more robust: ``tactical`` takes immediate wins and blocks immediate losses at
the root, and ``radius`` restricts tree moves to cells near existing stones
on big boards (rollouts still use the whole board).
"""

from __future__ import annotations

import math
import time

from ..game import DRAW, EMPTY, GameConfig, State
from ..playout import random_playout
from .base import Agent, Budget, CostEstimate

BYTES_PER_NODE = 220


class Node:
    __slots__ = ("move", "parent", "children", "untried", "visits", "wins", "player")

    def __init__(self, move: int | None, parent: Node | None, untried: list[int], player: int):
        self.move = move
        self.parent = parent
        self.children: list[Node] = []
        self.untried = untried
        self.visits = 0
        self.wins = 0.0  # from the point of view of ``player``, who made ``move``
        self.player = player


class MCTSAgent(Agent):
    name = "mcts"

    def __init__(
        self,
        seed: int | None = None,
        iterations: int = 1000,
        time_limit: float | None = None,
        c: float = 1.4,
        tactical: bool = True,
        radius: int | None = None,
    ):
        super().__init__(seed)
        self.iterations = iterations
        self.time_limit = time_limit
        self.c = c
        self.tactical = tactical
        self.radius = radius
        self.last_tree_size = 0

    def params(self) -> dict:
        return {"iterations": self.iterations, "time_limit": self.time_limit, "c": self.c,
                "tactical": self.tactical, "radius": self.radius}

    def _radius_for(self, state: State) -> int | None:
        if state.game.gravity:
            return None
        if self.radius is not None:
            return self.radius or None
        return 2 if state.game.size > 49 else None

    def _moves(self, state: State, radius: int | None) -> list[int]:
        moves = state.legal_moves()
        if radius is None or not state.history or not moves:
            return moves
        near = state.game.neighbors1 if radius <= 1 else state.game.neighbors2
        cells = state.cells
        cand = {c for mv in state.history for c in near[mv] if cells[c] == EMPTY}
        return list(cand) if cand else moves

    def select_move(self, state: State) -> int:
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]
        me = state.to_move
        if self.tactical:
            wins = [mv for mv in legal if state.would_win(mv, me)]
            if wins:
                return self.rng.choice(wins)
            blocks = [mv for mv in legal if state.would_win(mv, 3 - me)]
            if len(blocks) == 1:
                return blocks[0]
        if not state.history and not state.game.gravity and state.game.size > 49:
            return state.game.center_order[0]

        rng = self.rng
        radius = self._radius_for(state)
        root_moves = self._moves(state, radius)
        rng.shuffle(root_moves)
        root = Node(None, None, root_moves, 3 - me)
        c = self.c
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        max_iters = self.iterations if (self.iterations or deadline is None) else None
        if max_iters is None and deadline is None:
            max_iters = 1000
        log = math.log
        sqrt = math.sqrt
        it = 0
        while True:
            if max_iters is not None and it >= max_iters:
                break
            if deadline is not None and it and time.perf_counter() > deadline:
                break
            it += 1
            node = root
            sim = state.copy()
            # Selection
            while not node.untried and node.children:
                lnN = log(node.visits)
                best, best_v = None, -1.0
                for ch in node.children:
                    v = ch.wins / ch.visits + c * sqrt(lnN / ch.visits)
                    if v > best_v:
                        best, best_v = ch, v
                node = best
                sim.push(node.move)
            # Expansion
            if node.untried and sim.winner is None:
                untried = node.untried
                i = int(rng.random() * len(untried))
                untried[i], untried[-1] = untried[-1], untried[i]
                mv = untried.pop()
                player = sim.to_move
                sim.push(mv)
                untried = self._moves(sim, radius) if sim.winner is None else []
                child = Node(mv, node, untried, player)
                node.children.append(child)
                node = child
            # Rollout
            winner = random_playout(sim, rng) if sim.winner is None else sim.winner
            # Backpropagation
            while node is not None:
                node.visits += 1
                if winner == node.player:
                    node.wins += 1.0
                elif winner == DRAW:
                    node.wins += 0.5
                node = node.parent
        self.last_tree_size = _count(root)
        self.add_stat("iterations", it)
        self.add_stat("moves", 1)
        best = max(root.children, key=lambda ch: (ch.visits, ch.wins))
        return best.move

    def memory_bytes(self) -> float:
        return self.last_tree_size * BYTES_PER_NODE

    def apply_budget(self, budget: Budget) -> None:
        # Stop early (keeping the best move so far) if playouts are slower than estimated.
        if self.time_limit is None or self.time_limit > budget.time_per_move:
            self.time_limit = 0.9 * budget.time_per_move

    # ------------------------------------------------------------ feasibility
    @staticmethod
    def seconds_per_iteration(config: GameConfig) -> float:
        """Calibrated cost of one iteration (mostly the random playout):
        ~13us on (3,3,3), ~35us on (6,6,4), ~130us on (15,15,5)."""
        n = config.size
        return min(6e-6 + 0.75e-6 * n, 18e-6 + 0.5e-6 * n)

    def estimate(self, config: GameConfig) -> CostEstimate:
        per = self.seconds_per_iteration(config)
        iters = float(self.iterations) if self.iterations else float("inf")
        if self.time_limit:
            iters = min(iters, self.time_limit / per)
        if iters == float("inf"):
            iters = 1000.0
        t = iters * per
        return CostEstimate(time_per_move=t, memory_bytes=iters * (BYTES_PER_NODE + 8 * min(config.size, 50)),
                            notes=f"~{iters:.0f} playouts/move")

    def degrade(self, config: GameConfig, budget: Budget):
        per = self.seconds_per_iteration(config)
        iters = int(budget.time_per_move / per)
        if iters < 50:
            return None
        iters = min(iters, self.iterations or iters)
        est = CostEstimate(time_per_move=iters * per, memory_bytes=iters * (BYTES_PER_NODE + 8 * min(config.size, 50)),
                           notes=f"{iters} playouts/move")
        return {"iterations": iters}, est


def _count(node: Node) -> int:
    n, stack = 0, [node]
    while stack:
        cur = stack.pop()
        n += 1
        stack.extend(cur.children)
    return n

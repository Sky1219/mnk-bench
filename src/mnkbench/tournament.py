"""Seeded games, matches and round-robin tournaments.

Every random choice flows from the ``seed`` given to :func:`play_match` or
:func:`round_robin`, so re-running with the same seed reproduces the same
games (for agents that only use their own ``rng``).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from .agents.base import Agent, AgentRefused, BudgetExceeded
from .game import DRAW, O, X, Game, GameConfig, IllegalMoveError, State, get_game


@dataclass
class GameRecord:
    x: str
    o: str
    moves: list[int]
    winner: int | None  # X, O, DRAW, or None if aborted
    move_times: dict[str, list[float]] = field(default_factory=dict)
    aborted: str | None = None  # reason, if the game was cut short
    aborted_by: str | None = None

    @property
    def length(self) -> int:
        return len(self.moves)

    def score_for(self, name: str) -> float | None:
        """1 for a win, 0.5 for a draw, 0 for a loss, None if aborted."""
        if self.winner is None:
            return None
        if self.winner == DRAW:
            return 0.5
        winner_name = self.x if self.winner == X else self.o
        return 1.0 if winner_name == name else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def play_game(
    game: Game | GameConfig,
    agent_x: Agent,
    agent_o: Agent,
    x_name: str | None = None,
    o_name: str | None = None,
    on_move: Callable[[State, int], None] | None = None,
    start: State | None = None,
) -> GameRecord:
    """Play one game. Agents that raise BudgetExceeded/AgentRefused or make an
    illegal move abort the game (recorded, not raised)."""
    game = game if isinstance(game, Game) else get_game(game)
    x_name = x_name or agent_x.name
    o_name = o_name or agent_o.name
    state = start.copy() if start is not None else game.new_state()
    agents = {X: agent_x, O: agent_o}
    names = {X: x_name, O: o_name}
    times: dict[str, list[float]] = {x_name: [], o_name: []}
    if agent_x is agent_o:
        agent_x.new_game(game, X)
    else:
        agent_x.new_game(game, X)
        agent_o.new_game(game, O)
    aborted = aborted_by = None
    while state.winner is None:
        player = state.to_move
        agent = agents[player]
        t0 = time.perf_counter()
        try:
            move = agent.select_move(state)
        except (BudgetExceeded, AgentRefused) as exc:
            aborted, aborted_by = f"{type(exc).__name__}: {exc}", names[player]
            break
        times[names[player]].append(time.perf_counter() - t0)
        if not state.is_legal(move):
            aborted = f"illegal move {move!r}"
            aborted_by = names[player]
            break
        if on_move is not None:
            on_move(state, move)
        state.push(move)
    record = GameRecord(x_name, o_name, state.history[:], state.winner, times, aborted, aborted_by)
    if aborted is None:
        agent_x.end_game(state, X)
        agent_o.end_game(state, O)
    return record


@dataclass
class MatchResult:
    """Results of a series of games between agents ``a`` and ``b``."""

    a: str
    b: str
    config: str
    wins_a: int = 0
    wins_b: int = 0
    draws: int = 0
    aborted: int = 0
    # Breakdown by side: results when a played X / when a played O.
    a_as_x: dict[str, int] = field(default_factory=lambda: {"win": 0, "draw": 0, "loss": 0})
    a_as_o: dict[str, int] = field(default_factory=lambda: {"win": 0, "draw": 0, "loss": 0})
    time_per_move: dict[str, float] = field(default_factory=dict)
    max_time_per_move: dict[str, float] = field(default_factory=dict)
    mean_length: float = 0.0
    abort_reasons: list[str] = field(default_factory=list)
    games: list[GameRecord] = field(default_factory=list)

    @property
    def n_games(self) -> int:
        return self.wins_a + self.wins_b + self.draws

    @property
    def score_a(self) -> float:
        """a's score fraction (wins + draws/2) / games."""
        return (self.wins_a + 0.5 * self.draws) / self.n_games if self.n_games else float("nan")

    def to_dict(self, include_games: bool = False) -> dict:
        d = asdict(self)
        if not include_games:
            d.pop("games")
        else:
            d["games"] = [g.to_dict() for g in self.games]
        d["n_games"] = self.n_games
        d["score_a"] = self.score_a
        return d


def play_match(
    game: Game | GameConfig,
    a: Agent,
    b: Agent,
    n_games: int,
    seed: int = 0,
    alternate: bool = True,
    a_name: str | None = None,
    b_name: str | None = None,
    keep_games: bool = False,
    stop_on_abort: bool = True,
) -> MatchResult:
    """Play ``n_games`` between ``a`` and ``b``, alternating who plays X.

    Both agents are re-seeded from ``seed`` at the start of the match. If an
    agent aborts (time/memory budget), the match stops early by default and
    the reason is recorded.
    """
    game = game if isinstance(game, Game) else get_game(game)
    a_name = a_name or a.name
    b_name = b_name or b.name
    if a_name == b_name:
        a_name, b_name = a_name + "#1", b_name + "#2"
    a.seed(seed * 1_000_003 + 1)
    if b is not a:
        b.seed(seed * 1_000_003 + 2)
    result = MatchResult(a_name, b_name, game.config.label)
    times: dict[str, list[float]] = {a_name: [], b_name: []}
    lengths = []
    for i in range(n_games):
        a_is_x = (i % 2 == 0) or not alternate
        if a_is_x:
            rec = play_game(game, a, b, a_name, b_name)
        else:
            rec = play_game(game, b, a, b_name, a_name)
        for name, ts in rec.move_times.items():
            times[name].extend(ts)
        if keep_games:
            result.games.append(rec)
        if rec.aborted:
            result.aborted += 1
            result.abort_reasons.append(f"{rec.aborted_by}: {rec.aborted}")
            if stop_on_abort:
                break
            continue
        lengths.append(rec.length)
        side = result.a_as_x if a_is_x else result.a_as_o
        s = rec.score_for(a_name)
        if s == 1.0:
            result.wins_a += 1
            side["win"] += 1
        elif s == 0.0:
            result.wins_b += 1
            side["loss"] += 1
        else:
            result.draws += 1
            side["draw"] += 1
    for name, ts in times.items():
        result.time_per_move[name] = sum(ts) / len(ts) if ts else 0.0
        result.max_time_per_move[name] = max(ts) if ts else 0.0
    result.mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    return result


@dataclass
class TournamentResult:
    config: str
    agents: list[str]
    matches: list[MatchResult]
    seed: int
    games_per_pair: int

    def score_matrix(self) -> dict[str, dict[str, float]]:
        """score[a][b] = a's score fraction against b (NaN if no games)."""
        m: dict[str, dict[str, float]] = {a: {} for a in self.agents}
        for r in self.matches:
            m[r.a][r.b] = r.score_a
            m[r.b][r.a] = 1 - r.score_a if r.n_games else float("nan")
        return m

    def wdl_table(self) -> list[dict]:
        """Total W/D/L per agent across the tournament."""
        rows = {a: {"agent": a, "win": 0, "draw": 0, "loss": 0, "aborted": 0} for a in self.agents}
        for r in self.matches:
            rows[r.a]["win"] += r.wins_a
            rows[r.a]["loss"] += r.wins_b
            rows[r.b]["win"] += r.wins_b
            rows[r.b]["loss"] += r.wins_a
            rows[r.a]["draw"] += r.draws
            rows[r.b]["draw"] += r.draws
            rows[r.a]["aborted"] += r.aborted
            rows[r.b]["aborted"] += r.aborted
        return list(rows.values())

    def pairwise_games(self) -> list[tuple[str, str, float]]:
        """(a, b, score_a) for every completed game, for Elo fitting."""
        out = []
        for r in self.matches:
            out += [(r.a, r.b, 1.0)] * r.wins_a
            out += [(r.a, r.b, 0.0)] * r.wins_b
            out += [(r.a, r.b, 0.5)] * r.draws
        return out

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "agents": self.agents,
            "seed": self.seed,
            "games_per_pair": self.games_per_pair,
            "matches": [r.to_dict() for r in self.matches],
            "score_matrix": self.score_matrix(),
            "wdl": self.wdl_table(),
        }


AgentFactory = Callable[[], Agent]


def round_robin(
    game: Game | GameConfig,
    agents: dict[str, AgentFactory] | Sequence[Agent],
    games_per_pair: int,
    seed: int = 0,
    include_self_play: bool = False,
    progress: Callable[[str], None] | None = None,
) -> TournamentResult:
    """Every agent plays every other agent ``games_per_pair`` times.

    ``agents`` maps names to zero-argument factories (a fresh agent per match,
    so learning agents don't carry state between pairings) or is a list of
    ready agents (reused, keyed by ``agent.name``).
    """
    game = game if isinstance(game, Game) else get_game(game)
    if isinstance(agents, dict):
        names = list(agents)
        make = agents
    else:
        names = [a.name for a in agents]
        by_name = {a.name: a for a in agents}
        make = {n: (lambda n=n: by_name[n]) for n in names}
    matches = []
    pair_index = 0
    for i, na in enumerate(names):
        for j, nb in enumerate(names):
            if j < i or (i == j and not include_self_play):
                continue
            pair_index += 1
            if progress:
                progress(f"{na} vs {nb}")
            a, b = make[na](), make[nb]()
            r = play_match(game, a, b, games_per_pair, seed=seed + pair_index, a_name=na, b_name=nb)
            matches.append(r)
    return TournamentResult(game.config.label, names, matches, seed, games_per_pair)

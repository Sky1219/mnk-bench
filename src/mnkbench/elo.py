"""Elo ratings from game results.

:func:`fit_elo` fits a Bradley-Terry model by minorise-maximise iterations
(draws count as half a win for each side) and reports ratings on the Elo
scale. A weak prior (one virtual draw against an average player) keeps
ratings finite for agents that win or lose every game. :func:`bootstrap_elo`
resamples games to give confidence intervals. :func:`online_elo` is the
classic sequential update, included for comparison.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence

Game = tuple[str, str, float]  # (a, b, score of a: 1, 0.5 or 0)


def fit_elo(
    games: Iterable[Game],
    prior_draws: float = 1.0,
    anchor: str | None = None,
    anchor_rating: float = 1000.0,
    mean_rating: float = 1500.0,
    iterations: int = 2000,
    tol: float = 1e-10,
) -> dict[str, float]:
    games = list(games)
    players = sorted({p for a, b, _ in games for p in (a, b)})
    if not players:
        return {}
    wins: dict[str, float] = defaultdict(float)
    pair_games: dict[tuple[str, str], float] = defaultdict(float)
    for a, b, s in games:
        wins[a] += s
        wins[b] += 1 - s
        key = (a, b) if a < b else (b, a)
        pair_games[key] += 1
    opponents: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for (a, b), n in pair_games.items():
        opponents[a].append((b, n))
        opponents[b].append((a, n))

    gamma = {p: 1.0 for p in players}
    VIRTUAL = 1.0  # strength of the virtual average opponent
    for _ in range(iterations):
        new = {}
        for p in players:
            num = wins[p] + 0.5 * prior_draws
            den = prior_draws / (gamma[p] + VIRTUAL)
            for q, n in opponents[p]:
                den += n / (gamma[p] + gamma[q])
            new[p] = num / den if den > 0 else gamma[p]
        # Normalise (geometric mean 1) for numerical stability.
        g = math.exp(sum(math.log(v) for v in new.values()) / len(new))
        new = {p: v / g for p, v in new.items()}
        delta = max(abs(math.log(new[p]) - math.log(gamma[p])) for p in players)
        gamma = new
        if delta < tol:
            break
    ratings = {p: 400.0 * math.log10(gamma[p]) for p in players}
    if anchor is not None and anchor in ratings:
        shift = anchor_rating - ratings[anchor]
    else:
        shift = mean_rating - sum(ratings.values()) / len(ratings)
    return {p: r + shift for p, r in ratings.items()}


def bootstrap_elo(
    games: Sequence[Game],
    samples: int = 100,
    seed: int = 0,
    level: float = 0.95,
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """Percentile confidence intervals by resampling games with replacement."""
    rng = random.Random(seed)
    games = list(games)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sample = [games[rng.randrange(len(games))] for _ in games]
        for p, r in fit_elo(sample, iterations=300, tol=1e-6, **kwargs).items():
            draws[p].append(r)
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out = {}
    for p, rs in draws.items():
        rs.sort()
        out[p] = (rs[int(lo_q * (len(rs) - 1))], rs[int(hi_q * (len(rs) - 1))])
    return out


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def online_elo(games: Iterable[Game], k: float = 16.0, start: float = 1500.0) -> dict[str, float]:
    ratings: dict[str, float] = defaultdict(lambda: start)
    for a, b, s in games:
        ea = expected_score(ratings[a], ratings[b])
        ratings[a] += k * (s - ea)
        ratings[b] -= k * (s - ea)
    return dict(ratings)

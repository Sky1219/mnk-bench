"""Submodular set functions and greedy maximisation, including the PRISM
family of submodular information measures (Kothawade et al., "PRISM: A Rich
Class of Parameterized Submodular Information Measures for Guided Subset
Selection", AAAI 2022).

A set function f on a ground set V = {0..n-1} is *submodular* if adding an
element helps less the more you already have (diminishing returns):
f(A + j) - f(A) >= f(B + j) - f(B) whenever A is a subset of B. For monotone
submodular f, the greedy algorithm (repeatedly add the element with the
largest marginal gain) is within a factor (1 - 1/e) of the best size-k set.

Every function here is defined by a similarity kernel S (n x n, entries
>= 0) over the ground set and, for the PRISM measures, a cross-similarity
S_VQ (n x |Q|) to a *query* set Q (what we want more of) and/or S_VP to a
*private* set P (what we want to avoid or already have).

Classic functions (diversity / representation):
    FacilityLocation   f(A) = sum_i max_{j in A} S_ij
    GraphCut           f(A) = sum_{i in V, j in A} S_ij - lam sum_{i,j in A} S_ij
    LogDeterminant     f(A) = log det(S_A + lam I)

PRISM submodular mutual information I_f(A; Q) (query-guided):
    FLMI     sum_{i in V} min(max_{j in A} S_ij, eta max_{q in Q} S_iq)
    FLVMI    sum_{q in Q} max_{j in A} S_qj + eta sum_{j in A} max_{q in Q} S_jq
    GCMI     2 lam sum_{j in A} sum_{q in Q} S_jq                     (pure relevance)
    COM      eta sum_{j in A} psi(sum_q S_jq) + sum_q psi(sum_{j in A} S_jq)
    LogDetMI log det(S_A) - log det(S_A - eta^2 S_AQ S_Q^-1 S_AQ^T)

PRISM conditional gain f(A | P) and conditional mutual information I_f(A; Q | P):
    FLCG     sum_i max(max_{j in A} S_ij - nu max_{p in P} S_ip, 0)
    FLCMI    sum_i max(min(max_{j in A} S_ij, eta max_q S_iq) - nu max_p S_ip, 0)

``eta`` trades query relevance (large eta) against diversity and query
coverage (small eta); ``nu`` sets how strongly the private set is avoided.

Each function exposes ``evaluate(A)`` (direct definition, for testing) and a
vectorised incremental interface (``start`` / ``gains`` / ``add``) used by
:func:`greedy`, which computes the marginal gain of every candidate at once.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# ------------------------------------------------------------------ kernels


def cosine_kernel(X: np.ndarray, Y: np.ndarray | None = None, clip: bool = True) -> np.ndarray:
    """Cosine similarity between rows of X and rows of Y (default Y = X).

    Negative values are clipped to 0 (facility location needs S >= 0). For
    non-negative features (e.g. ReLU activations) nothing is clipped and the
    X-vs-X kernel is positive semi-definite.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    Yn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
    S = Xn @ Yn.T
    return np.clip(S, 0.0, 1.0) if clip else S


def rbf_kernel(X: np.ndarray, Y: np.ndarray | None = None, gamma: float | None = None) -> np.ndarray:
    """exp(-gamma ||x - y||^2); gamma defaults to 1 / median squared distance."""
    X = np.asarray(X, dtype=np.float64)
    Y = X if Y is None else np.asarray(Y, dtype=np.float64)
    d2 = np.maximum((X * X).sum(1)[:, None] + (Y * Y).sum(1)[None, :] - 2 * X @ Y.T, 0.0)
    if gamma is None:
        med = float(np.median(d2[d2 > 0])) if np.any(d2 > 0) else 1.0
        gamma = 1.0 / max(med, 1e-12)
    return np.exp(-gamma * d2)


# --------------------------------------------------------------- base class


class SetFunction:
    """Interface: ``evaluate(A)``; ``start() -> state``; ``gains(state) -> array``
    (marginal gain of every element); ``add(state, j)``."""

    name = "f"
    n: int

    def evaluate(self, A: Sequence[int]) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def start(self):  # pragma: no cover - abstract
        raise NotImplementedError

    def gains(self, state) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def add(self, state, j: int) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def _max_rows(M: np.ndarray, idx: Sequence[int]) -> np.ndarray:
    """max over the given columns of M, per row (0 if idx is empty)."""
    if len(idx) == 0:
        return np.zeros(M.shape[0])
    return M[:, list(idx)].max(axis=1)


# ---------------------------------------------------- classic functions


class FacilityLocation(SetFunction):
    name = "FL"

    def gain(self, state, j):
        return float(np.maximum(self.S[:, j] - state["cur"], 0.0).sum())

    def __init__(self, S: np.ndarray):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]

    def evaluate(self, A):
        return float(_max_rows(self.S, A).sum())

    def start(self):
        return {"cur": np.zeros(self.n)}

    def gains(self, state):
        return np.maximum(self.S - state["cur"][:, None], 0.0).sum(axis=0)

    def add(self, state, j):
        np.maximum(state["cur"], self.S[:, j], out=state["cur"])


class GraphCut(SetFunction):
    name = "GC"

    def __init__(self, S: np.ndarray, lam: float = 0.5):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]
        self.lam = lam
        self.colsum = self.S.sum(axis=0)

    def evaluate(self, A):
        A = list(A)
        if not A:
            return 0.0
        return float(self.colsum[A].sum() - self.lam * self.S[np.ix_(A, A)].sum())

    def start(self):
        return {"inA": np.zeros(self.n)}

    def gains(self, state):
        return self.colsum - self.lam * (2 * state["inA"] + np.diag(self.S))

    def add(self, state, j):
        state["inA"] += self.S[j, :]


class _IncrementalLogDet:
    """Incremental Cholesky for greedy log-det (Chen et al., NeurIPS 2018):
    d2[j] is the Schur complement of j given the selected set, so the gain
    of adding j is log(d2[j])."""

    def __init__(self, K: np.ndarray):
        self.K = K
        self.n = K.shape[0]
        self.d2 = np.diag(K).astype(np.float64).copy()
        self.C: list[np.ndarray] = []

    def gain(self) -> np.ndarray:
        return np.log(np.maximum(self.d2, 1e-300))

    def add(self, j: int) -> None:
        dj = math.sqrt(max(self.d2[j], 1e-300))
        row = self.K[j, :].copy()
        for c in self.C:
            row -= c[j] * c
        e = row / dj
        self.C.append(e)
        self.d2 = self.d2 - e * e
        self.d2[j] = 0.0


def _logdet(M: np.ndarray) -> float:
    if M.size == 0:
        return 0.0
    sign, val = np.linalg.slogdet(M)
    return float(val) if sign > 0 else float("-inf")


class LogDeterminant(SetFunction):
    name = "LogDet"

    def __init__(self, S: np.ndarray, lam: float = 1.0):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]
        self.lam = lam
        self.K = self.S + lam * np.eye(self.n)

    def evaluate(self, A):
        A = list(A)
        return _logdet(self.K[np.ix_(A, A)])

    def start(self):
        return {"chol": _IncrementalLogDet(self.K)}

    def gains(self, state):
        return state["chol"].gain()

    def add(self, state, j):
        state["chol"].add(j)


# ------------------------------------------- PRISM: mutual information


class FLMI(SetFunction):
    """Facility-location mutual information (FL1MI / FLQMI in PRISM)."""

    name = "FLMI"

    def __init__(self, S: np.ndarray, S_VQ: np.ndarray, eta: float = 1.0):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]
        self.cap = eta * np.asarray(S_VQ, dtype=np.float64).max(axis=1)
        self.eta = eta

    def evaluate(self, A):
        return float(np.minimum(_max_rows(self.S, A), self.cap).sum())

    def start(self):
        return {"cur": np.zeros(self.n)}

    def gains(self, state):
        cur = np.minimum(state["cur"], self.cap)
        new = np.minimum(np.maximum(self.S, state["cur"][:, None]), self.cap[:, None])
        return (new - cur[:, None]).sum(axis=0)

    def add(self, state, j):
        np.maximum(state["cur"], self.S[:, j], out=state["cur"])

    def gain(self, state, j):
        cur = state["cur"]
        return float((np.minimum(np.maximum(self.S[:, j], cur), self.cap) - np.minimum(cur, self.cap)).sum())


class FLVMI(SetFunction):
    """Facility-location variant mutual information (FL2MI / FLVMI): only
    data-to-query similarities; never saturates."""

    name = "FLVMI"

    def __init__(self, S_VQ: np.ndarray, eta: float = 1.0):
        self.SQ = np.asarray(S_VQ, dtype=np.float64)
        self.n = self.SQ.shape[0]
        self.eta = eta
        self.rel = self.SQ.max(axis=1)

    def evaluate(self, A):
        A = list(A)
        if not A:
            return 0.0
        return float(self.SQ[A, :].max(axis=0).sum() + self.eta * self.rel[A].sum())

    def start(self):
        return {"curQ": np.zeros(self.SQ.shape[1])}

    def gains(self, state):
        return np.maximum(self.SQ - state["curQ"][None, :], 0.0).sum(axis=1) + self.eta * self.rel

    def add(self, state, j):
        np.maximum(state["curQ"], self.SQ[j, :], out=state["curQ"])


class GCMI(SetFunction):
    """Graph-cut mutual information: modular, pure query relevance."""

    name = "GCMI"

    def __init__(self, S_VQ: np.ndarray, lam: float = 0.5):
        self.score = 2 * lam * np.asarray(S_VQ, dtype=np.float64).sum(axis=1)
        self.n = self.score.shape[0]

    def evaluate(self, A):
        return float(self.score[list(A)].sum()) if len(A) else 0.0

    def start(self):
        return {}

    def gains(self, state):
        return self.score.copy()

    def add(self, state, j):
        pass


class COM(SetFunction):
    """Concave-over-modular generalised mutual information (psi = sqrt)."""

    name = "COM"

    def __init__(self, S_VQ: np.ndarray, eta: float = 1.0):
        self.SQ = np.asarray(S_VQ, dtype=np.float64)
        self.n = self.SQ.shape[0]
        self.eta = eta
        self.own = np.sqrt(self.SQ.sum(axis=1))

    def evaluate(self, A):
        A = list(A)
        if not A:
            return 0.0
        return float(self.eta * self.own[A].sum() + np.sqrt(self.SQ[A, :].sum(axis=0)).sum())

    def start(self):
        return {"t": np.zeros(self.SQ.shape[1])}

    def gains(self, state):
        t = state["t"]
        return self.eta * self.own + (np.sqrt(t[None, :] + self.SQ) - np.sqrt(t)[None, :]).sum(axis=1)

    def add(self, state, j):
        state["t"] += self.SQ[j, :]


class LogDetMI(SetFunction):
    """Log-determinant mutual information:
    log det(S_A) - log det(S_A - eta^2 S_AQ S_Q^-1 S_AQ^T)."""

    name = "LogDetMI"

    def __init__(self, S: np.ndarray, S_VQ: np.ndarray, S_QQ: np.ndarray, eta: float = 1.0,
                 lam: float = 1.0):
        S = np.asarray(S, dtype=np.float64)
        SVQ = np.asarray(S_VQ, dtype=np.float64)
        SQQ = np.asarray(S_QQ, dtype=np.float64) + lam * np.eye(len(S_QQ))
        self.n = S.shape[0]
        self.K1 = S + lam * np.eye(self.n)
        proj = SVQ @ np.linalg.solve(SQQ, SVQ.T)
        self.K2 = self.K1 - (eta**2) * proj
        # Keep the conditional kernel positive definite for numerical safety.
        w_min = float(np.linalg.eigvalsh(self.K2).min()) if self.n <= 4000 else 0.0
        if w_min <= 1e-8:
            self.K2 += (1e-6 - w_min) * np.eye(self.n)
        self.eta = eta

    def evaluate(self, A):
        A = list(A)
        return _logdet(self.K1[np.ix_(A, A)]) - _logdet(self.K2[np.ix_(A, A)])

    def start(self):
        return {"a": _IncrementalLogDet(self.K1), "b": _IncrementalLogDet(self.K2)}

    def gains(self, state):
        return state["a"].gain() - state["b"].gain()

    def add(self, state, j):
        state["a"].add(j)
        state["b"].add(j)


# ------------------------------- PRISM: conditional gain / conditional MI


class FLCG(SetFunction):
    """Facility-location conditional gain f(A | P): representation of V not
    already covered by the private set P."""

    name = "FLCG"

    def __init__(self, S: np.ndarray, S_VP: np.ndarray, nu: float = 1.0):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]
        self.base = nu * np.asarray(S_VP, dtype=np.float64).max(axis=1)

    def evaluate(self, A):
        return float(np.maximum(_max_rows(self.S, A) - self.base, 0.0).sum())

    def start(self):
        return {"cur": np.zeros(self.n)}

    def gains(self, state):
        cur = np.maximum(state["cur"] - self.base, 0.0)
        new = np.maximum(np.maximum(self.S, state["cur"][:, None]) - self.base[:, None], 0.0)
        return (new - cur[:, None]).sum(axis=0)

    def add(self, state, j):
        np.maximum(state["cur"], self.S[:, j], out=state["cur"])

    def gain(self, state, j):
        cur = state["cur"]
        return float((np.maximum(np.maximum(self.S[:, j], cur) - self.base, 0.0)
                      - np.maximum(cur - self.base, 0.0)).sum())


class FLCMI(SetFunction):
    """Facility-location conditional mutual information I_f(A; Q | P):
    like the query, unlike the private set."""

    name = "FLCMI"

    def __init__(self, S: np.ndarray, S_VQ: np.ndarray, S_VP: np.ndarray, eta: float = 1.0, nu: float = 1.0):
        self.S = np.asarray(S, dtype=np.float64)
        self.n = self.S.shape[0]
        self.cap = eta * np.asarray(S_VQ, dtype=np.float64).max(axis=1)
        self.base = nu * np.asarray(S_VP, dtype=np.float64).max(axis=1)

    def _value(self, cur: np.ndarray) -> np.ndarray:
        cap, base = self.cap, self.base
        if cur.ndim == 2:
            cap, base = cap[:, None], base[:, None]
        return np.maximum(np.minimum(cur, cap) - base, 0.0)

    def evaluate(self, A):
        return float(self._value(_max_rows(self.S, A)).sum())

    def start(self):
        return {"cur": np.zeros(self.n)}

    def gains(self, state):
        cur = self._value(state["cur"])
        new = self._value(np.maximum(self.S, state["cur"][:, None]))
        return (new - cur[:, None]).sum(axis=0)

    def gain(self, state, j):
        cur = state["cur"]
        return float((self._value(np.maximum(self.S[:, j], cur)) - self._value(cur)).sum())

    def add(self, state, j):
        np.maximum(state["cur"], self.S[:, j], out=state["cur"])


# ----------------------------------------------------------- maximisation


def greedy(f: SetFunction, k: int, rng: np.random.Generator | None = None,
           exclude: Sequence[int] = (), random_ties: bool = True) -> list[int]:
    """Greedy maximisation of ``f`` subject to |A| <= k (ties broken at random,
    or by lowest index with ``random_ties=False``).

    For monotone submodular f this is a (1 - 1/e)-approximation. Each step
    computes all marginal gains at once (vectorised), so cost is k gain sweeps.
    """
    rng = rng or np.random.default_rng(0)
    state = f.start()
    chosen: list[int] = []
    mask = np.zeros(f.n, dtype=bool)
    mask[list(exclude)] = True
    k = min(k, f.n - int(mask.sum()))
    for _ in range(k):
        g = f.gains(state).astype(np.float64)
        g[mask] = -np.inf
        best = g.max()
        ties = np.flatnonzero(g >= best - 1e-12 * max(1.0, abs(best)))
        j = int(ties[rng.integers(len(ties))]) if random_ties and len(ties) > 1 else int(ties[0])
        f.add(state, j)
        chosen.append(j)
        mask[j] = True
    return chosen


def lazy_greedy(f: SetFunction, k: int, rng: np.random.Generator | None = None,
                exclude: Sequence[int] = ()) -> list[int]:
    """Minoux's lazy greedy: same result as :func:`greedy` for submodular f
    (up to ties), but re-evaluates only candidates whose stale upper bound
    reaches the top of a priority queue. Uses ``f.gain(state, j)`` when the
    function provides it, else falls back to the vectorised ``gains``."""
    import heapq

    if not hasattr(f, "gain"):
        return greedy(f, k, rng, exclude)
    rng = rng or np.random.default_rng(0)
    state = f.start()
    excluded = set(int(x) for x in exclude)
    g0 = f.gains(state)
    jitter = rng.random(f.n)  # random tie-breaking
    heap = [(-float(g0[j]), jitter[j], j) for j in range(f.n) if j not in excluded]
    heapq.heapify(heap)
    chosen: list[int] = []
    fresh = {j: 0 for j in range(f.n)}
    step = 0
    while heap and len(chosen) < k:
        neg, tie, j = heapq.heappop(heap)
        if fresh[j] == step:
            f.add(state, j)
            chosen.append(j)
            step += 1
            continue
        g = f.gain(state, j)
        fresh[j] = step
        heapq.heappush(heap, (-g, tie, j))
    return chosen


def naive_greedy(f: SetFunction, k: int) -> list[int]:
    """Reference greedy using ``evaluate`` only (slow; for tests)."""
    chosen: list[int] = []
    for _ in range(min(k, f.n)):
        base = f.evaluate(chosen)
        best, best_j = -math.inf, None
        for j in range(f.n):
            if j in chosen:
                continue
            g = f.evaluate(chosen + [j]) - base
            if g > best + 1e-12:
                best, best_j = g, j
        chosen.append(best_j)
    return chosen


def make_function(name: str, S: np.ndarray, S_VQ: np.ndarray | None = None, S_QQ: np.ndarray | None = None,
                  S_VP: np.ndarray | None = None, eta: float = 1.0, nu: float = 1.0, lam: float = 1.0
                  ) -> SetFunction:
    """Build a function by name (FL, GC, LogDet, FLMI, FLVMI, GCMI, COM, LogDetMI, FLCG, FLCMI)."""
    if name == "FL":
        return FacilityLocation(S)
    if name == "GC":
        return GraphCut(S, lam=0.5)
    if name == "LogDet":
        return LogDeterminant(S, lam=lam)
    if name == "FLMI":
        return FLMI(S, S_VQ, eta=eta)
    if name == "FLVMI":
        return FLVMI(S_VQ, eta=eta)
    if name == "GCMI":
        return GCMI(S_VQ)
    if name == "COM":
        return COM(S_VQ, eta=eta)
    if name == "LogDetMI":
        return LogDetMI(S, S_VQ, S_QQ, eta=eta, lam=lam)
    if name == "FLCG":
        return FLCG(S, S_VP, nu=nu)
    if name == "FLCMI":
        return FLCMI(S, S_VQ, S_VP, eta=eta, nu=nu)
    raise ValueError(f"unknown submodular function {name!r}")

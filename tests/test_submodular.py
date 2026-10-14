import itertools
import math

import numpy as np
import pytest

from mnkbench.submodular import (
    COM,
    FLCG,
    FLCMI,
    FLMI,
    FLVMI,
    GCMI,
    FacilityLocation,
    GraphCut,
    LogDeterminant,
    LogDetMI,
    cosine_kernel,
    greedy,
    make_function,
    naive_greedy,
    rbf_kernel,
)


def instance(n=12, q=3, p=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((n + q + p, 4))  # non-negative features -> PSD cosine kernel
    K = cosine_kernel(X)
    V, Q, P = range(n), range(n, n + q), range(n + q, n + q + p)
    return K[np.ix_(V, V)], K[np.ix_(V, Q)], K[np.ix_(Q, Q)], K[np.ix_(V, P)]


def all_functions(S, SVQ, SQQ, SVP):
    return [
        FacilityLocation(S), GraphCut(S, 0.4), LogDeterminant(S, 1.0),
        FLMI(S, SVQ, 1.0), FLMI(S, SVQ, 3.0), FLVMI(SVQ, 1.0), GCMI(SVQ), COM(SVQ, 1.0),
        LogDetMI(S, SVQ, SQQ, eta=1.0), FLCG(S, SVP, 1.0), FLCMI(S, SVQ, SVP, 1.0, 1.0),
    ]


def test_kernels():
    X = np.random.default_rng(1).random((5, 3))
    K = cosine_kernel(X)
    assert np.allclose(np.diag(K), 1) and (K >= 0).all() and np.allclose(K, K.T)
    R = rbf_kernel(X)
    assert np.allclose(np.diag(R), 1) and (R > 0).all()
    assert np.linalg.eigvalsh(K).min() > -1e-9


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_incremental_gains_match_definition(seed):
    S, SVQ, SQQ, SVP = instance(seed=seed)
    rng = np.random.default_rng(seed)
    for f in all_functions(S, SVQ, SQQ, SVP):
        state = f.start()
        A = []
        for _ in range(5):
            g = f.gains(state)
            for j in rng.choice([j for j in range(f.n) if j not in A], 4, replace=False):
                expected = f.evaluate(A + [int(j)]) - f.evaluate(A)
                assert g[j] == pytest.approx(expected, rel=1e-6, abs=1e-8), f.name
            j = int(rng.choice([j for j in range(f.n) if j not in A]))
            f.add(state, j)
            A.append(j)


@pytest.mark.parametrize("seed", [0, 3])
def test_submodular_and_monotone(seed):
    S, SVQ, SQQ, SVP = instance(n=9, seed=seed)
    rng = np.random.default_rng(seed)
    for f in all_functions(S, SVQ, SQQ, SVP):
        monotone = f.name != "GC"
        for _ in range(30):
            B = list(rng.choice(f.n, 5, replace=False))
            A = B[:2]
            j = next(x for x in range(f.n) if x not in B)
            gA = f.evaluate(A + [j]) - f.evaluate(A)
            gB = f.evaluate(B + [j]) - f.evaluate(B)
            assert gA >= gB - 1e-8, f"{f.name} not submodular"
            if monotone:
                assert gB >= -1e-8, f"{f.name} not monotone"


def test_greedy_matches_naive_and_approximation():
    S, SVQ, SQQ, SVP = instance(n=10, seed=4)
    for f in all_functions(S, SVQ, SQQ, SVP):
        fast = greedy(f, 3, random_ties=False)
        slow = naive_greedy(f, 3)
        assert f.evaluate(fast) == pytest.approx(f.evaluate(slow), rel=1e-9)
        if f.name in ("FL", "LogDet", "FLMI", "FLVMI", "COM", "LogDetMI", "FLCG", "FLCMI", "GCMI"):
            opt = max(f.evaluate(list(c)) for c in itertools.combinations(range(f.n), 3))
            lo = min(0.0, opt)
            assert f.evaluate(fast) - lo >= (1 - 1 / math.e) * (opt - lo) - 1e-9


def test_query_relevance_vs_diversity():
    # Two clusters; the query sits in cluster 0. MI functions should pick from
    # cluster 0, plain facility location should cover both clusters.
    rng = np.random.default_rng(0)
    c0 = rng.normal([5, 0.2], 0.3, (20, 2))
    c1 = rng.normal([0.2, 5], 0.3, (20, 2))
    X = np.abs(np.vstack([c0, c1]))
    Qx = np.abs(rng.normal([5, 0.2], 0.3, (3, 2)))
    S, SVQ = cosine_kernel(X), cosine_kernel(X, Qx)
    for name in ("FLMI", "FLVMI", "GCMI", "COM"):
        sel = greedy(make_function(name, S, SVQ, eta=1.0), 5)
        assert sum(j < 20 for j in sel) >= 4, name
    fl = greedy(make_function("FL", S), 4)
    assert any(j < 20 for j in fl) and any(j >= 20 for j in fl)


def test_conditional_gain_avoids_private_set():
    rng = np.random.default_rng(0)
    X = np.abs(np.vstack([rng.normal([5, 0.2], 0.3, (20, 2)), rng.normal([0.2, 5], 0.3, (20, 2))]))
    P = np.abs(rng.normal([5, 0.2], 0.3, (3, 2)))
    S, SVP = cosine_kernel(X), cosine_kernel(X, P)
    sel = greedy(make_function("FLCG", S, S_VP=SVP, nu=1.0), 3)
    assert all(j >= 20 for j in sel)


def test_greedy_exclude_and_unknown():
    S, SVQ, SQQ, SVP = instance()
    sel = greedy(FacilityLocation(S), 4, exclude=[0, 1, 2])
    assert not set(sel) & {0, 1, 2} and len(sel) == 4
    with pytest.raises(ValueError):
        make_function("nope", S)


def test_lazy_greedy_matches_greedy():
    from mnkbench.submodular import lazy_greedy

    rng = np.random.default_rng(7)
    X = rng.random((60, 5)) + rng.normal(0, 0.01, (60, 5))
    Qx = rng.random((4, 5))
    S, SVQ = cosine_kernel(X), cosine_kernel(X, Qx)
    for name in ("FL", "FLMI", "FLCG", "FLCMI"):
        f = make_function(name, S, SVQ, S_VP=SVQ[:, :2], eta=1.0, nu=0.5)
        a, b = greedy(f, 10, random_ties=False), lazy_greedy(f, 10)
        assert f.evaluate(a) == pytest.approx(f.evaluate(b), rel=1e-9), name
        state = f.start()
        f.add(state, 3)
        assert f.gain(state, 5) == pytest.approx(f.gains(state)[5])

import numpy as np
import pytest

from mnkbench.experiments.targeted import motif, paired_vs_random, pooled_over_targets, summarize
from mnkbench.game import GameConfig, State

C = GameConfig(5, 5, 4, gravity=True)


def test_motifs():
    # X to move with three in the bottom row and d5 playable: an immediate win.
    w = State.from_string(C, "...../...../...../OO.../XXX.O")
    assert w.to_move == 1 and motif(w) == "win"
    # O to move, X threatens d5: O must block.
    b = State.from_string(C, "...../...../...../OO..X/XXX.O")
    assert b.to_move == 2 and motif(b) == "block"
    # O to move; X threatens c4, which only becomes playable once c5 is
    # filled, so c5 is a poisoned move.
    p = State.from_string(C, "...../...../...../XX.X./OX.OO")
    assert p.to_move == 2 and motif(p) == "poison"
    q = State.from_string(C, "...../...../...../...../..X..")
    assert motif(q) == "quiet"


def test_summaries_and_paired():
    rows = []
    for seed in range(3):
        for m, acc in (("random", 0.5), ("FLMI", 0.6)):
            rows.append({"target": "win", "method": m, "seed": seed, "k": 50, "acc_target": acc + 0.01 * seed,
                         "acc_critical": 0.6, "hit_target": 0.2 if m == "random" else 0.4})
    summ = summarize(rows)
    flmi = next(r for r in summ if r["method"] == "FLMI")
    assert flmi["acc_target"] == pytest.approx(0.61) and flmi["seeds"] == 3
    paired = paired_vs_random(rows)
    assert paired[0]["wins"] == 3 and paired[0]["mean_diff"] == pytest.approx(0.1)
    pooled = pooled_over_targets(rows)
    assert {r["method"] for r in pooled} == {"random", "FLMI"}


def test_quick_experiment(tmp_path):
    pytest.importorskip("torch")
    from mnkbench.experiments.targeted import exp_targeted

    d = exp_targeted(tmp_path, quick=True, log=lambda s: None, jobs=1)
    methods = {r["method"] for r in d["rows"]}
    assert {"base", "query-only", "random", "FLMI", "LogDetMI", "oracle"} <= methods
    oracle = [r for r in d["rows"] if r["method"] == "oracle"]
    assert all(r["hit_target"] == 1.0 for r in oracle if r["k"] <= 10)
    assert (tmp_path / "targeted" / "targeted_budget.png").exists()
    assert np.isfinite([r["acc_critical"] for r in d["rows"]]).all()

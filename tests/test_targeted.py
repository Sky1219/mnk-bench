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


def test_significance_and_dose_response():
    from mnkbench.experiments.targeted import _sign_test, dose_response, significance

    assert _sign_test(5, 5) == 1.0
    assert _sign_test(10, 0) == pytest.approx(2 / 1024)
    rows = []
    for seed in range(6):
        for k in (25, 400):
            rows.append({"target": "win", "method": "random", "seed": seed, "k": k, "acc_target": 0.5,
                         "acc_critical": 0.6, "hit_target": 0.1})
            rows.append({"target": "win", "method": "LogDetMI", "seed": seed, "k": k, "acc_target": 0.55,
                         "acc_critical": 0.59, "hit_target": 0.3})
            rows.append({"target": "win", "method": "oracle", "seed": seed, "k": k,
                         "acc_target": 0.5 + 0.05 * np.log2(k + 20), "acc_critical": 0.6, "hit_target": 1.0})
        rows.append({"target": "win", "method": "query-only", "seed": seed, "k": 0, "acc_target": 0.5,
                     "acc_critical": 0.6, "hit_target": None})
    sig = {(x["metric"], x["budgets"], x["method"]): x for x in significance(rows)}
    s = sig[("acc_target", "all", "LogDetMI")]
    assert s["wins"] == 12 and s["mean_diff"] == pytest.approx(0.05) and s["p_sign"] < 0.001
    assert sig[("acc_critical", "all", "LogDetMI")]["losses"] == 12
    dr = dose_response(rows, nq=20)
    assert dr["fit"]["slope_per_doubling"] == pytest.approx(0.05, abs=1e-6)
    assert abs(dr["residuals"]["oracle"]) < 1e-9


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

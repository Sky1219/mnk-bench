from mnkbench.agents import make_agent
from mnkbench.agents.base import AgentRefused, Budget, Status
from mnkbench.feasibility import assess, feasibility_map, probe
from mnkbench.game import GOMOKU, TIC_TAC_TOE, GameConfig

import pytest


def test_minimax_ok_small_refused_large():
    b = Budget()
    assert assess("minimax", TIC_TAC_TOE, b).status == Status.OK
    rep = assess("minimax", GameConfig(6, 6, 4), b)
    assert rep.status == Status.REFUSED and "time/move" in rep.reason
    with pytest.raises(AgentRefused):
        make_agent("minimax").adapt(GameConfig(6, 6, 4), b)


def test_tabular_degrades_then_refuses():
    b = Budget()
    assert assess("qlearning", GameConfig(4, 4, 4), b).status == Status.DEGRADED
    assert assess("qlearning", GameConfig(6, 6, 4), b).status == Status.REFUSED


def test_search_agents_always_have_a_plan():
    b = Budget(time_per_move=0.05)
    for spec in ("alphabeta:depth=6", "mcts:iterations=5000"):
        rep = assess(spec, GOMOKU, b)
        assert rep.status in (Status.OK, Status.DEGRADED)
        agent = make_agent(spec).adapt(GOMOKU, b)
        assert agent.estimate(GOMOKU).time_per_move <= 0.05 + 1e-9


def test_probe_ok_and_forced_abort():
    b = Budget(time_per_move=0.2)
    ok = probe("random", TIC_TAC_TOE, b, moves=2)
    assert ok.status == Status.OK and ok.moves == 2
    bad = probe("minimax", GameConfig(5, 5, 4), b, moves=1, force=True, slack=1.0, startup=2.0)
    assert bad.status == Status.ABORTED


def test_probe_refused():
    r = probe("minimax", GameConfig(6, 6, 4), Budget(), moves=1)
    assert r.status == Status.REFUSED


def test_feasibility_map_structure():
    d = feasibility_map(["random", "minimax"], [TIC_TAC_TOE, GameConfig(6, 6, 4)], Budget())
    assert d["boards"] == ["(3,3,3)", "(6,6,4)"]
    status = {(c["agent"], c["config"]): c["status"] for c in d["cells"]}
    assert status[("minimax", "(6,6,4)")] == "refused"
    assert status[("random", "(6,6,4)")] == "ok"


def test_feasibility_experiment_quick(tmp_path):
    from mnkbench.experiments.feasible import exp_feasibility

    d = exp_feasibility(tmp_path, quick=True, log=lambda s: None)
    assert set(d["maps"]) == {"standard", "tight"}
    assert (tmp_path / "feasibility" / "feasibility_standard.png").exists()

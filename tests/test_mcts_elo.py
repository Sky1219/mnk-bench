import pytest

from mnkbench.agents import make_agent
from mnkbench.agents.base import Budget, Status
from mnkbench.agents.mcts import MCTSAgent
from mnkbench.elo import bootstrap_elo, expected_score, fit_elo, online_elo
from mnkbench.game import GOMOKU, TIC_TAC_TOE, GameConfig, State
from mnkbench.metrics import optimal_move_rate
from mnkbench.tournament import play_match


def test_mcts_finds_wins_without_tactics():
    a = MCTSAgent(seed=0, iterations=400, tactical=False)
    s = State.from_string(TIC_TAC_TOE, "XX./OO./...")
    assert a.select_move(s) == 2


def test_mcts_mostly_optimal_on_tic_tac_toe():
    r = optimal_move_rate(make_agent("mcts:iterations=500"), TIC_TAC_TOE, 20, seed=0)
    assert r["optimal_rate"] >= 0.9


def test_mcts_gravity_and_reproducible():
    cfg = GameConfig(5, 5, 4, gravity=True)
    r1 = play_match(cfg, make_agent("mcts:iterations=100"), make_agent("random"), 4, seed=1, keep_games=True)
    r2 = play_match(cfg, make_agent("mcts:iterations=100"), make_agent("random"), 4, seed=1, keep_games=True)
    assert [g.moves for g in r1.games] == [g.moves for g in r2.games]
    assert r1.wins_a >= 3


def test_mcts_time_limit_and_degrade():
    a = MCTSAgent(seed=0, iterations=None, time_limit=0.05)
    s = State.from_string(TIC_TAC_TOE, "X../.../...")
    import time

    t0 = time.perf_counter()
    a.select_move(s)
    assert time.perf_counter() - t0 < 0.5
    rep = MCTSAgent(iterations=100000).feasibility(GOMOKU, Budget(time_per_move=1.0))
    assert rep.status == Status.DEGRADED
    assert 1000 < rep.degraded_params["iterations"] < 100000
    rep = MCTSAgent().feasibility(GOMOKU, Budget(time_per_move=0.001))
    assert rep.status == Status.REFUSED


def test_elo_orders_players():
    games = [("a", "b", 1.0)] * 30 + [("a", "b", 0.0)] * 10 + [("b", "c", 1.0)] * 30 + [("b", "c", 0.5)] * 10
    r = fit_elo(games)
    assert r["a"] > r["b"] > r["c"]
    assert sum(r.values()) / 3 == pytest.approx(1500)
    # 75% score -> about 191 Elo
    two = fit_elo([("a", "b", 1.0)] * 75 + [("a", "b", 0.0)] * 25, prior_draws=0.0)
    assert two["a"] - two["b"] == pytest.approx(190.8, abs=1)
    anchored = fit_elo(games, anchor="c", anchor_rating=1000)
    assert anchored["c"] == pytest.approx(1000)


def test_elo_finite_for_perfect_records():
    r = fit_elo([("a", "b", 1.0)] * 20)
    assert r["a"] > r["b"] and abs(r["a"]) < 1e5


def test_bootstrap_and_online():
    games = [("a", "b", 1.0)] * 20 + [("a", "b", 0.0)] * 10
    ci = bootstrap_elo(games, samples=30)
    lo, hi = ci["a"]
    assert lo <= hi
    on = online_elo(games)
    assert set(on) == {"a", "b"}
    assert expected_score(1600, 1400) == pytest.approx(0.76, abs=0.01)


def test_quick_mcts_elo_experiments(tmp_path):
    from mnkbench.experiments.sampling import exp_elo, exp_mcts

    d = exp_mcts(tmp_path, quick=True, log=lambda s: None)
    assert d["rows"]
    d = exp_elo(tmp_path, quick=True, log=lambda s: None)
    assert d["boards"][0]["elo"]["ratings"]["random"] == pytest.approx(1000)

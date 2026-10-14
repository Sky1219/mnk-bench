import random

from mnkbench.agents import make_agent
from mnkbench.agents.alphabeta import AlphaBetaAgent, _Board
from mnkbench.agents.base import Budget, Status
from mnkbench.game import GOMOKU, TIC_TAC_TOE, GameConfig, State, get_game
from mnkbench.metrics import optimal_move_rate
from mnkbench.tournament import play_match


def test_incremental_eval_matches_fresh():
    g = get_game(GameConfig(6, 6, 4))
    s = g.new_state()
    b = _Board(s, 8.0, 2)
    rng = random.Random(0)
    for _ in range(15):
        mv = rng.choice(s.legal_moves())
        s.push(mv)
        b.push(mv)
        f = _Board(s, 8.0, 2)
        assert abs(f.sx - b.sx) < 1e-6 and abs(f.so - b.so) < 1e-6
        assert f.near == b.near
    for _ in range(7):
        s.pop()
        b.pop()
    f = _Board(s, 8.0, 2)
    assert abs(f.sx - b.sx) < 1e-6 and abs(f.so - b.so) < 1e-6


def test_perfect_on_tic_tac_toe():
    r = optimal_move_rate(make_agent("alphabeta:depth=4"), TIC_TAC_TOE, 30, seed=1)
    assert r["optimal_rate"] == 1.0


def test_blocks_and_wins_on_big_board():
    a = AlphaBetaAgent(seed=0, depth=2)
    rows = ["." * 15] * 15
    rows[7] = "....XXXX......."
    rows[8] = "....OOO........"
    s = State.from_string(GOMOKU, "/".join(rows))
    assert s.to_move == 2
    assert a.select_move(s) in (7 * 15 + 3, 7 * 15 + 8)  # must block (open four: lost, but block)
    rows[8] = "....OOOO......."
    rows[9] = "X.............."
    s = State.from_string(GOMOKU, "/".join(rows))
    assert s.to_move == 2
    assert s.would_win(a.select_move(s), 2)


def test_beats_rule_on_6x6():
    r = play_match(GameConfig(6, 6, 4), make_agent("alphabeta:depth=4"), make_agent("rule"), 6, seed=0)
    assert r.wins_a >= 4


def test_time_limit_degrades_gracefully():
    a = AlphaBetaAgent(seed=0, depth=8, time_limit=0.05)
    s = get_game(GOMOKU).new_state()
    for mv in (112, 113, 127, 98):
        s.push(mv)
    import time

    t0 = time.perf_counter()
    mv = a.select_move(s)
    assert time.perf_counter() - t0 < 0.5 and s.is_legal(mv)


def test_feasibility_degrades_depth():
    a = AlphaBetaAgent(depth=8)
    rep = a.feasibility(GOMOKU, Budget(time_per_move=0.5))
    assert rep.status == Status.DEGRADED and rep.degraded_params["depth"] < 8
    smaller = a.adapt(GOMOKU, Budget(time_per_move=0.5))
    assert smaller.depth == rep.degraded_params["depth"]


def test_bigboards_quick(tmp_path):
    from mnkbench.experiments.bigboards import exp_bigboards

    data = exp_bigboards(tmp_path, quick=True, log=lambda s: None)
    assert data["scaling"] and data["tournament"]["agents"]
    assert (tmp_path / "bigboards" / "table_size.png").exists()

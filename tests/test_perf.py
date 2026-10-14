import random
from collections import Counter

from mnkbench.game import DRAW, GOMOKU, TIC_TAC_TOE, GameConfig, State, get_game
from mnkbench.perf import bench_agent_move, bench_playouts, bench_push_pop
from mnkbench.playout import random_playout, reference_playout


def test_playout_does_not_modify_state():
    s = State.from_string(TIC_TAC_TOE, "X../.O./...")
    before = (s.cells[:], s.history[:], s.to_move)
    random_playout(s, random.Random(0))
    assert (s.cells, s.history, s.to_move) == before


def test_playout_terminal_state():
    s = State.from_string(TIC_TAC_TOE, "XXX/OO./...")
    assert random_playout(s, random.Random(0)) == 1


def test_fast_playout_matches_reference_distribution():
    for cfg in (TIC_TAC_TOE, GameConfig(4, 4, 3), GameConfig(5, 5, 4, gravity=True)):
        s = get_game(cfg).new_state()
        fast = Counter(random_playout(s, random.Random(i)) for i in range(3000))
        ref = Counter(reference_playout(s, random.Random(10_000 + i)) for i in range(3000))
        for outcome in (1, 2, DRAW):
            assert abs(fast[outcome] - ref[outcome]) / 3000 < 0.05


def test_playout_respects_exact_k():
    # Each player can only fill a gap to make four in a row (an overline),
    # which does not win under exact-k rules, so every playout is a draw.
    s = State.from_string(GameConfig(2, 4, 3, exact_k=True), "XX.X/OO.O")
    for i in range(20):
        assert random_playout(s, random.Random(i)) == DRAW
    # Without exact-k, X wins whenever it fills its own gap first.
    s = State.from_string(GameConfig(2, 4, 3), "XX.X/OO.O")
    assert 1 in {random_playout(s, random.Random(i)) for i in range(20)}


def test_benchmarks_run():
    assert bench_push_pop(TIC_TAC_TOE, 0.01) > 1000
    assert bench_playouts(GOMOKU, 0.01) > 10
    assert bench_agent_move("rule", GameConfig(6, 6, 4), 1) < 1.0


def test_perf_and_gomoku_quick(tmp_path):
    from mnkbench.experiments.gomoku import exp_gomoku, exp_perf

    d = exp_perf(tmp_path, quick=True, log=lambda s: None)
    assert d["playouts"][0]["speedup"] > 0.5
    d = exp_gomoku(tmp_path, quick=True, log=lambda s: None)
    assert d["elo"]["ratings"]
    assert (tmp_path / "gomoku" / "gomoku_matrix.png").exists()

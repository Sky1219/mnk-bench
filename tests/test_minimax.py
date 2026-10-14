import pytest

from mnkbench.agents.minimax import MinimaxAgent, Solver
from mnkbench.agents.random_agent import RandomAgent
from mnkbench.game import TIC_TAC_TOE, GameConfig, State, get_game
from mnkbench.tournament import play_match


def test_tic_tac_toe_is_a_draw():
    assert Solver(TIC_TAC_TOE).outcome(get_game(TIC_TAC_TOE).new_state()) == 0
    assert Solver(TIC_TAC_TOE, use_symmetry=False).value(get_game(TIC_TAC_TOE).new_state()) == 0


@pytest.mark.parametrize("cfg,expected", [
    (GameConfig(3, 4, 3), 1),   # first player wins (3,4,3)
    (GameConfig(4, 4, 4), 0),   # (4,4,4) is a draw
    (GameConfig(2, 2, 2), 1),
])
def test_known_values(cfg, expected):
    assert Solver(cfg).outcome(get_game(cfg).new_state()) == expected


def test_takes_immediate_win_and_blocks():
    s = State.from_string(TIC_TAC_TOE, "XX./OO./...")
    agent = MinimaxAgent(seed=0)
    assert agent.select_move(s) == 2  # X wins now
    s = State.from_string(TIC_TAC_TOE, "XX./.O./...")
    assert agent.select_move(s) == 2  # O must block
    solver = Solver(TIC_TAC_TOE)
    assert solver.outcome_preserving_moves(s) == [2]


def test_prefers_faster_win():
    s = State.from_string(TIC_TAC_TOE, "X.X/OO./...")
    scores = Solver(TIC_TAC_TOE).move_scores(s)
    assert scores[1] == max(scores.values())


def test_perfect_play_never_loses():
    r = play_match(TIC_TAC_TOE, MinimaxAgent(), RandomAgent(), 60, seed=3)
    assert r.wins_b == 0 and r.wins_a > 40
    r = play_match(TIC_TAC_TOE, MinimaxAgent(), MinimaxAgent(), 10, seed=3)
    assert r.draws == 10


def test_budget_abort_is_clean():
    cfg = GameConfig(5, 5, 4)
    agent = MinimaxAgent(max_nodes=2000)
    r = play_match(cfg, agent, RandomAgent(), 2, seed=0)
    assert r.aborted == 1 and "BudgetExceeded" in r.abort_reasons[0]


def test_estimate_grows_with_board():
    a = MinimaxAgent()
    small = a.estimate(TIC_TAC_TOE).time_per_move
    big = a.estimate(GameConfig(6, 6, 4)).time_per_move
    assert big > 1000 * small

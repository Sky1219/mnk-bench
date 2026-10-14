import pytest

from mnkbench.agents import make_agent
from mnkbench.agents.minimax import MinimaxAgent
from mnkbench.agents.tabular import QLearningAgent, TDAgent
from mnkbench.experiments.learning import train
from mnkbench.game import TIC_TAC_TOE, State, get_game
from mnkbench.metrics import wdl
from mnkbench.tournament import play_game


def test_q_update_on_win_and_bootstrap():
    a = QLearningAgent(seed=0, epsilon=0.0, alpha=0.5, gamma=0.9)
    a.new_game(get_game(TIC_TAC_TOE), 1)
    a.q = {b"k": {0: 0.0}}
    a._pending[1] = (b"k", 0)
    a.end_game(State.from_string(TIC_TAC_TOE, "XXX/OO./..."), 1)
    assert a.q[b"k"][0] == pytest.approx(0.5)
    # Bootstrapped update at the next decision point: target = gamma * max Q(s').
    a.new_game(get_game(TIC_TAC_TOE), 1)
    a._pending[1] = (b"k", 0)
    s = State.from_string(TIC_TAC_TOE, "X../.O./...")
    key, _ = a._canon.canonical(s.cells)
    a.q[key] = {m: 1.0 for m in range(9)}
    a.select_move(s)
    assert a.q[b"k"][0] == pytest.approx(0.5 + 0.5 * (0.9 - 0.5))


def test_td_values_move_towards_result():
    a = TDAgent(seed=0, epsilon=0.0, alpha=0.5)
    game = get_game(TIC_TAC_TOE)
    a.new_game(game, 2)
    s = State.from_string(TIC_TAC_TOE, "X../.../...")
    a.select_move(s)
    key = a._pending[2]
    lost = State.from_string(TIC_TAC_TOE, "XXX/OO./...")
    a.end_game(lost, 2)
    assert a.values[key] == pytest.approx(0.25)


@pytest.mark.parametrize("spec", ["qlearning", "td"])
def test_learners_reach_good_play(spec):
    agent = make_agent(spec, seed=1)
    train(agent, TIC_TAC_TOE, 2500, opponent="random+minimax")
    agent.training = False
    r = wdl(agent, MinimaxAgent(), TIC_TAC_TOE, 40, seed=3)
    assert r["loss"] <= 0.2
    r = wdl(agent, make_agent("random"), TIC_TAC_TOE, 40, seed=3)
    assert r["win"] >= 0.7


def test_frozen_agent_does_not_grow_table():
    a = QLearningAgent(seed=0)
    train(a, TIC_TAC_TOE, 50)
    n = len(a.q)
    a.training = False
    for _ in range(20):
        play_game(TIC_TAC_TOE, a, make_agent("random", seed=1))
    assert len(a.q) == n


def test_epsilon_decays_once_per_game():
    a = QLearningAgent(seed=0, epsilon=0.5, epsilon_decay=0.5, min_epsilon=0.01)
    train(a, TIC_TAC_TOE, 2, opponent="self")
    assert a.epsilon == pytest.approx(0.125)

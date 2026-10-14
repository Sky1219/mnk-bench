from mnkbench.agents import make_agent
from mnkbench.agents.rule_based import RuleBasedAgent, threat_cells
from mnkbench.experiments.suite import exp_h2h, wdl_markdown
from mnkbench.game import TIC_TAC_TOE, GameConfig, State
from mnkbench.metrics import optimal_move_rate
from mnkbench.tournament import play_match


def test_rule_wins_blocks_and_forks():
    a = RuleBasedAgent(seed=0)
    assert a.select_move(State.from_string(TIC_TAC_TOE, "XX./OO./...")) == 2
    assert a.select_move(State.from_string(TIC_TAC_TOE, "XX./.O./...")) == 2
    # X can fork with a corner: X at a1 and c3 (O centre, O b1)... use a classic
    s = State.from_string(TIC_TAC_TOE, "X../.O./..X")  # O to move must not play a corner
    mv = a.select_move(s)
    assert mv in (1, 3, 5, 7)


def test_threat_cells():
    s = State.from_string(TIC_TAC_TOE, "X../.../..O")
    assert threat_cells(s, 1, 1) == {2}
    assert threat_cells(s, 4, 1) == set()  # 8 is taken by O


def test_rule_is_perfect_on_tic_tac_toe():
    r = optimal_move_rate(make_agent("rule"), TIC_TAC_TOE, 60, seed=2)
    assert r["optimal_rate"] == 1.0
    m = play_match(TIC_TAC_TOE, make_agent("rule"), make_agent("minimax"), 20, seed=1)
    assert m.wins_b == 0


def test_rule_beats_random_on_bigger_boards():
    for cfg in [GameConfig(6, 6, 4), GameConfig(6, 7, 4, gravity=True)]:
        m = play_match(cfg, make_agent("rule"), make_agent("random"), 20, seed=1)
        assert m.wins_a >= 18


def test_h2h_experiment_quick(tmp_path):
    data = exp_h2h(tmp_path, quick=True)
    assert set(data["agents"]) >= {"random", "rule", "minimax", "menace"}
    assert (tmp_path / "h2h" / "h2h_matrix.png").exists()
    assert "| agent |" in wdl_markdown(data)

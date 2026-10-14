from mnkbench.agents.menace import MenaceAgent, count_matchboxes
from mnkbench.agents.minimax import MinimaxAgent
from mnkbench.experiments.learning import learning_curve, train
from mnkbench.game import TIC_TAC_TOE, State, get_game
from mnkbench.metrics import wdl


def test_matchbox_count_matches_michie():
    assert count_matchboxes(TIC_TAC_TOE, player=1) == 304
    assert count_matchboxes(TIC_TAC_TOE, player=1, use_symmetry=False, skip_forced=False) > 2000


def test_menace_bead_updates():
    agent = MenaceAgent(seed=0)
    game = get_game(TIC_TAC_TOE)
    agent.new_game(game, 1)
    s = game.new_state()
    mv = agent.select_move(s)
    assert s.is_legal(mv)
    key = next(iter(agent.boxes))
    box = agent.boxes[key]
    assert len(box) == 3 and all(b == 4 for b in box.values())
    (k, cm), = agent._trail[1]
    won = State.from_string(TIC_TAC_TOE, "XXX/OO./...")
    agent.end_game(won, 1)
    assert agent.boxes[k][cm] == 4 + agent.win_reward
    agent.new_game(game, 1)
    agent.select_move(game.new_state())
    (k2, cm2), = agent._trail[1]
    lost = State.from_string(TIC_TAC_TOE, "XX./OOO/X..")
    before = agent.boxes[k2][cm2]
    agent.end_game(lost, 1)
    assert agent.boxes[k2][cm2] == before - 1


def test_menace_symmetry_keeps_boxes_near_michie():
    agent = MenaceAgent(seed=1)
    train(agent, TIC_TAC_TOE, 1500, opponent="random", sides="x")
    assert 150 <= agent.num_boxes <= 304


def test_menace_learns_against_perfect_play():
    agent = MenaceAgent(seed=0)
    before = wdl(agent, MinimaxAgent(), TIC_TAC_TOE, 40, seed=5)
    train(agent, TIC_TAC_TOE, 1500, opponent="minimax")
    agent.training = False
    after = wdl(agent, MinimaxAgent(), TIC_TAC_TOE, 40, seed=5)
    assert after["loss"] < before["loss"]
    assert after["loss"] <= 0.2


def test_learning_curve_shape():
    run = learning_curve("menace", TIC_TAC_TOE, total_games=200, eval_every=100, eval_games=10)
    assert [cp["games"] for cp in run["checkpoints"]] == [0, 100, 200]
    cp = run["checkpoints"][-1]
    assert {"vs_random", "vs_minimax", "optimal_rate", "table_size"} <= set(cp)


def test_symmetry_reduces_boxes():
    sym, nosym = MenaceAgent(seed=0), MenaceAgent(seed=0, use_symmetry=False)
    train(sym, TIC_TAC_TOE, 600, opponent="random")
    train(nosym, TIC_TAC_TOE, 600, opponent="random")
    assert sym.num_boxes * 2 < nosym.num_boxes


def test_symmetry_experiment_quick(tmp_path):
    from mnkbench.experiments.suite import exp_symmetry

    data = exp_symmetry(tmp_path, quick=True)
    counts = {(c["player"], c["symmetry"], c["skip_forced"]): c["matchboxes"] for c in data["matchbox_counts"]}
    assert counts[("X", True, True)] == 304
    assert (tmp_path / "symmetry" / "symmetry_learning.png").exists()

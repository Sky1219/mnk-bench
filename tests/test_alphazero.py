import pytest

torch = pytest.importorskip("torch")

from mnkbench.agents import make_agent  # noqa: E402
from mnkbench.agents.alphazero import (  # noqa: E402
    DATA_DIR,
    AlphaZeroAgent,
    Evaluator,
    PolicyValueNet,
    PUCTSearch,
    _augment,
    checkpoint_name,
    encode,
    load_checkpoint,
    train,
)
from mnkbench.game import TIC_TAC_TOE, GameConfig, State, get_game  # noqa: E402
from mnkbench.metrics import optimal_move_rate  # noqa: E402
from mnkbench.symmetry import symmetries  # noqa: E402


def test_network_shapes_any_board():
    net = PolicyValueNet(8, 1)
    for cfg in (TIC_TAC_TOE, GameConfig(4, 5, 3)):
        s = get_game(cfg).new_state()
        x = torch.tensor([encode(s)], dtype=torch.float32)
        logits, v = net(x)
        assert logits.shape == (1, cfg.size) and v.shape == (1,)
        assert -1 <= float(v[0].detach()) <= 1


def test_encode_is_from_side_to_move():
    s = State.from_string(TIC_TAC_TOE, "X../.../...")
    planes = encode(s)  # O to move: X stones are the opponent's
    assert planes[0][0][0] == 0.0 and planes[1][0][0] == 1.0
    assert planes[2][0][0] == 0.0 and planes[2][1][1] == 1.0
    assert planes[3][0][0] == 0.0


def test_puct_finds_win_with_untrained_net():
    torch.manual_seed(0)
    net = PolicyValueNet(8, 1).eval()
    s = State.from_string(TIC_TAC_TOE, "XX./OO./...")
    search = PUCTSearch(Evaluator(net, s.game), simulations=200)
    visits = search.run(s)
    assert max(visits, key=visits.get) == 2


def test_augment_preserves_distribution():
    s = get_game(TIC_TAC_TOE).new_state().play(0)
    pi = [0.0] * 9
    pi[4], pi[8] = 0.75, 0.25
    out = list(_augment(encode(s), pi, symmetries(TIC_TAC_TOE), 3, 3))
    assert len(out) == 8
    for _planes, p in out:
        assert sum(p) == pytest.approx(1.0)
        assert p[4] == pytest.approx(0.75)  # the centre is fixed by every symmetry


def test_tiny_training_run_and_checkpoint(tmp_path):
    run = train(TIC_TAC_TOE, iterations=1, games_per_iter=3, simulations=8, channels=8, blocks=1,
                epochs=1, eval_games=2, out_dir=tmp_path, log=lambda s: None)
    assert len(run["history"]) == 2 and run["history"][-1]["games"] == 3
    path = tmp_path / checkpoint_name(TIC_TAC_TOE)
    net, meta = load_checkpoint(path)
    assert meta["config"]["k"] == 3
    agent = AlphaZeroAgent(checkpoint=str(path), simulations=8, seed=0)
    s = get_game(TIC_TAC_TOE).new_state()
    assert s.is_legal(agent.select_move(s))


@pytest.mark.skipif(not (DATA_DIR / checkpoint_name(TIC_TAC_TOE)).exists(), reason="no bundled checkpoint")
def test_bundled_tic_tac_toe_checkpoint_plays_well():
    agent = make_agent("alphazero:simulations=50", seed=0)
    r = optimal_move_rate(agent, TIC_TAC_TOE, 10, seed=1)
    assert agent.loaded_from is not None
    assert r["optimal_rate"] >= 0.9


def test_untrained_agent_and_estimate():
    agent = AlphaZeroAgent(simulations=4, seed=0)
    cfg = GameConfig(5, 5, 4)
    s = get_game(cfg).new_state()
    assert s.is_legal(agent.select_move(s))
    assert agent.estimate(cfg).time_per_move > 0


def test_quick_experiment(tmp_path):
    from mnkbench.experiments.az_exp import exp_alphazero

    d = exp_alphazero(tmp_path, quick=True, log=lambda s: None)
    assert d["runs"][0]["history"]
    assert (tmp_path / "alphazero" / "alphazero_training.png").exists()

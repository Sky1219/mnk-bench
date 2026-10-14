import math

from mnkbench.agents.minimax import MinimaxAgent
from mnkbench.agents.random_agent import RandomAgent
from mnkbench.experiments.learning import learning_curve
from mnkbench.game import TIC_TAC_TOE
from mnkbench.plots import plot_learning_curves, plot_score_matrix
from mnkbench.results import load_json, save_csv, save_json
from mnkbench.tournament import round_robin


def test_json_roundtrip_handles_nan(tmp_path):
    p = save_json({"a": math.nan, "b": (1, 2)}, tmp_path / "x.json")
    assert load_json(p) == {"a": None, "b": [1, 2]}


def test_csv(tmp_path):
    p = save_csv([{"a": 1}, {"a": 2, "b": 3}], tmp_path / "x.csv")
    assert p.read_text().splitlines()[0] == "a,b"


def test_plots_from_saved_data(tmp_path):
    run = learning_curve("menace", TIC_TAC_TOE, total_games=40, eval_every=20, eval_games=4)
    save_json([run], tmp_path / "runs.json")
    runs = load_json(tmp_path / "runs.json")
    out = plot_learning_curves(runs, tmp_path / "lc.png")
    assert out.stat().st_size > 1000
    t = round_robin(TIC_TAC_TOE, {"random": RandomAgent, "minimax": MinimaxAgent}, 4)
    save_json(t.to_dict(), tmp_path / "t.json")
    out = plot_score_matrix(load_json(tmp_path / "t.json"), tmp_path / "m.png")
    assert out.exists()

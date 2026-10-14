import json

import pytest

from mnkbench.cli import main
from mnkbench.report import build_report


def test_agents_and_play(capsys):
    assert main(["agents"]) == 0
    assert "minimax" in capsys.readouterr().out
    assert main(["play", "minimax", "rule"]) == 0
    assert "draw" in capsys.readouterr().out


def test_match_json(tmp_path, capsys):
    out = tmp_path / "m.json"
    assert main(["match", "rule", "random", "--games", "4", "--json", str(out)]) == 0
    d = json.loads(out.read_text())
    assert d["n_games"] == 4 and len(d["games"]) == 4


def test_tournament_and_feasibility(tmp_path, capsys):
    out = tmp_path / "t.json"
    assert main(["tournament", "--agents", "random", "rule", "--games", "4", "--out", str(out), "--plot"]) == 0
    assert "Elo" in capsys.readouterr().out
    assert json.loads(out.read_text())["elo"]["ratings"]["random"] == pytest.approx(1000)
    assert main(["feasibility", "--agents", "minimax", "random", "--boards", "3,3,3", "6,6,4",
                 "--explain", "--out", str(tmp_path / "f.json")]) == 0
    text = capsys.readouterr().out
    assert "refused" in text and (tmp_path / "f.png").exists()


def test_solve(capsys):
    assert main(["solve", "--position", "XX./OO./..."]) == 0
    assert "win" in capsys.readouterr().out
    assert main(["solve", "--board", "6,6,4"]) == 2


def test_train_flag_and_budget(capsys):
    assert main(["match", "qlearning", "random", "--games", "2", "--train", "50"]) == 0
    assert main(["match", "alphabeta:depth=9", "random", "--board", "9,9,5", "--games", "1", "--budget",
                 "--time", "0.2"]) == 0


def test_human_agent():
    from mnkbench.agents.human import HumanAgent
    from mnkbench.game import TIC_TAC_TOE, get_game

    answers = iter(["zz", "a1", "b2"])
    h = HumanAgent(input_fn=lambda prompt: next(answers), output_fn=lambda s: None)
    s = get_game(TIC_TAC_TOE).new_state().play(0)
    assert h.select_move(s) == 4  # "zz" invalid, a1 taken, b2 ok


def test_experiment_replot_report(tmp_path, capsys):
    out = tmp_path / "r"
    assert main(["experiment", "menace", "h2h", "--quick", "--out", str(out)]) == 0
    assert main(["replot", "--out", str(out)]) == 0
    html = build_report(out).read_text()
    assert "MENACE learns tic-tac-toe" in html and "menace/menace_learning.png" in html
    assert "Head to head" in html


def test_empty_report(tmp_path):
    assert "No results found" in build_report(tmp_path).read_text()


def test_run_all_skips_missing_optional_dependency(tmp_path, monkeypatch):
    from mnkbench.experiments import suite

    def needs_torch(*a, **k):
        raise ImportError("needs PyTorch")

    monkeypatch.setattr(suite, "EXPERIMENTS", {"alphazero": needs_torch})
    logs = []
    suite.run(None, tmp_path, quick=True, log=logs.append)
    assert any("skipping alphazero" in m for m in logs)
    with pytest.raises(ImportError):
        suite.run(["alphazero"], tmp_path, quick=True, log=logs.append)

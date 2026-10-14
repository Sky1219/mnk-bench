from mnkbench.agents.minimax import MinimaxAgent
from mnkbench.agents.random_agent import RandomAgent
from mnkbench.game import TIC_TAC_TOE, GameConfig
from mnkbench.tournament import play_game, play_match, round_robin


def test_play_game_records():
    rec = play_game(TIC_TAC_TOE, RandomAgent(seed=1), RandomAgent(seed=2), "a", "b")
    assert rec.winner is not None and rec.aborted is None
    assert len(rec.move_times["a"]) + len(rec.move_times["b"]) == rec.length


def test_match_is_reproducible():
    r1 = play_match(GameConfig(4, 4, 3), RandomAgent(), RandomAgent(), 30, seed=7, keep_games=True)
    r2 = play_match(GameConfig(4, 4, 3), RandomAgent(), RandomAgent(), 30, seed=7, keep_games=True)
    assert [g.moves for g in r1.games] == [g.moves for g in r2.games]
    r3 = play_match(GameConfig(4, 4, 3), RandomAgent(), RandomAgent(), 30, seed=8, keep_games=True)
    assert [g.moves for g in r1.games] != [g.moves for g in r3.games]


def test_match_alternates_sides():
    r = play_match(TIC_TAC_TOE, MinimaxAgent(), RandomAgent(), 10, seed=0)
    assert sum(r.a_as_x.values()) == 5 and sum(r.a_as_o.values()) == 5
    assert r.n_games == 10


def test_round_robin():
    t = round_robin(TIC_TAC_TOE, {"random": RandomAgent, "minimax": MinimaxAgent}, 10, seed=1)
    assert len(t.matches) == 1
    m = t.score_matrix()
    assert m["minimax"]["random"] > 0.8
    rows = {r["agent"]: r for r in t.wdl_table()}
    assert rows["minimax"]["loss"] == 0
    d = t.to_dict()
    assert d["agents"] == ["random", "minimax"]

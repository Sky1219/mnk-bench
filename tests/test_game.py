import pytest

from mnkbench.game import (
    CONNECT_FOUR,
    DRAW,
    EMPTY,
    TIC_TAC_TOE,
    GameConfig,
    IllegalMoveError,
    O,
    State,
    X,
    get_game,
)


def test_config_parse_and_label():
    assert GameConfig.parse("3,3,3") == TIC_TAC_TOE
    assert GameConfig.parse("(6,7,4)+gravity") == CONNECT_FOUR
    cfg = GameConfig.parse("15x15x5 exact")
    assert cfg.exact_k and cfg.size == 225
    assert cfg.label == "(15,15,5)+exact"
    with pytest.raises(ValueError):
        GameConfig.parse("3,3")


def test_new_state_and_legal_moves(ttt):
    s = get_game(ttt).new_state()
    assert s.to_move == X
    assert s.legal_moves() == list(range(9))
    s.push(4)
    assert s.to_move == O and 4 not in s.legal_moves()
    assert s.pop() == 4
    assert s.cells == [EMPTY] * 9


def test_row_column_diagonal_wins(ttt):
    for moves, winner in [
        ([0, 3, 1, 4, 2], X),  # top row
        ([0, 1, 3, 2, 6], X),  # left column
        ([0, 1, 4, 2, 8], X),  # diagonal
        ([0, 2, 1, 4, 5, 6], O),  # anti-diagonal for O
    ]:
        s = get_game(ttt).new_state()
        for mv in moves:
            s = s.play(mv)
        assert s.winner == winner
        assert s.legal_moves() == []


def test_draw(ttt):
    s = get_game(ttt).new_state()
    for mv in [0, 1, 2, 4, 3, 5, 7, 6, 8]:
        s = s.play(mv)
    assert s.winner == DRAW


def test_illegal_move_raises(ttt):
    s = get_game(ttt).new_state().play(0)
    with pytest.raises(IllegalMoveError):
        s.play(0)
    with pytest.raises(IllegalMoveError):
        s.play(9)


def test_gravity_moves_fall():
    cfg = GameConfig(4, 4, 3, gravity=True)
    s = get_game(cfg).new_state()
    # Bottom row is row 3: cells 12..15.
    assert s.legal_moves() == [12, 13, 14, 15]
    assert not s.is_legal(0)
    s = s.play(13)
    assert 9 in s.legal_moves() and 13 not in s.legal_moves()
    # Fill a column completely.
    for mv in [9, 5, 1]:
        s = s.play(mv)
    assert all(mv % 4 != 1 for mv in s.legal_moves())
    s.pop()
    assert 1 in s.legal_moves()


def test_connect_four_vertical_win():
    s = get_game(CONNECT_FOUR).new_state()
    n = 7
    for r in range(3):
        s = s.play((5 - r) * n + 0)  # X in column a
        s = s.play((5 - r) * n + 1)  # O in column b
    s = s.play(2 * n + 0)
    assert s.winner == X


def test_exact_k_overline_does_not_win():
    cfg = GameConfig(1, 7, 3, exact_k=True)
    # X X . X with X to fill the gap makes 4 in a row: an overline, no win.
    s = State.from_string(cfg, "XX.X.OO")
    assert s.to_move == O or s.to_move == X
    assert not s.would_win(2, X)
    free = GameConfig(1, 7, 3)
    assert State.from_string(free, "XX.X.OO").would_win(2, X)


def test_exact_k_exact_run_wins():
    cfg = GameConfig(1, 6, 3, exact_k=True)
    s = State.from_string(cfg, "XX..OO")
    assert s.would_win(2, X)
    assert s.would_win(3, O)


def test_from_string_and_render(ttt):
    s = State.from_string(ttt, "X.O/.X./...")
    assert s.to_move == O
    assert s.winner is None
    assert s.winning_moves(X) == [8]
    text = s.render()
    assert "a b c" in text and "1 X . O" in text
    won = State.from_string(ttt, "XXX/OO./...")
    assert won.winner == X


def test_cell_names(ttt):
    g = get_game(ttt)
    assert g.cell_name(0) == "a1"
    assert g.cell_name(5) == "c2"
    assert g.parse_cell("c2") == 5
    assert g.parse_cell(" B3 ") == 7
    with pytest.raises(ValueError):
        g.parse_cell("d1")


def test_windows_count():
    g = get_game(TIC_TAC_TOE)
    assert len(g.windows) == 8
    g = get_game(GameConfig(15, 15, 5))
    assert len(g.windows) == 2 * 15 * 11 + 2 * 11 * 11


def test_random_playouts_are_consistent():
    import random

    rng = random.Random(0)
    for cfg in [GameConfig(4, 5, 3), GameConfig(5, 5, 4, gravity=True), GameConfig(6, 6, 4, exact_k=True)]:
        g = get_game(cfg)
        for _ in range(30):
            s = g.new_state()
            while not s.is_terminal:
                s.push(rng.choice(s.legal_moves()))
            # Recomputing the winner from scratch must agree with the incremental check.
            fresh = State.from_string(cfg, "/".join(
                "".join(".XO"[v] for v in s.cells[r * cfg.n:(r + 1) * cfg.n]) for r in range(cfg.m)))
            if s.winner == DRAW:
                assert fresh.winner == DRAW
            else:
                assert fresh.winner == s.winner

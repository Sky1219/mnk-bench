from mnkbench.game import TIC_TAC_TOE, GameConfig, State
from mnkbench.symmetry import Canonicalizer, symmetries


def test_symmetry_counts():
    assert len(symmetries(TIC_TAC_TOE)) == 8
    assert len(symmetries(GameConfig(3, 4, 3))) == 4
    assert len(symmetries(GameConfig(6, 7, 4, gravity=True))) == 2


def test_symmetries_are_permutations():
    for cfg in [TIC_TAC_TOE, GameConfig(3, 5, 3), GameConfig(4, 4, 3)]:
        for p in symmetries(cfg):
            assert sorted(p) == list(range(cfg.size))


def test_equivalent_positions_share_canonical_key():
    c = Canonicalizer(TIC_TAC_TOE)
    corners = [State.from_string(TIC_TAC_TOE, d) for d in ["X../.../...", "..X/.../...", ".../.../X..", ".../.../..X"]]
    keys = {c.canonical(s.cells)[0] for s in corners}
    assert len(keys) == 1
    edge = State.from_string(TIC_TAC_TOE, ".X./.../...")
    assert c.canonical(edge.cells)[0] not in keys


def test_move_round_trip():
    c = Canonicalizer(TIC_TAC_TOE)
    s = State.from_string(TIC_TAC_TOE, "..X/.O./...")
    key, sym = c.canonical(s.cells)
    for mv in s.legal_moves():
        assert c.from_canon(sym, c.to_canon(sym, mv)) == mv
    # Applying the move in the canonical frame matches transforming the moved position.
    for mv in s.legal_moves():
        child = s.play(mv)
        canon_cells = list(key)
        canon_cells[c.to_canon(sym, mv)] = s.to_move
        assert c.canonical(child.cells)[0] == c.canonical(canon_cells)[0]


def test_orbit_reps_on_empty_board():
    c = Canonicalizer(TIC_TAC_TOE)
    key, _ = c.canonical([0] * 9)
    assert len(c.move_orbit_reps(key, list(range(9)))) == 3  # corner, edge, centre


def test_no_symmetry_mode():
    c = Canonicalizer(TIC_TAC_TOE, use_symmetry=False)
    s = State.from_string(TIC_TAC_TOE, "X../.../...")
    assert c.canonical(s.cells) == (bytes(s.cells), 0)

import pytest

from mnkbench.game import GameConfig


@pytest.fixture
def ttt():
    return GameConfig(3, 3, 3)

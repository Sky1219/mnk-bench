"""mnk-bench: where does each game-playing AI fall apart as the board grows?"""

from .game import (
    CONNECT_FOUR,
    DRAW,
    EMPTY,
    GOMOKU,
    TIC_TAC_TOE,
    Game,
    GameConfig,
    IllegalMoveError,
    O,
    State,
    X,
    get_game,
)

__version__ = "0.1.0"

__all__ = [
    "CONNECT_FOUR",
    "DRAW",
    "EMPTY",
    "GOMOKU",
    "TIC_TAC_TOE",
    "Game",
    "GameConfig",
    "IllegalMoveError",
    "O",
    "State",
    "X",
    "get_game",
    "__version__",
]

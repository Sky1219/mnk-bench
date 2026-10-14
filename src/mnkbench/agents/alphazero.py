"""AlphaZero-style agent: a policy/value network guiding MCTS, trained by self-play.

* :class:`PolicyValueNet` is a small residual CNN. It is fully convolutional
  (policy = 1x1 conv per cell, value = global average pool + MLP), so the
  same architecture works on any m x n board.
* :class:`PUCTSearch` is AlphaZero's MCTS: each simulation descends by
  ``Q + c_puct * P * sqrt(N_parent) / (1 + N_child)``, evaluates the leaf with
  the network instead of a random playout, and backs the value up.
* :func:`train` runs the self-play loop: play games with search (with
  Dirichlet noise and temperature for exploration), store (position, visit
  distribution, outcome) triples augmented by board symmetries, fit the
  network, repeat, evaluating after every iteration.

PyTorch is an optional dependency (``pip install mnk-bench[alphazero]``).
Small networks (2 residual blocks, 32 channels) train on a laptop CPU in
minutes on tic-tac-toe and other small boards.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

from ..game import DRAW, Game, GameConfig, State, get_game
from ..symmetry import symmetries
from .base import Agent, Budget, CostEstimate

try:  # optional dependency
    import torch
    from torch import nn
    from torch.nn import functional as F

    HAVE_TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    torch = None
    nn = None
    F = None
    HAVE_TORCH = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "alphazero"


def require_torch() -> None:
    if not HAVE_TORCH:
        raise ImportError("the AlphaZero agent needs PyTorch: pip install 'mnk-bench[alphazero]'")


def has_bundled_checkpoint(config: GameConfig) -> bool:
    """True if PyTorch is installed and a trained network ships for ``config``."""
    return HAVE_TORCH and (DATA_DIR / checkpoint_name(config)).exists()


def checkpoint_name(config: GameConfig) -> str:
    slug = f"{config.m}-{config.n}-{config.k}" + ("-gravity" if config.gravity else "") + (
        "-exact" if config.exact_k else "")
    return f"az_{slug}.pt"


# ------------------------------------------------------------------ network
if HAVE_TORCH:

    class ResBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(x + y)

    class PolicyValueNet(nn.Module):
        """Input planes: own stones, opponent stones, legal moves, side-to-move is X."""

        IN_PLANES = 4

        def __init__(self, channels: int = 32, blocks: int = 2):
            super().__init__()
            self.channels = channels
            self.blocks = blocks
            self.stem = nn.Sequential(
                nn.Conv2d(self.IN_PLANES, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
            self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
            self.policy = nn.Conv2d(channels, 1, 1)
            self.value = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())

        def forward(self, x):
            h = self.body(self.stem(x))
            logits = self.policy(h).flatten(1)  # (B, m*n)
            v = self.value(h.mean(dim=(2, 3))).squeeze(1)
            return logits, v


def count_params(net) -> int:
    return sum(p.numel() for p in net.parameters())


def encode(state: State) -> list[list[list[float]]]:
    """4 x m x n planes from the side to move's point of view."""
    game = state.game
    m, n = game.m, game.n
    me = state.to_move
    cells = state.cells
    own = [[0.0] * n for _ in range(m)]
    opp = [[0.0] * n for _ in range(m)]
    legal = [[0.0] * n for _ in range(m)]
    for i, v in enumerate(cells):
        if v:
            r, c = divmod(i, n)
            if v == me:
                own[r][c] = 1.0
            else:
                opp[r][c] = 1.0
    for mv in state.legal_moves():
        r, c = divmod(mv, n)
        legal[r][c] = 1.0
    turn = [[1.0 if me == 1 else 0.0] * n for _ in range(m)]
    return [own, opp, legal, turn]


class Evaluator:
    """Wraps a network with a position cache (valid while weights are fixed)."""

    def __init__(self, net, game: Game):
        self.net = net
        self.game = game
        self.cache: dict[tuple[bytes, int], tuple[dict[int, float], float]] = {}
        self.calls = 0

    def clear(self) -> None:
        self.cache.clear()

    def __call__(self, state: State) -> tuple[dict[int, float], float]:
        key = (bytes(state.cells), state.to_move)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        self.calls += 1
        moves = state.legal_moves()
        with torch.no_grad():
            x = torch.tensor([encode(state)], dtype=torch.float32)
            logits, v = self.net(x)
            logits = logits[0]
            sel = logits[moves]
            p = torch.softmax(sel, dim=0).tolist()
        out = ({mv: pi for mv, pi in zip(moves, p)}, float(v[0]))
        self.cache[key] = out
        return out


# --------------------------------------------------------------------- PUCT
class PNode:
    __slots__ = ("prior", "visits", "value_sum", "children", "expanded")

    def __init__(self, prior: float):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0  # from the view of the player who moved INTO this node
        self.children: dict[int, PNode] = {}
        self.expanded = False


class PUCTSearch:
    def __init__(self, evaluator: Evaluator, simulations: int = 100, c_puct: float = 1.5,
                 rng: random.Random | None = None):
        self.evaluate = evaluator
        self.simulations = simulations
        self.c_puct = c_puct
        self.rng = rng or random.Random()

    def _expand(self, node: PNode, state: State) -> float:
        priors, value = self.evaluate(state)
        for mv, p in priors.items():
            node.children[mv] = PNode(p)
        node.expanded = True
        return value

    def run(self, state: State, noise: float = 0.0, alpha: float = 0.3) -> dict[int, int]:
        """Return visit counts per root move."""
        root = PNode(1.0)
        self._expand(root, state)
        if noise > 0 and root.children:
            gammas = [self.rng.gammavariate(alpha, 1.0) for _ in root.children]
            total = sum(gammas) or 1.0
            for ch, g in zip(root.children.values(), gammas):
                ch.prior = (1 - noise) * ch.prior + noise * g / total
        sim = state.copy()
        c = self.c_puct
        for _ in range(self.simulations):
            node = root
            path = [root]
            depth = 0
            while node.expanded and node.children:
                sqrt_n = math.sqrt(node.visits + 1)
                best, best_score = None, -1e9
                for mv, ch in node.children.items():
                    q = ch.value_sum / ch.visits if ch.visits else 0.0
                    u = q + c * ch.prior * sqrt_n / (1 + ch.visits)
                    if u > best_score:
                        best, best_score = mv, u
                node = node.children[best]
                sim.push(best)
                depth += 1
                path.append(node)
                if sim.winner is not None:
                    break
            if sim.winner is not None:
                # The player who just moved either won (+1 for them) or drew.
                value_for_mover = 0.0 if sim.winner == DRAW else 1.0
                leaf_value = -value_for_mover  # from the view of the player to move
            else:
                leaf_value = self._expand(node, sim)
            # Back up: node values are stored from the view of the player who
            # moved into them, i.e. the opponent of the player to move there.
            v = -leaf_value
            for n_ in reversed(path):
                n_.visits += 1
                n_.value_sum += v
                v = -v
            for _ in range(depth):
                sim.pop()
        return {mv: ch.visits for mv, ch in root.children.items()}


# -------------------------------------------------------------------- agent
class AlphaZeroAgent(Agent):
    name = "alphazero"

    def __init__(self, seed: int | None = None, checkpoint: str | None = None, simulations: int = 100,
                 c_puct: float = 1.5, channels: int = 32, blocks: int = 2, temperature: float = 0.0):
        super().__init__(seed)
        require_torch()
        self.checkpoint = checkpoint
        self.simulations = simulations
        self.c_puct = c_puct
        self.channels = channels
        self.blocks = blocks
        self.temperature = temperature
        self.net = None
        self._game: Game | None = None
        self._evaluator: Evaluator | None = None
        self.loaded_from: str | None = None

    def params(self) -> dict:
        return {"checkpoint": self.checkpoint, "simulations": self.simulations, "c_puct": self.c_puct,
                "channels": self.channels, "blocks": self.blocks, "temperature": self.temperature}

    def _setup(self, game: Game) -> None:
        if self._game is not None and self._game.config == game.config and self.net is not None:
            return
        self._game = game
        path = Path(self.checkpoint) if self.checkpoint else DATA_DIR / checkpoint_name(game.config)
        if path.exists():
            self.net, meta = load_checkpoint(path)
            self.loaded_from = str(path)
        else:
            if self.checkpoint:
                raise FileNotFoundError(f"no AlphaZero checkpoint at {path}")
            torch.manual_seed(self.rng.randrange(2**31))
            self.net = PolicyValueNet(self.channels, self.blocks)
            self.loaded_from = None  # untrained network
        self.net.eval()
        self._evaluator = Evaluator(self.net, game)

    def select_move(self, state: State) -> int:
        self._setup(state.game)
        legal = state.legal_moves()
        if len(legal) == 1:
            return legal[0]
        search = PUCTSearch(self._evaluator, self.simulations, self.c_puct, self.rng)
        visits = search.run(state)
        self.add_stat("moves", 1)
        return pick_move(visits, self.temperature, self.rng)

    def memory_bytes(self) -> float:
        if self.net is None:
            return 0.0
        return count_params(self.net) * 4.0 + len(self._evaluator.cache) * 200.0

    def estimate(self, config: GameConfig) -> CostEstimate:
        # One small-net forward pass on CPU is ~0.15ms + a little per cell.
        per_eval = 1.5e-4 + 2e-6 * config.size * (self.channels / 32) * max(1, self.blocks) / 2
        trained = (DATA_DIR / checkpoint_name(config)).exists() or bool(self.checkpoint)
        # Self-play training cost: ~games x moves x simulations x eval, with the
        # number of games needed growing with the board (rough power law).
        games_needed = 300 * (config.size / 9) ** 1.5
        train_time = 0.0 if trained else games_needed * (0.6 * config.size) * 50 * per_eval
        params = (4 * 9 + 2 * 2 * 9 * self.channels) * self.channels + 50 * self.channels
        return CostEstimate(time_per_move=self.simulations * per_eval,
                            memory_bytes=params * 4 + self.simulations * config.size * 64 + 450e6,
                            train_time=train_time,
                            notes="bundled checkpoint" if trained else "needs self-play training")

    def degrade(self, config: GameConfig, budget: Budget):
        est = self.estimate(config)
        per_eval = est.time_per_move / self.simulations
        sims = int(budget.time_per_move / per_eval)
        if sims < 8 or est.train_time > budget.train_time or est.memory_bytes > budget.memory_bytes:
            return None
        sims = min(sims, self.simulations)
        return {"simulations": sims}, CostEstimate(sims * per_eval, est.memory_bytes, est.train_time,
                                                  notes=f"{sims} simulations/move")


def pick_move(visits: dict[int, int], temperature: float, rng: random.Random) -> int:
    if temperature <= 1e-6:
        top = max(visits.values())
        return rng.choice([mv for mv, v in visits.items() if v == top])
    moves = list(visits)
    weights = [visits[m] ** (1.0 / temperature) for m in moves]
    return rng.choices(moves, weights=weights)[0]


def save_checkpoint(net, path: Path, config: GameConfig, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "channels": net.channels, "blocks": net.blocks,
                "config": config.to_dict(), "meta": meta}, path)


def load_checkpoint(path: Path):
    require_torch()
    data = torch.load(path, map_location="cpu", weights_only=False)
    net = PolicyValueNet(data["channels"], data["blocks"])
    net.load_state_dict(data["state_dict"])
    net.eval()
    return net, data


# ----------------------------------------------------------------- training
def _augment(planes, pi: list[float], perms, m: int, n: int):
    """Yield (planes, pi) under every board symmetry."""
    for perm in perms:
        new_planes = []
        for plane in planes:
            flat = [x for row in plane for x in row]
            out = [0.0] * (m * n)
            for cell, img in enumerate(perm):
                out[img] = flat[cell]
            new_planes.append([out[r * n:(r + 1) * n] for r in range(m)])
        new_pi = [0.0] * (m * n)
        for cell, img in enumerate(perm):
            new_pi[img] = pi[cell]
        yield new_planes, new_pi


def self_play_game(search: PUCTSearch, game: Game, temp_moves: int, noise: float, rng: random.Random):
    state = game.new_state()
    records = []
    while state.winner is None:
        visits = search.run(state, noise=noise)
        total = sum(visits.values())
        pi = [0.0] * game.size
        for mv, v in visits.items():
            pi[mv] = v / total
        records.append((encode(state), pi, state.to_move))
        mv = pick_move(visits, 1.0 if state.move_count < temp_moves else 0.0, rng)
        state.push(mv)
    out = []
    for planes, pi, player in records:
        z = 0.0 if state.winner == DRAW else (1.0 if state.winner == player else -1.0)
        out.append((planes, pi, z))
    return out, state


def evaluate_agent(agent: AlphaZeroAgent, config: GameConfig, games: int, seed: int) -> dict:
    from ..metrics import optimal_move_rate, wdl
    from .random_agent import RandomAgent
    from .rule_based import RuleBasedAgent

    out: dict[str, Any] = {}
    r = wdl(agent, RandomAgent(), config, games, seed)
    out["score_vs_random"] = r["win"] + 0.5 * r["draw"]
    out["win_vs_random"] = r["win"]
    r = wdl(agent, RuleBasedAgent(), config, games, seed + 1)
    out["score_vs_rule"] = r["win"] + 0.5 * r["draw"]
    from .minimax import MinimaxAgent, estimate_nodes

    if estimate_nodes(config) < 2e5:  # perfect play is cheap enough to compare against
        r = wdl(agent, MinimaxAgent(), config, games, seed + 2)
        out["nonloss_vs_minimax"] = 1 - r["loss"]
        out["optimal_rate"] = optimal_move_rate(agent, config, max(4, games // 2), seed=seed + 3)["optimal_rate"]
    return out


def train(
    config: GameConfig,
    iterations: int = 20,
    games_per_iter: int = 40,
    simulations: int = 50,
    channels: int = 32,
    blocks: int = 2,
    epochs: int = 4,
    batch_size: int = 64,
    lr: float = 2e-3,
    buffer_size: int = 20000,
    eval_games: int = 20,
    eval_simulations: int | None = None,
    seed: int = 0,
    out_dir: str | Path | None = None,
    log=print,
) -> dict:
    """Self-play training loop. Returns a run record with per-iteration history."""
    require_torch()
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    game = get_game(config)
    net = PolicyValueNet(channels, blocks)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    perms = symmetries(config)
    buffer: deque = deque(maxlen=buffer_size)
    temp_moves = max(2, config.size // 3)
    history = []
    t_start = time.perf_counter()
    games_played = 0
    eval_sims = eval_simulations or simulations
    out_dir = Path(out_dir) if out_dir else None

    def evaluate(it: int) -> dict:
        agent = AlphaZeroAgent(seed=seed + it, simulations=eval_sims, channels=channels, blocks=blocks)
        agent.net = net
        agent._game = game
        agent._evaluator = Evaluator(net, game)
        net.eval()
        return evaluate_agent(agent, config, eval_games, seed * 100 + it)

    ev = evaluate(0)
    history.append({"iteration": 0, "games": 0, "elapsed": 0.0, "loss": None, **ev})
    log(f"[az {config.label}] iter 0: {ev}")
    for it in range(1, iterations + 1):
        net.eval()
        evaluator = Evaluator(net, game)
        search = PUCTSearch(evaluator, simulations, rng=rng)
        outcomes = {1: 0, 2: 0, DRAW: 0}
        for _ in range(games_per_iter):
            samples, final = self_play_game(search, game, temp_moves, 0.25, rng)
            outcomes[final.winner] += 1
            for planes, pi, z in samples:
                for p2, pi2 in _augment(planes, pi, perms, game.m, game.n):
                    buffer.append((p2, pi2, z))
        games_played += games_per_iter

        net.train()
        data = list(buffer)
        steps = max(1, epochs * games_per_iter * int(0.6 * game.size) * len(perms) // batch_size)
        steps = min(steps, epochs * len(data) // batch_size + 1)
        total_loss = 0.0
        for _ in range(steps):
            batch = [data[rng.randrange(len(data))] for _ in range(batch_size)]
            x = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            pi = torch.tensor([b[1] for b in batch], dtype=torch.float32)
            z = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            logits, v = net(x)
            legal = x[:, 2].flatten(1) > 0
            logits = logits.masked_fill(~legal, -1e9)
            policy_loss = -(pi * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(v, z)
            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        net.eval()
        ev = evaluate(it)
        entry = {"iteration": it, "games": games_played, "elapsed": time.perf_counter() - t_start,
                 "loss": total_loss / steps, "train_steps": steps, "buffer": len(buffer),
                 "selfplay_outcomes": {"x": outcomes[1], "o": outcomes[2], "draw": outcomes[DRAW]}, **ev}
        history.append(entry)
        log(f"[az {config.label}] iter {it}: loss {entry['loss']:.3f}, {ev}, {entry['elapsed']:.0f}s")
        if out_dir:
            save_checkpoint(net, out_dir / checkpoint_name(config), config,
                            {"iteration": it, "games": games_played})
    run = {
        "config": config.label,
        "config_dict": config.to_dict(),
        "params": count_params(net),
        "hyper": {"iterations": iterations, "games_per_iter": games_per_iter, "simulations": simulations,
                  "channels": channels, "blocks": blocks, "epochs": epochs, "batch_size": batch_size,
                  "lr": lr, "eval_games": eval_games, "seed": seed},
        "history": history,
    }
    return run


def train_main(args) -> int:
    """Entry point for ``mnk-bench train-alphazero``."""
    from ..plots import plot_alphazero
    from ..results import metadata, save_json

    out = Path(args.out)
    run = train(args.board, iterations=args.iterations, games_per_iter=args.games,
                simulations=args.simulations, channels=args.channels, blocks=args.blocks,
                epochs=args.epochs, eval_games=args.eval_games, seed=args.seed, out_dir=out)
    save_json({"meta": metadata(seed=args.seed), "runs": [run]}, out / "alphazero.json")
    plot_alphazero([run], out / f"alphazero_{checkpoint_name(args.board)[3:-3]}.png")
    print(f"checkpoint: {out / checkpoint_name(args.board)}")
    return 0

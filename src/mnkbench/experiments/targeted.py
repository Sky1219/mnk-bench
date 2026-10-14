"""Targeted oracle labelling with submodular information measures (PRISM for games).

Research question
-----------------
A game-playing network has a specific weakness, e.g. it misses immediate
wins. We can ask an expensive oracle (here: the exact solver; in practice a
deep search) which moves are correct in a position, but each label costs
time, so we can only afford k labels. Given a handful of example positions
of the weakness (the *query set* Q), which k positions from a large
unlabelled pool should we label to fix it?

This mirrors PRISM's *targeted learning* setting (Kothawade et al., AAAI
2022): there, a classifier underperforms on a rare class and a few examples
of that class guide which unlabelled images to label. Here the "rare
classes" are three tactical motifs, each about 10-16% of positions, on which
a briefly trained network is weak:

* ``win``: the side to move has an immediate win (the net misses ~35%);
* ``block``: the opponent threatens to win and must be blocked;
* ``poison``: some legal move lets the opponent win at once (in a gravity
  game, playing directly below the opponent's threat cell).

The selector never sees motif labels or solver labels for the pool; it only
sees network embeddings and the query examples. Selection strategies:

* baselines: random, uncertainty (policy entropy, top-2 margin), pure
  diversity (facility location, log-determinant);
* PRISM submodular mutual information with Q: GCMI, FLMI, FLVMI, COM, LogDetMI;
* PRISM conditional MI with Q and a private set P of ordinary (non-motif)
  positions: FLCMI ("like the query, unlike the common case");
* an oracle reference that samples only true motif positions (it uses the
  hidden labels, so it is not a real method).

The selected positions are labelled by the solver and used to fine-tune the
network (plus the labelled Q, the same for every method). We measure the
network's raw-policy accuracy (does its top move preserve the game-theoretic
outcome?) on held-out decision positions of the target motif (target
accuracy) and on all held-out decision positions (overall accuracy), the
share of selected positions that belong to the motif (hit rate), and game
results.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..agents.base import Agent
from ..game import GameConfig, State, get_game
from ..results import load_json, metadata, save_csv, save_json
from ..submodular import cosine_kernel, greedy, lazy_greedy, make_function
from ..symmetry import Canonicalizer

Log = Callable[[str], None]

BOARD = GameConfig(5, 5, 4, gravity=True)
SMI_METHODS = ["GCMI", "FLMI", "FLVMI", "COM", "LogDetMI"]
METHODS = ["random", "entropy", "margin", "FL", "LogDet", *SMI_METHODS, "FLCMI", "oracle"]


TARGETS = ["win", "block", "poison"]


@dataclass
class Labeled:
    state: State
    good: list[int]  # outcome-preserving moves
    critical: bool  # some legal move changes the outcome
    outcome: int  # +1 / 0 / -1 for the side to move
    motif: str = "quiet"


def motif(st: State) -> str:
    """Tactical motif of a position (see the module docstring)."""
    me, opp = st.to_move, 3 - st.to_move
    if st.winning_moves(me):
        return "win"
    threats = st.winning_moves(opp)
    if len(threats) == 1:
        return "block"
    if len(threats) >= 2:
        return "lost"
    work = st.copy()
    for mv in st.legal_moves():
        work.push(mv)
        bad = work.winner is None and bool(work.winning_moves(opp))
        work.pop()
        if bad:
            return "poison"
    return "quiet"


# ----------------------------------------------------------------- data
def generate_positions(config: GameConfig, n: int, rng: random.Random, seen: set) -> list[State]:
    """Positions from games between a mix of agents (with 15% random moves),
    up to three per game, deduplicated up to symmetry."""
    from ..agents import make_agent

    game = get_game(config)
    canon = Canonicalizer(game)
    specs = ["random", "rule", "mcts:iterations=100", "alphabeta:depth=2"]
    agents = {s: make_agent(s, seed=rng.randrange(2**31)) for s in specs}
    out: list[State] = []
    while len(out) < n:
        a, b = rng.choice(specs), rng.choice(specs)
        s = game.new_state()
        traj = []
        while s.winner is None:
            if s.move_count:
                traj.append(s.copy())
            agent = agents[a if s.to_move == 1 else b]
            s.push(agent.select_move(s) if rng.random() > 0.15 else rng.choice(s.legal_moves()))
        for st in rng.sample(traj, min(3, len(traj))):
            key = (canon.canonical(st.cells)[0], st.to_move)
            if key in seen:
                continue
            seen.add(key)
            out.append(st)
            if len(out) >= n:
                break
    return out


def label(states: list[State], solver) -> list[Labeled]:
    out = []
    for st in states:
        good = solver.outcome_preserving_moves(st)
        out.append(Labeled(st, good, len(good) < len(st.legal_moves()), solver.outcome(st), motif(st)))
    return out


# ---------------------------------------------------------------- model
def _torch():
    import torch
    import torch.nn.functional as F

    return torch, F


def _x(states: list[State]):
    torch, _ = _torch()
    from ..agents.alphazero import encode

    return torch.tensor([encode(s) for s in states], dtype=torch.float32)


def embed(net, states: list[State]) -> np.ndarray:
    """Penultimate-layer features (residual tower output, flattened). They are
    ReLU outputs, so cosine similarities are non-negative."""
    torch, _ = _torch()
    net.eval()
    with torch.no_grad():
        h = net.body(net.stem(_x(states)))
    return h.flatten(1).numpy().astype(np.float64)


def board_features(states: list[State]) -> np.ndarray:
    """Raw board planes (own / opponent / legal), for the embedding ablation."""
    return _x(states)[:, :3].flatten(1).numpy().astype(np.float64) + 1e-3


def policy_probs(net, states: list[State]) -> list[np.ndarray]:
    torch, F = _torch()
    net.eval()
    with torch.no_grad():
        logits, _ = net(_x(states))
    out = []
    for i, s in enumerate(states):
        legal = s.legal_moves()
        p = F.softmax(logits[i, legal], dim=0).numpy()
        full = np.zeros(logits.shape[1])
        full[legal] = p
        out.append(full)
    return out


def top_moves(net, states: list[State]) -> list[int]:
    probs = policy_probs(net, states)
    return [int(np.argmax(p)) for p in probs]


def accuracy(net, data: list[Labeled]) -> float:
    if not data:
        return float("nan")
    moves = top_moves(net, [d.state for d in data])
    return sum(m in d.good for m, d in zip(moves, data)) / len(data)


def finetune(base, train: list[Labeled], unlabeled: list[State], steps: int, seed: int,
             lr: float = 1e-3, batch: int = 64, kd_weight: float = 1.0):
    """Fine-tune a copy of ``base`` on oracle labels.

    Policy target: uniform over the outcome-preserving moves, applied only to
    decision positions (where every move is equally good the oracle says
    nothing about which to prefer, and a uniform target would just flatten
    the policy). Value target: the solved outcome, for every position. A
    distillation term keeps the policy close to the base network on
    unlabelled pool positions, so every method is protected from forgetting
    in the same way. Mirror augmentation doubles the data.
    """
    torch, F = _torch()
    from ..symmetry import symmetries

    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = copy.deepcopy(base)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    game = train[0].state.game
    perms = symmetries(game.config)

    xs, pis, zs, ws = [], [], [], []
    for d in train:
        x = _x([d.state])[0]
        pi = torch.zeros(game.size)
        pi[d.good] = 1.0 / len(d.good)
        for perm in perms:
            idx = torch.tensor([0] * game.size)
            for cell, img in enumerate(perm):
                idx[img] = cell
            xs.append(x.flatten(1)[:, idx].reshape(x.shape))
            pis.append(pi[idx])
            zs.append(float(d.outcome))
            ws.append(1.0 if d.critical else 0.0)
    X, PI, Z, W = torch.stack(xs), torch.stack(pis), torch.tensor(zs), torch.tensor(ws)
    U = _x(unlabeled)
    base.eval()
    with torch.no_grad():
        base_logits_U, _ = base(U)
    net.train()
    for _ in range(steps):
        bi = torch.tensor([rng.randrange(len(X)) for _ in range(batch)])
        logits, v = net(X[bi])
        legal = X[bi][:, 2].flatten(1) > 0
        logp = F.log_softmax(logits.masked_fill(~legal, -1e9), dim=1)
        policy_loss = -(PI[bi] * logp).sum(1)
        w = W[bi]
        loss = (policy_loss * w).sum() / w.sum().clamp(min=1.0) + F.mse_loss(v, Z[bi])
        if kd_weight:
            ui = torch.tensor([rng.randrange(len(U)) for _ in range(batch)])
            lu, _ = net(U[ui])
            legal_u = U[ui][:, 2].flatten(1) > 0
            teacher = F.softmax(base_logits_U[ui].masked_fill(~legal_u, -1e9), dim=1)
            student = F.log_softmax(lu.masked_fill(~legal_u, -1e9), dim=1)
            loss = loss + kd_weight * F.kl_div(student, teacher, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    return net


# ------------------------------------------------------------ selection
def select(method: str, k: int, pool: list[Labeled], feats: np.ndarray, q_feats: np.ndarray,
           p_feats: np.ndarray, base_probs: list[np.ndarray], target: str, rng: np.random.Generator,
           eta: float = 1.0, nu: float = 1.0, S: np.ndarray | None = None) -> list[int]:
    n = len(pool)
    k = min(k, n)
    if method == "random":
        return list(rng.permutation(n)[:k])
    if method in ("entropy", "margin"):
        scores = []
        for p in base_probs:
            nz = p[p > 0]
            if method == "entropy":
                scores.append(float(-(nz * np.log(nz)).sum()))
            else:
                top = np.sort(nz)[::-1]
                scores.append(-(top[0] - (top[1] if len(top) > 1 else 0.0)))
        order = np.lexsort((rng.random(n), -np.array(scores)))
        return list(order[:k])
    if method == "oracle":
        hits = [i for i, d in enumerate(pool) if d.motif == target]
        hset = set(hits)
        rest = [i for i in range(n) if i not in hset]
        rng.shuffle(hits)
        rng.shuffle(rest)
        return (hits + rest)[:k]
    S = cosine_kernel(feats) if S is None else S
    S_VQ = cosine_kernel(feats, q_feats)
    S_QQ = cosine_kernel(q_feats)
    S_VP = cosine_kernel(feats, p_feats)
    f = make_function(method, S, S_VQ, S_QQ, S_VP, eta=eta, nu=nu)
    if hasattr(f, "gain"):
        return lazy_greedy(f, k, rng)
    return greedy(f, k, rng)


# ------------------------------------------------------------- one seed
@dataclass
class SeedData:
    pool: list[Labeled]
    dev: list[Labeled]
    test: list[Labeled]


def build_data(config: GameConfig, sizes: tuple[int, int, int], seed: int, solver, log: Log) -> SeedData:
    rng = random.Random(seed)
    seen: set = set()
    t0 = time.perf_counter()
    states = generate_positions(config, sum(sizes), rng, seen)
    rng.shuffle(states)
    n_pool, n_dev, _ = sizes
    labeled = label(states, solver)
    log(f"seed {seed}: {len(states)} positions generated and labelled in {time.perf_counter() - t0:.0f}s")
    return SeedData(labeled[:n_pool], labeled[n_pool:n_pool + n_dev], labeled[n_pool + n_dev:])


def run_seed(args: dict) -> dict:
    """Everything for one seed (runs in a worker process)."""
    import torch

    torch.set_num_threads(1)
    from ..agents.alphazero import load_checkpoint
    from ..agents.minimax import Solver

    seed = args["seed"]
    log = print if args.get("verbose") else (lambda s: None)
    config = GameConfig(**args["config"])
    base, _ = load_checkpoint(Path(args["base_checkpoint"]))
    solver = Solver(config)
    data = build_data(config, tuple(args["sizes"]), seed, solver, log)

    pool_states = [d.state for d in data.pool]
    base_probs = policy_probs(base, pool_states)
    test_moves = top_moves(base, [d.state for d in data.test])
    test_crit = [d for d in data.test if d.critical]
    feats_net = embed(base, pool_states)
    S_net = cosine_kernel(feats_net)
    feats_board = S_board = None
    unlabeled = [d.state for d in random.Random(seed).sample(data.pool, min(1000, len(data.pool)))]
    steps = args["steps"]
    budgets = sorted(args["budgets"])
    kmax = budgets[-1]
    kab = args["ablation_budget"]
    results = []

    def features(kind: str, items: list[Labeled]) -> np.ndarray:
        states = [d.state for d in items]
        return embed(base, states) if kind == "net" else board_features(states)

    def base_row(target: str, test_target: list[Labeled]) -> dict:
        row = {"method": "base", "target": target, "seed": seed, "k": 0,
               "acc_target": accuracy(base, test_target), "acc_critical": accuracy(base, test_crit),
               "hit_target": None}
        if args.get("games"):
            row.update(play_games(base, config, args["games"], seed))
        return row

    stats = {"seed": seed, "pool": len(data.pool), "dev": len(data.dev), "test": len(data.test),
             "pool_critical": float(np.mean([d.critical for d in data.pool])),
             "base_acc_critical": accuracy(base, test_crit)}
    for target in args["targets"]:
        in_target = np.array([d.motif == target for d in data.pool])
        test_target = [d for d in data.test if d.critical and d.motif == target]
        dev_target = [d for d in data.dev if d.motif == target]
        dev_other = [d for d in data.dev if d.motif == "quiet"]
        stats[f"pool_share_{target}"] = float(in_target.mean())
        stats[f"test_{target}"] = len(test_target)
        stats[f"base_acc_{target}"] = accuracy(base, test_target)
        stats[f"base_blunder_{target}"] = float(np.mean([m not in d.good for m, d in zip(test_moves, data.test)
                                                         if d.critical and d.motif == target]))

        def query(nq: int, target=target, dev_target=dev_target, dev_other=dev_other):
            r = random.Random(seed * 7919 + nq * 31 + TARGETS.index(target))
            return (r.sample(dev_target, min(nq, len(dev_target))), r.sample(dev_other, min(nq, len(dev_other))))

        def run(tag: dict, sel: list[int], Q: list[Labeled], with_games: bool = False,
                target=target, in_target=in_target, test_target=test_target) -> dict:
            train = [data.pool[i] for i in sel] + Q
            net = finetune(base, train, unlabeled, steps, seed=seed * 1000 + len(sel))
            row = {**tag, "target": target, "seed": seed, "k": len(sel),
                   "acc_target": accuracy(net, test_target), "acc_critical": accuracy(net, test_crit),
                   "hit_target": float(in_target[sel].mean()) if sel else None}
            if with_games and args.get("games"):
                row.update(play_games(net, config, args["games"], seed))
            return row

        Q_main, P_main = query(args["nq"])
        results.append(base_row(target, test_target))
        results.append(run({"method": "query-only"}, [], Q_main))
        q_feats, p_feats = features("net", Q_main), features("net", P_main)
        for method in args["methods"]:
            t0 = time.perf_counter()
            order = select(method, kmax, data.pool, feats_net, q_feats, p_feats, base_probs, target,
                           np.random.default_rng(seed + 17), S=S_net)
            sel_time = time.perf_counter() - t0
            for k in budgets:
                row = run({"method": method}, order[:k], Q_main, with_games=k == args.get("games_budget"))
                row["select_seconds"] = sel_time
                results.append(row)
            log(f"seed {seed} {target}: {method} done")

        if target not in args.get("ablation_targets", []):
            continue
        for eta in args.get("etas", []):
            for method in ("FLMI", "FLVMI", "COM", "LogDetMI"):
                order = select(method, kab, data.pool, feats_net, q_feats, p_feats, base_probs, target,
                               np.random.default_rng(seed + 17), eta=eta, S=S_net)
                results.append(run({"method": method, "ablation": "eta", "eta": eta}, order, Q_main))
        for nq in args.get("query_sizes", []):
            Q, P = query(nq)
            qf, pf = features("net", Q), features("net", P)
            for method in ("random", "GCMI", "FLMI", "FLVMI", "LogDetMI"):
                order = select(method, kab, data.pool, feats_net, qf, pf, base_probs, target,
                               np.random.default_rng(seed + 17), S=S_net)
                results.append(run({"method": method, "ablation": "query_size", "nq": len(Q)}, order, Q))
        if args.get("embedding_ablation"):
            if feats_board is None:
                feats_board = board_features(pool_states)
                S_board = cosine_kernel(feats_board)
            qb, pb = features("board", Q_main), features("board", P_main)
            for method in ("FL", "GCMI", "FLMI", "FLVMI", "LogDetMI"):
                order = select(method, kab, data.pool, feats_board, qb, pb, base_probs, target,
                               np.random.default_rng(seed + 17), S=S_board)
                results.append(run({"method": method, "ablation": "embedding", "embedding": "board"},
                                   order, Q_main))
    return {"seed": seed, "rows": results, "stats": stats}


class PolicyAgent(Agent):
    """Plays the network's top policy move (no search)."""

    name = "policy"

    def __init__(self, net, seed: int | None = None):
        super().__init__(seed)
        self.net = net

    def select_move(self, state: State) -> int:
        return top_moves(self.net, [state])[0]


def play_games(net, config: GameConfig, games: int, seed: int, opening_plies: int = 4) -> dict:
    """Policy-only play against perfect play and depth-2 alpha-beta.

    A greedy policy against near-deterministic opponents replays almost the
    same game every time, so each game starts from a different random
    opening of ``opening_plies`` moves (the same openings for every method),
    with the network alternating sides.
    """
    from ..agents import make_agent
    from ..tournament import play_game

    game = get_game(config)
    rng = random.Random(seed * 101 + 7)
    openings = []
    while len(openings) < games:
        s = game.new_state()
        for _ in range(opening_plies):
            s.push(rng.choice(s.legal_moves()))
            if s.winner is not None:
                break
        if s.winner is None and not s.winning_moves(s.to_move):
            openings.append(s)
    out = {}
    for opp_spec, key in (("minimax", "minimax"), ("alphabeta:depth=2", "ab2")):
        opp = make_agent(opp_spec, seed=seed)
        me = PolicyAgent(net)
        nonloss = score = 0.0
        for i, start in enumerate(openings):
            if i % 2 == 0:  # the network plays X
                rec = play_game(game, me, opp, "net", "opp", start=start)
            else:
                rec = play_game(game, opp, me, "opp", "net", start=start)
            s_ = rec.score_for("net")
            nonloss += s_ >= 0.5
            score += s_
        out[f"nonloss_vs_{key}"] = nonloss / len(openings)
        out[f"score_vs_{key}"] = score / len(openings)
    return out


# ------------------------------------------------------------- experiment
def train_base(config: GameConfig, out: Path, seed: int, quick: bool, log: Log) -> Path:
    from ..agents.alphazero import checkpoint_name, train

    path = out / "targeted" / "base" / checkpoint_name(config)
    if path.exists():
        return path
    log("training the weak base network by self-play")
    train(config, iterations=2 if quick else 4, games_per_iter=8 if quick else 30, simulations=16 if quick else 32,
          eval_games=4, seed=seed, out_dir=path.parent, log=log)
    return path


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and 95% CI over seeds for each (method, k, ablation setting)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r.get("target"), r["method"], r["k"], r.get("ablation"), r.get("eta"), r.get("nq"),
               r.get("embedding"))
        groups.setdefault(key, []).append(r)
    out = []
    for (target, method, k, abl, eta, nq, emb), rs in groups.items():
        row = {"target": target, "method": method, "k": k, "ablation": abl, "eta": eta, "nq": nq,
               "embedding": emb, "seeds": len(rs)}
        for metric in ("acc_target", "acc_critical", "hit_target", "nonloss_vs_minimax", "score_vs_ab2",
                       "select_seconds"):
            vals = [r[metric] for r in rs if r.get(metric) is not None and not
                    (isinstance(r[metric], float) and math.isnan(r[metric]))]
            if not vals:
                continue
            m = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[metric] = m
            row[metric + "_ci"] = 2.776 * sd / math.sqrt(len(vals)) if len(vals) == 5 else (
                1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        out.append(row)
    return out


def paired_vs_random(rows: list[dict], metric: str = "acc_target") -> list[dict]:
    """Per (target, method, k): mean paired difference to random over seeds,
    and in how many seeds the method beat random."""
    main = [r for r in rows if not r.get("ablation")]
    rand = {(r["target"], r["seed"], r["k"]): r[metric] for r in main if r["method"] == "random"}
    out: dict[tuple, list[float]] = {}
    for r in main:
        key = (r["target"], r["seed"], r["k"])
        if r["method"] in ("random", "base", "query-only") or key not in rand or r.get(metric) is None:
            continue
        out.setdefault((r["target"], r["method"], r["k"]), []).append(r[metric] - rand[key])
    return [{"target": t, "method": m, "k": k, "metric": metric, "mean_diff": float(np.mean(d)),
             "wins": int(sum(x > 0 for x in d)), "losses": int(sum(x < 0 for x in d)), "seeds": len(d)}
            for (t, m, k), d in sorted(out.items())]


def dose_response(rows: list[dict], nq: int) -> dict:
    """Does target accuracy depend on *which* method selected the data, or
    only on how many motif positions ended up in the training set?

    For every main run: x = motif positions labelled (hit rate * k + |Q|),
    y = target accuracy minus the query-only run of the same seed and motif.
    Fits y = a + b log2(x) on the oracle runs (pure motif data) and reports
    each method's mean residual against that curve: about zero means the
    method helps exactly as much as its hit rate predicts.
    """
    qonly = {(r["target"], r["seed"]): r["acc_target"] for r in rows
             if r["method"] == "query-only" and not r.get("ablation")}
    pts = []
    for r in rows:
        if r.get("ablation") or r["method"] in ("base", "query-only") or r.get("hit_target") is None:
            continue
        key = (r["target"], r["seed"])
        if key not in qonly:
            continue
        n_motif = r["hit_target"] * r["k"] + nq
        pts.append({"method": r["method"], "target": r["target"], "seed": r["seed"], "k": r["k"],
                    "n_motif": n_motif, "gain": r["acc_target"] - qonly[key]})
    orc = [p for p in pts if p["method"] == "oracle"]
    if len(orc) < 3:
        return {"points": pts, "fit": None, "residuals": {}}
    xs = np.log2([p["n_motif"] for p in orc])
    ys = np.array([p["gain"] for p in orc])
    b, a = np.polyfit(xs, ys, 1)
    res: dict[str, list[float]] = {}
    for p in pts:
        pred = a + b * math.log2(p["n_motif"])
        res.setdefault(p["method"], []).append(p["gain"] - pred)
    return {"points": pts, "fit": {"intercept": float(a), "slope_per_doubling": float(b)},
            "residuals": {m: float(np.mean(v)) for m, v in res.items()}}


def _sign_test(wins: int, losses: int) -> float:
    """Two-sided exact sign test (ties dropped)."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def significance(rows: list[dict], n_methods: int | None = None) -> list[dict]:
    """Paired differences to random over every (motif, seed, budget), with a
    95% CI, a two-sided sign test and its Bonferroni correction across the
    compared methods, for all budgets, k = 400 and k <= 50."""
    main = [r for r in rows if not r.get("ablation")]
    rand = {(r["target"], r["seed"], r["k"]): r for r in main if r["method"] == "random"}
    methods = [m for m in METHODS if m != "random" and any(r["method"] == m for r in main)]
    n_methods = n_methods or len(methods)
    budgets = sorted({r["k"] for r in main if r["k"]})
    out = []
    for metric in ("acc_target", "acc_critical"):
        for label, keep in (("all", lambda k: True), (f"k={budgets[-1]}", lambda k: k == budgets[-1]),
                            ("k<=50", lambda k: k <= 50)):
            for m in methods:
                diffs = [r[metric] - rand[(r["target"], r["seed"], r["k"])][metric] for r in main
                         if r["method"] == m and keep(r["k"]) and (r["target"], r["seed"], r["k"]) in rand]
                if len(diffs) < 2:
                    continue
                wins, losses = sum(x > 0 for x in diffs), sum(x < 0 for x in diffs)
                mean = float(np.mean(diffs))
                ci = float(1.96 * np.std(diffs, ddof=1) / math.sqrt(len(diffs)))
                p = _sign_test(wins, losses)
                out.append({"metric": metric, "budgets": label, "method": m, "n": len(diffs), "wins": wins,
                            "losses": losses, "mean_diff": mean, "ci95": ci, "p_sign": p,
                            "p_bonferroni": min(1.0, p * n_methods)})
    return out


def pooled_over_targets(rows: list[dict]) -> list[dict]:
    """Average each (method, k) over targets and seeds: the headline numbers."""
    main = [r for r in rows if not r.get("ablation")]
    groups: dict[tuple, list[dict]] = {}
    for r in main:
        groups.setdefault((r["method"], r["k"]), []).append(r)
    out = []
    for (method, k), rs in sorted(groups.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row = {"method": method, "k": k, "runs": len(rs)}
        for metric in ("acc_target", "acc_critical", "hit_target"):
            vals = [r[metric] for r in rs if r.get(metric) is not None]
            if vals:
                row[metric] = float(np.mean(vals))
                row[metric + "_se"] = float(np.std(vals, ddof=1) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        out.append(row)
    return out


def exp_targeted(out: Path, quick: bool = False, seed: int = 0, log: Log = print, jobs: int | None = None) -> dict:
    from ..agents.alphazero import require_torch

    require_torch()
    config = BOARD
    base_path = train_base(config, out, seed, quick, log)
    seeds = [seed] if quick else [seed + i for i in range(5)]
    args = {
        "config": config.to_dict(),
        "base_checkpoint": str(base_path),
        "sizes": (300, 200, 200) if quick else (3000, 1000, 2000),
        "budgets": [10, 20] if quick else [25, 50, 100, 200, 400],
        "methods": ["random", "entropy", "FL", "FLMI", "LogDetMI", "oracle"] if quick else METHODS,
        "targets": ["poison"] if quick else TARGETS,
        "nq": 5 if quick else 20,
        "steps": 20 if quick else 300,
        "ablation_budget": 20 if quick else 100,
        "ablation_targets": [] if quick else ["poison"],
        "etas": [] if quick else [0.25, 0.5, 1.0, 2.0, 4.0],
        "query_sizes": [] if quick else [5, 10, 20, 50, 100],
        "embedding_ablation": not quick,
        "games": 0 if quick else 20,
        "games_budget": 200,
    }
    work = [{**args, "seed": s, "verbose": False} for s in seeds]
    log(f"targeted selection on {config.label}: {len(seeds)} seeds, methods {', '.join(args['methods'])}")
    jobs = jobs or min(len(work), 4)
    if jobs > 1 and len(work) > 1:
        with mp.get_context("spawn").Pool(jobs) as p:
            per_seed = p.map(run_seed, work)
    else:
        per_seed = [run_seed(w) for w in work]
    rows = [r for s in per_seed for r in s["rows"]]
    data = {
        "meta": metadata(seed=seed, quick=quick, board=config.label, **{k: v for k, v in args.items()
                                                                       if k not in ("config",)}),
        "stats": [s["stats"] for s in per_seed],
        "rows": rows,
        "summary": summarize(rows),
        "pooled": pooled_over_targets(rows),
        "paired_target": paired_vs_random(rows, "acc_target"),
        "paired_critical": paired_vs_random(rows, "acc_critical"),
        "paired_hit": paired_vs_random(rows, "hit_target"),
        "dose_response": dose_response(rows, args["nq"]),
        "significance": significance(rows),
    }
    save_json(data, out / "targeted" / "targeted.json")
    save_csv(rows, out / "targeted" / "targeted_runs.csv")
    save_csv(data["summary"], out / "targeted" / "targeted_summary.csv")
    save_csv(data["pooled"], out / "targeted" / "targeted_pooled.csv")
    save_csv(data["significance"], out / "targeted" / "targeted_significance.csv")
    save_csv(data["paired_target"] + data["paired_critical"] + data["paired_hit"],
             out / "targeted" / "targeted_paired.csv")
    replot_targeted(out)
    return data


def replot_targeted(out: Path) -> None:
    from ..plots import plot_targeted

    data = load_json(out / "targeted" / "targeted.json")
    plot_targeted(data, out / "targeted")

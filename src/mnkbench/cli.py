"""Command-line interface: ``mnk-bench <command> ...``.

Agent specs are ``name`` or ``name:key=value,key=value``, for example
``alphabeta:depth=4`` or ``mcts:iterations=2000,c=1.0``. Boards are
``m,n,k`` with optional ``+gravity`` / ``+exact`` flags, e.g. ``6,7,4+gravity``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agents import DESCRIPTIONS, REGISTRY, make_agent
from .agents.base import MB, AgentRefused, Budget, fmt_bytes, fmt_seconds
from .game import GameConfig, get_game


def _board(text: str) -> GameConfig:
    try:
        return GameConfig.parse(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _budget(args) -> Budget:
    return Budget(time_per_move=args.time, memory_bytes=args.memory * MB, train_time=args.train_time,
                  cost_per_move=args.cost)


def _add_budget(p: argparse.ArgumentParser) -> None:
    p.add_argument("--time", type=float, default=1.0, help="seconds per move (default 1.0)")
    p.add_argument("--memory", type=float, default=1024, help="memory budget in MB (default 1024)")
    p.add_argument("--train-time", type=float, default=600.0, help="training budget in seconds")
    p.add_argument("--cost", type=float, default=0.05, help="USD per move (LLM agents)")


# ------------------------------------------------------------------ commands
def cmd_agents(args) -> int:
    width = max(len(n) for n in REGISTRY)
    for name in REGISTRY:
        print(f"{name:<{width}}  {DESCRIPTIONS.get(name, '')}")
    return 0


def _prepare(spec: str, cfg: GameConfig, seed: int, args) -> object:
    agent = make_agent(spec, seed=seed)
    if getattr(args, "train", 0) and agent.learns:
        from .experiments.learning import train

        print(f"training {spec} for {args.train} games...", file=sys.stderr)
        train(agent, cfg, args.train, opponent=args.train_opponent)
        agent.training = False
    if getattr(args, "budget", False):
        agent = agent.adapt(cfg, _budget(args))
    return agent


def cmd_play(args) -> int:
    from .tournament import play_game

    cfg = args.board
    game = get_game(cfg)
    x = _prepare(args.x, cfg, args.seed, args)
    o = _prepare(args.o, cfg, args.seed + 1, args)

    def show(state, move):
        if args.verbose:
            print(f"{'XO'[state.to_move - 1]} plays {game.cell_name(move)}")

    rec = play_game(cfg, x, o, f"X:{args.x}", f"O:{args.o}", on_move=show)
    final = game.new_state()
    for mv in rec.moves:
        final.push(mv)
    print(final.render())
    if rec.aborted:
        print(f"aborted by {rec.aborted_by}: {rec.aborted}")
        return 1
    result = "draw" if rec.winner == 0 else f"{'XO'[rec.winner - 1]} wins"
    print(f"{result} after {rec.length} moves: " + " ".join(game.cell_name(m) for m in rec.moves))
    return 0


def cmd_match(args) -> int:
    from .results import save_json
    from .tournament import play_match

    cfg = args.board
    a = _prepare(args.a, cfg, args.seed, args)
    b = _prepare(args.b, cfg, args.seed + 1, args)
    r = play_match(cfg, a, b, args.games, seed=args.seed, a_name=args.a, b_name=args.b,
                   keep_games=bool(args.json))
    print(f"{cfg.label}: {r.a} vs {r.b} over {r.n_games} games")
    print(f"  {r.a}: {r.wins_a} wins, {r.draws} draws, {r.wins_b} losses  (score {r.score_a:.3f})")
    print(f"  as X: {r.a_as_x}   as O: {r.a_as_o}")
    for name, t in r.time_per_move.items():
        print(f"  {name}: {fmt_seconds(t)} per move (max {fmt_seconds(r.max_time_per_move[name])})")
    if r.aborted:
        print(f"  aborted: {r.abort_reasons}")
    if args.json:
        save_json(r.to_dict(include_games=True), args.json)
    return 0


def cmd_tournament(args) -> int:
    from .elo import bootstrap_elo, fit_elo
    from .experiments.suite import wdl_markdown
    from .results import save_json
    from .tournament import round_robin

    cfg = args.board
    factories = {s: (lambda s=s, i=i: _prepare(s, cfg, args.seed + i, args)) for i, s in enumerate(args.agents)}
    t = round_robin(cfg, factories, args.games, seed=args.seed,
                    progress=(lambda m: print(m, file=sys.stderr)) if args.verbose else None)
    games = t.pairwise_games()
    elo = fit_elo(games, anchor=args.agents[0], anchor_rating=1000.0) if games else {}
    ci = bootstrap_elo(games, samples=100, anchor=args.agents[0], anchor_rating=1000.0) if games else {}
    print(wdl_markdown(t.to_dict()))
    print(f"Elo ({args.agents[0]} = 1000):")
    for name, r in sorted(elo.items(), key=lambda kv: -kv[1]):
        lo, hi = ci.get(name, (r, r))
        print(f"  {name:<28} {r:7.0f}   95% CI [{lo:.0f}, {hi:.0f}]")
    if args.out:
        d = t.to_dict()
        d["elo"] = {"ratings": elo, "ci95": {k: list(v) for k, v in ci.items()}}
        save_json(d, args.out)
        if args.plot:
            from .plots import plot_elo, plot_score_matrix

            out = Path(args.out)
            plot_score_matrix(d, out.with_suffix(".matrix.png"))
            plot_elo(elo, out.with_suffix(".elo.png"), intervals=ci)
    return 0


def cmd_feasibility(args) -> int:
    from .feasibility import DEFAULT_BOARDS, feasibility_map
    from .results import save_json

    boards = args.boards or DEFAULT_BOARDS
    specs = args.agents or ["random", "rule", "minimax", "menace", "qlearning", "td", "alphabeta:depth=4",
                            "mcts:iterations=1000"]
    budget = _budget(args)
    data = feasibility_map(specs, boards, budget, measure=args.measure, verify_frontier=args.measure)
    width = max(len(s) for s in specs)
    print(f"budget: {fmt_seconds(budget.time_per_move)}/move, {fmt_bytes(budget.memory_bytes)}, "
          f"training {fmt_seconds(budget.train_time)}")
    header = " " * (width + 2) + " ".join(f"{b:>15}" for b in data["boards"])
    print(header)
    for spec in specs:
        cells = {c["config"]: c for c in data["cells"] if c["agent"] == spec}
        row = " ".join(f"{cells[b]['status'] + ('*' if cells[b]['measured'] else ''):>15}" for b in data["boards"])
        print(f"{spec:<{width}}  {row}")
    if args.explain:
        for c in data["cells"]:
            if c["status"] != "ok":
                print(f"{c['agent']} on {c['config']}: {c['status']} - {c['reason']} {c['estimate']['notes']}")
    if args.out:
        save_json(data, args.out)
        from .plots import plot_feasibility_map

        plot_feasibility_map(data, Path(args.out).with_suffix(".png"))
    return 0


def cmd_solve(args) -> int:
    from .agents.minimax import Solver
    from .game import State

    cfg = args.board
    agent = make_agent("minimax")
    try:
        agent.adapt(cfg, _budget(args))
    except AgentRefused as exc:
        print(exc)
        if not args.force:
            return 2
    state = State.from_string(cfg, args.position) if args.position else get_game(cfg).new_state()
    solver = Solver(cfg)
    outcome = solver.outcome(state)
    word = {1: "win", 0: "draw", -1: "loss"}[outcome]
    print(state.render())
    print(f"{'XO'[state.to_move - 1]} to move: {word} with perfect play ({solver.nodes} nodes)")
    if state.winner is None:
        game = get_game(cfg)
        best = solver.outcome_preserving_moves(state)
        print("outcome-preserving moves:", " ".join(game.cell_name(m) for m in best))
    return 0


def cmd_experiment(args) -> int:
    from .experiments.suite import EXPERIMENTS, run

    names = args.names or list(EXPERIMENTS)
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        print(f"unknown experiment(s) {unknown}; choose from {', '.join(EXPERIMENTS)}")
        return 2
    run(names, args.out, quick=args.quick, seed=args.seed, log=print if args.verbose else (lambda s: None))
    if not args.no_report:
        from .report import build_report

        print("report:", build_report(args.out))
    return 0


def cmd_replot(args) -> int:
    from .experiments.suite import replot

    done = replot(args.out, args.names or None)
    print("replotted:", ", ".join(done) if done else "nothing found")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    print(build_report(args.out, args.html))
    return 0


def cmd_bench(args) -> int:
    from .perf import PERF_BOARDS, run_benchmarks

    run_benchmarks(args.boards or PERF_BOARDS, quick=args.quick, log=print)
    return 0


def cmd_train_alphazero(args) -> int:
    from .agents.alphazero import train_main

    return train_main(args)


# -------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mnk-bench", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("agents", help="list available agents")
    s.set_defaults(fn=cmd_agents)

    def common(sp, games: int | None = None):
        sp.add_argument("--board", type=_board, default=GameConfig(3, 3, 3), help="m,n,k[+gravity][+exact]")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--train", type=int, default=0, help="pre-train learning agents for N games")
        sp.add_argument("--train-opponent", default="random+rule", help="opponent spec(s) for --train")
        sp.add_argument("--budget", action="store_true", help="adapt agents to the budget (refuse/degrade)")
        _add_budget(sp)
        if games is not None:
            sp.add_argument("--games", type=int, default=games)

    s = sub.add_parser("play", help="play one game and show it (use 'human' to play yourself)")
    s.add_argument("x", help="agent spec for X")
    s.add_argument("o", help="agent spec for O")
    s.add_argument("-v", "--verbose", action="store_true")
    common(s)
    s.set_defaults(fn=cmd_play)

    s = sub.add_parser("match", help="play a series between two agents")
    s.add_argument("a")
    s.add_argument("b")
    s.add_argument("--json", help="save the match (with all games) as JSON")
    common(s, 20)
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("tournament", help="round robin with W/D/L table and Elo")
    s.add_argument("--agents", nargs="+", required=True)
    s.add_argument("--out", help="save results JSON here")
    s.add_argument("--plot", action="store_true", help="also save matrix and Elo plots next to --out")
    s.add_argument("-v", "--verbose", action="store_true")
    common(s, 20)
    s.set_defaults(fn=cmd_tournament)

    s = sub.add_parser("feasibility", help="which agents can play which boards within a budget")
    s.add_argument("--agents", nargs="+")
    s.add_argument("--boards", nargs="+", type=_board)
    s.add_argument("--measure", action="store_true", help="confirm estimates with guarded probe runs")
    s.add_argument("--explain", action="store_true", help="print reasons for non-ok cells")
    s.add_argument("--out", help="save the map as JSON (and a PNG next to it)")
    _add_budget(s)
    s.set_defaults(fn=cmd_feasibility)

    s = sub.add_parser("solve", help="solve a position with perfect play")
    s.add_argument("--board", type=_board, default=GameConfig(3, 3, 3))
    s.add_argument("--position", help='diagram such as "X.O/.X./..."')
    s.add_argument("--force", action="store_true", help="solve even if the estimate says it won't fit")
    _add_budget(s)
    s.set_defaults(fn=cmd_solve)

    s = sub.add_parser("experiment", help="run benchmark experiments (writes JSON, CSV, plots, report)")
    s.add_argument("names", nargs="*", help="experiments to run (default: all)")
    s.add_argument("--out", default="results")
    s.add_argument("--quick", action="store_true", help="tiny sizes (smoke test)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--no-report", action="store_true")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_experiment)

    s = sub.add_parser("replot", help="redraw all plots from saved JSON")
    s.add_argument("names", nargs="*")
    s.add_argument("--out", default="results")
    s.set_defaults(fn=cmd_replot)

    s = sub.add_parser("report", help="build the HTML report from saved results")
    s.add_argument("--out", default="results")
    s.add_argument("--html", help="output file (default <out>/index.html)")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("bench", help="micro-benchmarks of the engine and agents")
    s.add_argument("--boards", nargs="+", type=_board)
    s.add_argument("--quick", action="store_true")
    s.set_defaults(fn=cmd_bench)

    s = sub.add_parser("train-alphazero", help="train an AlphaZero-style agent by self-play (needs torch)")
    s.add_argument("--board", type=_board, default=GameConfig(3, 3, 3))
    s.add_argument("--iterations", type=int, default=20, help="self-play/train iterations")
    s.add_argument("--games", type=int, default=40, help="self-play games per iteration")
    s.add_argument("--simulations", type=int, default=50, help="MCTS simulations per move")
    s.add_argument("--channels", type=int, default=32)
    s.add_argument("--blocks", type=int, default=2)
    s.add_argument("--epochs", type=int, default=4)
    s.add_argument("--eval-games", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="results/alphazero")
    s.set_defaults(fn=cmd_train_alphazero)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


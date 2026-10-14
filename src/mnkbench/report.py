"""Build a static HTML report from saved results (no experiments are re-run).

The report is a single ``index.html`` next to the result folders; images are
linked relatively, so the whole directory can be browsed or published as is.
"""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

from .results import load_json

CSS = """
:root { --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --line:#e1e0d9; --accent:#2a78d6; --ok:#0ca30c; --warn:#fab219; --bad:#d03b3b; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781; --line:#2c2c2a;
  --accent:#3987e5; } }
:root[data-theme="dark"] { --bg:#0d0d0d; --surface:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7;
  --muted:#898781; --line:#2c2c2a; --accent:#3987e5; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1080px; margin: 0 auto; padding: 24px 16px 64px; }
h1 { font-size: 28px; margin: 8px 0 4px; }
h2 { font-size: 20px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
h3 { font-size: 16px; margin: 20px 0 6px; }
p, li { color: var(--ink2); max-width: 75ch; }
.sub { color: var(--muted); margin-top: 0; }
nav { display:flex; flex-wrap:wrap; gap:6px 14px; margin: 16px 0; }
nav a { color: var(--accent); text-decoration: none; font-size: 14px; }
figure { margin: 12px 0; background: var(--surface); border:1px solid var(--line); border-radius: 8px;
  padding: 8px; overflow-x: auto; }
figure img { max-width: 100%; height: auto; display: block; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap: 12px; }
.grid figure.wide { grid-column: 1 / -1; }
@media (max-width: 520px) { .grid { grid-template-columns: minmax(0, 1fr); } }
main > *, section > * { min-width: 0; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; font-size: 13px; margin: 8px 0; background: var(--surface); }
th, td { padding: 5px 10px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }
th { color: var(--ink2); font-weight: 600; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display:inline-block; padding: 0 8px; border-radius: 10px; font-size: 12px; color:#fff; }
.ok { background: var(--ok); } .degraded { background: var(--warn); color:#0b0b0b; }
.refused { background: var(--bad); } .aborted { background:#ec835a; color:#0b0b0b; }
.unavailable { background: var(--muted); }
code { font-size: 13px; }
"""


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if math.isnan(v):
            return "-"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) >= 1:
            return f"{v:.2f}"
        if v == 0:
            return "0"
        if abs(v) < 0.001:
            return f"{v:.2g}"
        return f"{v:.3f}"
    if isinstance(v, list):
        return ", ".join(_fmt(x) for x in v)
    return str(v)


def table(rows: list[dict], columns: list[tuple[str, str]] | None = None) -> str:
    if not rows:
        return "<p>No data.</p>"
    columns = columns or [(k, k) for k in rows[0]]
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    body = []
    for r in rows:
        cells = []
        for key, _ in columns:
            v = r.get(key)
            cls = ' class="num"' if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
            if key == "status" and isinstance(v, str):
                cells.append(f'<td><span class="pill {html.escape(v)}">{html.escape(v)}</span></td>')
            else:
                cells.append(f"<td{cls}>{html.escape(_fmt(v))}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _png_size(path: Path) -> tuple[int, int] | None:
    head = path.read_bytes()[:24]
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    return int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")


def figure(out: Path, rel: str, alt: str) -> str:
    path = out / rel
    if not path.exists():
        return ""
    size = _png_size(path)
    wide = ' class="wide"' if size and size[0] > 2.2 * size[1] else ""
    return f'<figure{wide}><img src="{html.escape(rel)}" alt="{html.escape(alt)}" loading="lazy"></figure>'


def figures(out: Path, items: list[tuple[str, str]]) -> str:
    figs = [figure(out, rel, alt) for rel, alt in items]
    figs = [f for f in figs if f]
    return f'<div class="grid">{"".join(figs)}</div>' if figs else ""


# ----------------------------------------------------------------- sections
def _wdl_rows(wdl: list[dict]) -> list[dict]:
    rows = []
    for r in wdl:
        n = r["win"] + r["draw"] + r["loss"]
        rows.append({**r, "games": n, "score": (r["win"] + 0.5 * r["draw"]) / n if n else None})
    return sorted(rows, key=lambda r: -(r["score"] or 0))


WDL_COLS = [("agent", "agent"), ("games", "games"), ("win", "win"), ("draw", "draw"), ("loss", "loss"),
            ("score", "score")]


def sec_menace(out: Path) -> str:
    d = load_json(out / "menace" / "menace.json")
    rows = []
    for run in d["runs"]:
        last = run["checkpoints"][-1]
        rows.append({"trained vs": run["opponent"], "games": last["games"],
                     "games to learn": run["games_to_learn"], "matchboxes": last["table_size"],
                     "win vs random": last["vs_random"]["win"],
                     "non-loss vs perfect": 1 - last["vs_minimax"]["loss"],
                     "optimal-move rate": last.get("optimal_rate")})
    return (
        "<p>MENACE learns tic-tac-toe from scratch by bead reinforcement. Who it practises against matters: "
        "against random play it never learns to survive perfect play, while games against a perfect "
        "player teach it to draw. <em>Games to learn</em> is the first checkpoint from which it loses "
        "at most 5% of games against perfect play.</p>"
        + table(rows)
        + figures(out, [("menace/menace_learning.png", "MENACE learning curves"),
                        ("menace/menace_boxes.png", "MENACE matchboxes in use")])
    )


def sec_symmetry(out: Path) -> str:
    d = load_json(out / "symmetry" / "symmetry.json")
    return (
        "<p>Storing positions up to rotation and reflection is what got Michie down to about 300 "
        "matchboxes. The counts below are exact enumerations of every position MENACE could face.</p>"
        + table(d["matchbox_counts"])
        + figures(out, [("symmetry/symmetry_learning.png", "learning with and without symmetry"),
                        ("symmetry/symmetry_boxes.png", "matchboxes with and without symmetry")])
    )


def sec_tabular(out: Path) -> str:
    d = load_json(out / "tabular" / "tabular.json")
    cols = [("agent", "agent"), ("opponent", "trained vs"), ("games_to_learn", "games to learn (per seed)"),
            ("final_nonloss_vs_minimax", "non-loss vs perfect"), ("final_win_vs_random", "win vs random"),
            ("final_optimal_rate", "optimal-move rate"), ("table_size", "table entries"),
            ("train_time", "train time (s)")]
    return (
        "<p>MENACE, Q-learning and TD(0) all use one table entry per position. Averaged over seeds, after "
        "the same number of training games:</p>"
        + table(d["summary"], cols)
        + figures(out, [("tabular/tabular_mixed.png", "tabular learners, mixed opponents"),
                        ("tabular/tabular_selfplay.png", "tabular learners, self-play"),
                        ("tabular/tabular_tables.png", "table growth")])
    )


def sec_h2h(out: Path) -> str:
    d = load_json(out / "h2h" / "h2h.json")
    return ("<p>Every agent against every other on tic-tac-toe, alternating who moves first.</p>"
            + table(_wdl_rows(d["wdl"]), WDL_COLS)
            + figures(out, [("h2h/h2h_matrix.png", "head-to-head matrix")]))


def sec_bigboards(out: Path) -> str:
    d = load_json(out / "bigboards" / "bigboards.json")
    cols = [("config", "board"), ("agent", "agent"), ("table_size", "table entries"), ("coverage", "coverage"),
            ("score_vs_random", "score vs random"), ("score_vs_rule", "score vs rule")]
    rows = [r for r in d["scaling"] if r["agent"] in ("menace", "qlearning", "td", "alphabeta:depth=4")]
    return (
        f"<p>Tabular learners get the same {d['meta']['train_games']} training games on every board. "
        "As the board grows, the share of positions they have seen before (<em>coverage</em>) collapses, "
        "and so does their play; depth-limited alpha-beta has no table and keeps getting relatively "
        "stronger.</p>"
        + table(rows, cols)
        + figures(out, [("bigboards/score_vs_rule.png", "score vs rule"),
                        ("bigboards/table_size.png", "table size"),
                        ("bigboards/coverage.png", "coverage"),
                        ("bigboards/tournament_matrix.png", "tournament")])
        + f"<h3>Tournament on {d['tournament']['config']}</h3>"
        + table(_wdl_rows(d["tournament"]["wdl"]), WDL_COLS)
    )


def sec_mcts(out: Path) -> str:
    d = load_json(out / "mcts" / "mcts.json")
    return ("<p>MCTS needs no evaluation function, only playouts. More playouts per move help, but on "
            "boards where random games are uninformative (6x6, k=4) it stays weak.</p>"
            + table(d["rows"], [("iterations", "playouts"), ("config", "board"), ("metric", "metric"),
                                ("value", "value"), ("time_per_move", "s/move")])
            + figures(out, [("mcts/mcts_budget.png", "MCTS strength vs playouts")]))


def sec_elo(out: Path) -> str:
    d = load_json(out / "elo" / "elo.json")
    parts = ["<p>Ratings fitted to every game of a round robin (Bradley-Terry, random play anchored at "
             "1000, 95% bootstrap intervals).</p>"]
    from .experiments.sampling import _slug

    for b in d["boards"]:
        rows = [{"agent": a, "elo": r, "95% CI": f"{b['elo']['ci95'][a][0]:.0f} to {b['elo']['ci95'][a][1]:.0f}"}
                for a, r in sorted(b["elo"]["ratings"].items(), key=lambda kv: -kv[1])]
        slug = _slug(b["config"])
        parts.append(f"<h3>{html.escape(b['config'])}</h3>" + table(rows)
                     + figures(out, [(f"elo/elo_{slug}.png", f"Elo on {b['config']}"),
                                     (f"elo/matrix_{slug}.png", f"matrix on {b['config']}")]))
    return "".join(parts)


def sec_feasibility(out: Path) -> str:
    d = load_json(out / "feasibility" / "feasibility.json")
    parts = ["<p>Each agent estimates its time per move, memory and training time for a board and "
             "compares them with the budget: <em>ok</em>, <em>degraded</em> (runs in a cheaper "
             "configuration, e.g. shallower search, fewer playouts, or under-trained), or "
             "<em>refused</em>. Cells marked * were confirmed by running the agent in a guarded "
             "subprocess that kills it if it overruns (<em>aborted</em>).</p>"]
    for name, m in d["maps"].items():
        b = m["budget"]
        parts.append(f"<h3>{html.escape(name)} budget: {b['time_per_move']}s/move, "
                     f"{b['memory_bytes'] / 2**20:.0f} MB, {b['train_time']:.0f}s training</h3>")
        parts.append(figures(out, [(f"feasibility/feasibility_{name}.png", f"feasibility map {name}")]))
        parts.append(table(m.get("frontier", []), [("agent", "agent"), ("largest_ok", "largest ok board"),
                                                   ("largest_usable", "largest usable board"),
                                                   ("first_refused", "first refused")]))
    return "".join(parts)


def sec_perf(out: Path) -> str:
    d = load_json(out / "perf" / "perf.json")
    return ("<p>Throughput of the hot paths and time per move on growing boards. The fast playout "
            "(bare cell array, inlined win check, lazy shuffle) is what MCTS uses.</p>"
            + table(d["playouts"], [("config", "board"), ("reference_per_s", "reference playouts/s"),
                                    ("fast_per_s", "fast playouts/s"), ("speedup", "speedup")])
            + figures(out, [("perf/time_per_move.png", "time per move"), ("perf/playouts.png", "playouts")]))


def sec_gomoku(out: Path) -> str:
    d = load_json(out / "gomoku" / "gomoku.json")
    rows = [{"agent": a, "elo": r} for a, r in sorted(d["elo"]["ratings"].items(), key=lambda kv: -kv[1])]
    t = d["tournament"]
    times = {}
    for m in t["matches"]:
        for k, v in m["time_per_move"].items():
            times.setdefault(k, []).append(v)
    for r in rows:
        ts = times.get(r["agent"], [])
        r["s/move"] = sum(ts) / len(ts) if ts else None
    return (f"<p>The full gomoku board {html.escape(t['config'])}: only heuristic search and sampling "
            "remain feasible.</p>" + table(rows)
            + table(_wdl_rows(t["wdl"]), WDL_COLS)
            + figures(out, [("gomoku/gomoku_elo.png", "gomoku Elo"), ("gomoku/gomoku_matrix.png", "gomoku matrix")]))


def sec_alphazero(out: Path) -> str:
    d = load_json(out / "alphazero" / "alphazero.json")
    parts = ["<p>A small residual policy/value network guides MCTS (PUCT) and is trained only from "
             "self-play games. The table shows each training run's final evaluation.</p>"]
    rows = []
    for run in d["runs"]:
        last = run["history"][-1]
        rows.append({"board": run["config"], "iterations": len(run["history"]),
                     "self-play games": last["games"], "params": run["params"],
                     "train time (s)": last["elapsed"],
                     "score vs random": last.get("score_vs_random"),
                     "score vs rule": last.get("score_vs_rule"),
                     "non-loss vs perfect": last.get("nonloss_vs_minimax"),
                     "optimal-move rate": last.get("optimal_rate")})
    parts.append(table(rows))
    imgs = [(f"alphazero/{p.name}", p.stem) for p in sorted((out / "alphazero").glob("*.png"))]
    parts.append(figures(out, imgs))
    return "".join(parts)


def sec_llm(out: Path) -> str:
    d = load_json(out / "llm" / "llm.json")
    parts = [f"<p>{html.escape(d.get('note', ''))}</p>"]
    parts.append(table(d.get("rows", [])))
    imgs = [(f"llm/{p.name}", p.stem) for p in sorted((out / "llm").glob("*.png"))]
    parts.append(figures(out, imgs))
    return "".join(parts)


SECTIONS = [
    ("menace", "MENACE learns tic-tac-toe", sec_menace),
    ("symmetry", "Board symmetries", sec_symmetry),
    ("tabular", "MENACE vs Q-learning vs TD", sec_tabular),
    ("h2h", "Head to head on tic-tac-toe", sec_h2h),
    ("bigboards", "Bigger boards: where tables choke", sec_bigboards),
    ("mcts", "Monte Carlo Tree Search", sec_mcts),
    ("elo", "Elo ratings", sec_elo),
    ("feasibility", "Feasibility map", sec_feasibility),
    ("perf", "Performance", sec_perf),
    ("gomoku", "Gomoku (15,15,5)", sec_gomoku),
    ("alphazero", "AlphaZero-style agent", sec_alphazero),
    ("llm", "LLM players", sec_llm),
]


def build_report(out: str | Path, html_path: str | Path | None = None) -> Path:
    out = Path(out)
    html_path = Path(html_path) if html_path else out / "index.html"
    nav, body = [], []
    for key, title, fn in SECTIONS:
        if not (out / key / f"{key}.json").exists():
            continue
        try:
            content = fn(out)
        except (KeyError, TypeError, ValueError) as exc:  # stale or partial data
            content = f"<p>Could not render this section: {html.escape(str(exc))}</p>"
        nav.append(f'<a href="#{key}">{html.escape(title)}</a>')
        body.append(f'<section id="{key}"><h2>{html.escape(title)}</h2>{content}</section>')
    meta = ""
    for key, *_ in SECTIONS:
        p = out / key / f"{key}.json"
        if p.exists():
            m = load_json(p).get("meta", {})
            meta = f"mnk-bench {m.get('mnkbench_version', '')}, Python {m.get('python', '')}, generated {m.get('created', '')}"
            break
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mnk-bench results</title><style>{CSS}</style></head>
<body><main>
<h1>mnk-bench results</h1>
<p class="sub">Where does each game-playing AI fall apart as the board gets bigger? {html.escape(meta)}</p>
<nav>{''.join(nav)}</nav>
{''.join(body) if body else '<p>No results found. Run <code>mnk-bench experiment</code> first.</p>'}
</main></body></html>
"""
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(doc)
    return html_path

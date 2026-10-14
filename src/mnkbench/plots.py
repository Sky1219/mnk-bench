"""Plots, drawn from saved result data only (never by re-running games).

Colour follows the agent family, so the same agent has the same colour in
every figure. The palette is a validated colour-blind-safe categorical order;
random play is drawn in neutral grey as a baseline.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY_COLORS = {
    "minimax": SERIES[0],
    "menace": SERIES[1],
    "qlearning": SERIES[2],
    "td": SERIES[3],
    "mcts": SERIES[4],
    "alphabeta": SERIES[5],
    "alphazero": SERIES[6],
    "rule": SERIES[7],
    "random": MUTED,
    "llm": INK_2,
}
MARKERS = {"minimax": "o", "menace": "s", "qlearning": "D", "td": "^", "mcts": "v",
           "alphabeta": "P", "alphazero": "X", "rule": "h", "random": ".", "llm": "*"}

STATUS_COLORS = {
    "ok": "#0ca30c",
    "degraded": "#fab219",
    "aborted": "#ec835a",
    "refused": "#d03b3b",
    "unavailable": "#c3c2b7",
}

# Diverging blue <-> red through a neutral grey (for score matrices centred at 0.5).
DIVERGING = LinearSegmentedColormap.from_list(
    "mnk_div", ["#e34948", "#f3b2a8", "#f0efec", "#9ec5f4", "#1c5cab"]
)
SEQUENTIAL = LinearSegmentedColormap.from_list("mnk_seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def family(name: str) -> str:
    base = name.split(":")[0].split("#")[0].split("[")[0].strip().lower()
    for fam in FAMILY_COLORS:
        if base.startswith(fam):
            return fam
    return base


def color_for(name: str, fallback_index: int = 0) -> str:
    return FAMILY_COLORS.get(family(name), SERIES[fallback_index % len(SERIES)])


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "figure.dpi": 110,
        }
    )


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _get(d: dict, dotted: str) -> float:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur or cur[part] is None:
            return math.nan
        cur = cur[part]
    return float(cur)


def _run_label(run: dict) -> str:
    opp = run.get("opponent")
    return f"{run['agent']} (vs {opp})" if opp else run["agent"]


# --------------------------------------------------------------- learning
LEARNING_PANELS = (
    ("vs_random.win", "Win rate vs random"),
    ("vs_minimax.nonloss", "Non-loss rate vs perfect play"),
    ("optimal_rate", "Optimal-move rate"),
)


def plot_learning_curves(runs: Sequence[dict], path: str | Path, title: str | None = None) -> Path:
    """One panel per metric; one line per training run (agent and opponent)."""
    style()
    runs = list(runs)[:8]
    fig, axes = plt.subplots(1, len(LEARNING_PANELS), figsize=(4.2 * len(LEARNING_PANELS), 3.6), sharex=True)
    used_colors: dict[str, int] = {}
    # If every run is the same agent (e.g. MENACE vs different opponents),
    # colour by run in fixed palette order; otherwise colour by agent family.
    one_family = len({family(r["agent"]) for r in runs}) == 1
    for idx, run in enumerate(runs):
        label = _run_label(run)
        base = SERIES[idx] if one_family else color_for(run["agent"], idx)
        # Same agent trained against different opponents: vary the line style.
        n_same = used_colors.get(base, 0)
        used_colors[base] = n_same + 1
        linestyle = ["-", "--", ":", "-."][n_same % 4]
        xs = [cp["games"] for cp in run["checkpoints"]]
        for ax, (metric, _) in zip(axes, LEARNING_PANELS):
            if metric == "vs_minimax.nonloss":
                ys = [1 - _get(cp, "vs_minimax.loss") for cp in run["checkpoints"]]
            else:
                ys = [_get(cp, metric) for cp in run["checkpoints"]]
            ax.plot(xs, ys, color=base, linestyle=linestyle, marker=MARKERS.get(family(run["agent"]), "o"),
                    markersize=4, label=label)
    for ax, (_, name) in zip(axes, LEARNING_PANELS):
        ax.set_title(name, loc="left")
        ax.set_xlabel("training games")
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("rate")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.08))
    label = runs[0]["label"] if runs else ""
    fig.suptitle(title or f"Learning curves on {label}", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


def plot_table_growth(runs: Sequence[dict], path: str | Path, title: str | None = None) -> Path:
    """Number of table entries (matchboxes / Q states / values) vs games."""
    style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    runs = list(runs)[:8]
    one_family = len({family(r["agent"]) for r in runs}) == 1
    for idx, run in enumerate(runs):
        xs = [cp["games"] for cp in run["checkpoints"]]
        ys = [cp.get("table_size", 0) for cp in run["checkpoints"]]
        ax.plot(xs, ys, color=SERIES[idx] if one_family else color_for(run["agent"], idx), marker=MARKERS.get(family(run["agent"]), "o"),
                markersize=4, label=_run_label(run))
        ax.annotate(f"{ys[-1]}", (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center")
    ax.set_xlabel("training games")
    ax.set_ylabel("table entries")
    ax.set_title(title or "Table size during training", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.08, 0.5))
    fig.tight_layout()
    return _save(fig, path)


# ----------------------------------------------------------- head to head
def plot_score_matrix(tournament: dict, path: str | Path, title: str | None = None) -> Path:
    """Heatmap of row agent's score against column agent (0.5 = even)."""
    style()
    agents = tournament["agents"]
    matrix = tournament["score_matrix"]
    n = len(agents)
    fig, ax = plt.subplots(figsize=(1.0 + 0.75 * n, 0.8 + 0.65 * n))
    data = [[matrix.get(a, {}).get(b) for b in agents] for a in agents]
    vals = [[math.nan if v is None else v for v in row] for row in data]
    ax.imshow(vals, cmap=DIVERGING, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), agents, rotation=40, ha="right")
    ax.set_yticks(range(n), agents)
    for i in range(n):
        for j in range(n):
            v = vals[i][j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v - 0.5) > 0.35 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title or f"Score of row vs column on {tournament['config']}", loc="left")
    fig.tight_layout()
    return _save(fig, path)


def plot_elo(ratings: dict[str, float], path: str | Path, title: str | None = None,
             intervals: dict[str, tuple[float, float]] | None = None) -> Path:
    """Horizontal bars of Elo ratings, strongest on top."""
    style()
    items = sorted(ratings.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, 0.5 + 0.38 * len(items)))
    lo = min(v for _, v in items) - 100
    for i, (name, v) in enumerate(items):
        ax.barh(i, v - lo, left=lo, color=color_for(name, i), height=0.6)
        if intervals and name in intervals:
            a, b = intervals[name]
            ax.plot([a, b], [i, i], color=INK_2, linewidth=1)
        ax.text(v + 8, i, f"{v:.0f}", va="center", fontsize=8, color=INK_2)
    ax.set_yticks(range(len(items)), [k for k, _ in items])
    ax.set_xlabel("Elo")
    ax.grid(axis="y", visible=False)
    ax.set_title(title or "Elo ratings", loc="left")
    fig.tight_layout()
    return _save(fig, path)


# --------------------------------------------------------------- scaling
def plot_scaling(rows: Sequence[dict], x: str, y: str, path: str | Path, title: str,
                 xlabel: str, ylabel: str, logy: bool = True, group: str = "agent",
                 logx: bool = False) -> Path:
    """Line per agent of ``y`` against ``x`` (e.g. time per move vs board size).

    If ``x`` values are strings (board labels) they are placed in order of
    first appearance. Agents of the same family share a colour and differ
    by line style.
    """
    style()
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[group], []).append(r)
    categorical = any(isinstance(r.get(x), str) for r in rows)
    cats: list[str] = []
    if categorical:
        for r in rows:
            if r.get(x) not in cats:
                cats.append(r[x])
    family_seen: dict[str, int] = {}
    shown = list(groups.items())[:9]
    for idx, (name, rs) in enumerate(shown):
        pts = [((cats.index(r[x]) if categorical else r[x]), r[y]) for r in rs
               if r.get(y) is not None and r.get(x) is not None]
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        fam = family(name)
        k = family_seen.get(fam, 0)
        family_seen[fam] = k + 1
        ax.plot(xs, ys, color=color_for(name, idx), marker=MARKERS.get(fam, "o"),
                linestyle=["-", "--", ":", "-."][k % 4], label=name)
        if len(shown) <= 4:
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=7,
                        color=INK_2, va="center")
    if categorical:
        ax.set_xticks(range(len(cats)), cats, rotation=30, ha="right")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return _save(fig, path)


def plot_feasibility_map(data: dict, path: str | Path, title: str | None = None) -> Path:
    """Grid of agent x board coloured by feasibility status, with a text label
    in every cell so status never relies on colour alone."""
    style()
    agents = data["agents"]
    boards = data["boards"]
    cells = {(c["agent"], c["config"]): c for c in data["cells"]}
    order = list(STATUS_COLORS)
    cmap = ListedColormap([STATUS_COLORS[s] for s in order])
    grid = [[order.index(cells[(a, b)]["status"]) if (a, b) in cells else math.nan for b in boards]
            for a in agents]
    fig, ax = plt.subplots(figsize=(max(6.5, 1.2 + 0.78 * len(boards)), 0.9 + 0.48 * len(agents)))
    ax.imshow(grid, cmap=cmap, vmin=-0.5, vmax=len(order) - 0.5, aspect="auto")
    ax.grid(False)
    short = {"ok": "ok", "degraded": "degr", "aborted": "abort", "refused": "refuse", "unavailable": "n/a"}
    for i, a in enumerate(agents):
        for j, b in enumerate(boards):
            c = cells.get((a, b))
            if c:
                txt = short[c["status"]]
                if c.get("measured"):
                    txt += "*"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color=INK if c["status"] in ("degraded", "unavailable", "aborted") else "white")
    ax.set_xticks(range(len(boards)), boards, rotation=45, ha="right")
    ax.set_yticks(range(len(agents)), agents)
    # 2px surface-coloured gaps between cells.
    ax.set_xticks([x - 0.5 for x in range(1, len(boards))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(agents))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    from matplotlib.patches import Patch

    handles = [Patch(color=STATUS_COLORS[s], label=s) for s in order]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=len(order),
                    title="* = confirmed by running the agent in a guarded subprocess")
    leg.get_title().set_fontsize(7)
    leg.get_title().set_color(MUTED)
    budget = data.get("budget", {})
    sub = ""
    if budget:
        sub = f" (budget: {budget.get('time_per_move')}s/move, {budget.get('memory_bytes', 0) / 2**20:.0f}MB)"
    ax.set_title((title or "Feasibility map") + sub, loc="left")
    fig.tight_layout()
    return _save(fig, path)

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
import numpy as np

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


# -------------------------------------------------------------- AlphaZero
AZ_PANELS = (
    ("score_vs_random", "Score vs random"),
    ("score_vs_rule", "Score vs rule-based"),
    ("optimal_rate", "Optimal-move rate"),
)


def plot_alphazero(runs: Sequence[dict], path: str | Path, title: str | None = None) -> Path:
    """Evaluation after each self-play iteration; one line per board."""
    style()
    runs = list(runs)[:8]
    panels = [p for p in AZ_PANELS if any(h.get(p[0]) is not None for r in runs for h in r["history"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.5), squeeze=False)
    for idx, run in enumerate(runs):
        xs = [h["games"] for h in run["history"]]
        for ax, (key, _) in zip(axes[0], panels):
            ys = [h.get(key) for h in run["history"]]
            pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
            if pts:
                ax.plot(*zip(*pts), color=SERIES[idx], marker="o", markersize=4, label=run["config"])
    for ax, (_, name) in zip(axes[0], panels):
        ax.set_title(name, loc="left")
        ax.set_xlabel("self-play games")
        ax.set_ylim(-0.02, 1.02)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if len(runs) > 1:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title or "AlphaZero-style self-play training", x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout()
    return _save(fig, path)


# ------------------------------------------------- targeted selection (PRISM)
PRISM_COLORS = {"GCMI": SERIES[0], "FLMI": SERIES[1], "FLVMI": SERIES[2], "COM": SERIES[3],
                "LogDetMI": SERIES[6], "FLCMI": SERIES[7]}
BASELINE_STYLES = {"random": ("-", "o"), "entropy": ("--", "v"), "margin": (":", "^"),
                   "FL": ("-.", "s"), "LogDet": ((0, (1, 1)), "D")}


def _method_style(method: str) -> dict:
    if method in PRISM_COLORS:
        return {"color": PRISM_COLORS[method], "linestyle": "-", "marker": "o", "linewidth": 2}
    if method == "oracle":
        return {"color": INK, "linestyle": "--", "marker": "", "linewidth": 1.2}
    ls, mk = BASELINE_STYLES.get(method, ("-", "o"))
    return {"color": MUTED, "linestyle": ls, "marker": mk, "linewidth": 1.5}


def plot_targeted(data: dict, out_dir: str | Path) -> list[Path]:
    """Budget curves per target motif, ablations and game results for the
    targeted-labelling experiment."""
    style()
    out_dir = Path(out_dir)
    summ = data["summary"]
    main = [r for r in summ if not r.get("ablation")]
    targets = [t for t in ("win", "block", "poison") if any(r.get("target") == t for r in main)]
    methods = [m for m in ("random", "entropy", "margin", "FL", "LogDet", "GCMI", "FLMI", "FLVMI", "COM",
                           "LogDetMI", "FLCMI", "oracle") if any(r["method"] == m for r in main)]
    ks = sorted({r["k"] for r in main if r["k"]})
    seeds = max(r["seeds"] for r in main)
    paths = []
    panels = [("acc_target", "target accuracy"), ("acc_critical", "overall accuracy"),
              ("hit_target", "hit rate (selected in motif)")]
    fig, axes = plt.subplots(len(targets), 3, figsize=(14, 3.3 * len(targets)), squeeze=False)
    for row_i, target in enumerate(targets):
        rows_t = [r for r in main if r.get("target") == target]
        base = next((r for r in rows_t if r["method"] == "base"), None)
        qonly = next((r for r in rows_t if r["method"] == "query-only"), None)
        for col, (metric, title) in enumerate(panels):
            ax = axes[row_i][col]
            for m in methods:
                rs = sorted((r for r in rows_t if r["method"] == m and r.get(metric) is not None), key=lambda r: r["k"])
                if not rs:
                    continue
                xs = [r["k"] for r in rs]
                ys = [r[metric] for r in rs]
                ci = [r.get(metric + "_ci", 0.0) for r in rs]
                st = _method_style(m)
                ax.plot(xs, ys, label=m, markersize=3.5, **st)
                if m in ("random", "FLVMI", "LogDetMI"):
                    ax.fill_between(xs, [y - c for y, c in zip(ys, ci)], [y + c for y, c in zip(ys, ci)],
                                    color=st["color"], alpha=0.12, linewidth=0)
            if metric != "hit_target":
                for ref, lab, ls in ((base, "base", "-"), (qonly, "base + Q", ":")):
                    if ref is not None and ref.get(metric) is not None:
                        ax.axhline(ref[metric], color=AXIS, linewidth=1, linestyle=ls)
                        ax.annotate(lab, (1.0, ref[metric]), xycoords=("axes fraction", "data"), fontsize=7,
                                    color=MUTED, ha="right", va="bottom")
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.minorticks_off()
            ax.set_title(f"{target}: {title}", loc="left", fontsize=10)
            if row_i == len(targets) - 1:
                ax.set_xlabel("oracle labels k (log scale)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(12, len(labels)), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(f"Targeted oracle labelling on {data['meta'].get('board', '')}: PRISM measures (colour) vs "
                 f"baselines (grey) and oracle (black); mean of {seeds} seeds, 95% CI bands",
                 x=0.01, ha="left", color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths.append(_save(fig, out_dir / "targeted_budget.png"))

    pooled = data.get("pooled", [])
    if pooled:
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
        for ax, metric, title in ((axes[0], "acc_target", "Target accuracy, averaged over the 3 motifs"),
                                  (axes[1], "hit_target", "Hit rate, averaged over the 3 motifs")):
            for m in methods:
                rs = sorted((r for r in pooled if r["method"] == m and r.get(metric) is not None),
                            key=lambda r: r["k"])
                if rs:
                    ax.plot([r["k"] for r in rs], [r[metric] for r in rs], label=m, markersize=3.5,
                            **_method_style(m))
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.minorticks_off()
            ax.set_xlabel("oracle labels k (log scale)")
            ax.set_title(title, loc="left", fontsize=10)
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_pooled.png"))

    dose = data.get("dose_response") or {}
    if dose.get("points"):
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        groups = [("oracle", INK, "oracle (motif positions only)"), ("random", MUTED, "random / uncertainty / diversity")]
        for p_ in dose["points"]:
            m = p_["method"]
            if m == "oracle":
                color = INK
            elif m in PRISM_COLORS:
                color = PRISM_COLORS[m]
            else:
                color = MUTED
            ax.scatter(p_["n_motif"], p_["gain"], s=10, color=color, alpha=0.55, linewidths=0)
        fit = dose.get("fit")
        if fit:
            xs = np.logspace(np.log10(min(p_["n_motif"] for p_ in dose["points"])),
                             np.log10(max(p_["n_motif"] for p_ in dose["points"])), 50)
            ax.plot(xs, fit["intercept"] + fit["slope_per_doubling"] * np.log2(xs), color=INK, linewidth=1.5,
                    label=f"oracle fit: +{fit['slope_per_doubling'] * 100:.1f} pts per doubling")
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], marker="o", linestyle="", color=c, label=lab) for _, c, lab in groups]
        handles += [Line2D([], [], marker="o", linestyle="", color=PRISM_COLORS["FLVMI"], label="PRISM measures")]
        if fit:
            handles.append(ax.get_lines()[-1])
        ax.legend(handles=handles, loc="upper left")
        ax.set_xscale("log")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_xlabel("motif positions in the training set (selected hits + |Q|, log scale)")
        ax.set_ylabel("target accuracy gain over query-only")
        ax.set_title("Dose-response: gains track the number of motif positions labelled", loc="left", fontsize=10)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_dose.png"))

    abl_target = next((r.get("target") for r in summ if r.get("ablation")), None)
    eta_rows = [r for r in summ if r.get("ablation") == "eta"]
    if eta_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, metric, title in ((axes[0], "acc_target", "Target accuracy"), (axes[1], "hit_target", "Hit rate")):
            for m in ("FLMI", "FLVMI", "COM", "LogDetMI"):
                rs = sorted((r for r in eta_rows if r["method"] == m), key=lambda r: r["eta"])
                if rs:
                    ax.plot([r["eta"] for r in rs], [r[metric] for r in rs], label=m, markersize=4, **_method_style(m))
            ax.set_xscale("log", base=2)
            ax.set_xlabel("eta (query relevance vs diversity)")
            ax.set_title(f"{title} vs eta ({abl_target}, k={eta_rows[0]['k']})", loc="left", fontsize=10)
        axes[0].legend(loc="best")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_eta.png"))

    q_rows = [r for r in summ if r.get("ablation") == "query_size"]
    if q_rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, metric, title in ((axes[0], "acc_target", "Target accuracy"), (axes[1], "hit_target", "Hit rate")):
            for m in ("random", "GCMI", "FLMI", "FLVMI", "LogDetMI"):
                rs = sorted((r for r in q_rows if r["method"] == m), key=lambda r: r["nq"])
                if rs:
                    ax.plot([r["nq"] for r in rs], [r[metric] for r in rs], label=m, markersize=4, **_method_style(m))
            ax.set_xscale("log")
            ax.set_xlabel("query set size |Q|")
            ax.set_title(f"{title} vs |Q| ({abl_target}, k={q_rows[0]['k']})", loc="left", fontsize=10)
        axes[0].legend(loc="best")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_query_size.png"))

    emb = [r for r in summ if r.get("ablation") == "embedding"]
    if emb:
        kab = emb[0]["k"]
        names = [r["method"] for r in emb]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, metric, title in ((axes[0], "acc_target", "Target accuracy"), (axes[1], "hit_target", "Hit rate")):
            net_vals = [next((x.get(metric) for x in main if x["method"] == n and x["k"] == kab
                              and x.get("target") == abl_target), float("nan")) for n in names]
            xs = np.arange(len(names))
            ax.bar(xs - 0.2, net_vals, 0.38, color=SERIES[0], label="network features")
            ax.bar(xs + 0.2, [r.get(metric) for r in emb], 0.38, color=SERIES[3], label="raw board")
            ax.set_xticks(xs, names)
            ax.grid(axis="x", visible=False)
            ax.set_title(f"{title}: similarity features ({abl_target}, k={kab})", loc="left", fontsize=10)
        axes[0].legend(loc="lower right")
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_embedding.png"))

    games = [r for r in main if r.get("nonloss_vs_minimax") is not None]
    if games:
        fig, axes = plt.subplots(1, len(targets), figsize=(4.6 * len(targets), 0.8 + 0.3 * len(methods)),
                                 squeeze=False)
        for ax, target in zip(axes[0], targets):
            g = [r for r in games if r.get("target") == target]
            g.sort(key=lambda r: (r["method"] != "base", r["nonloss_vs_minimax"]))
            for i, r in enumerate(g):
                ax.barh(i, r["nonloss_vs_minimax"], color=_method_style(r["method"])["color"], height=0.6)
                ax.text(r["nonloss_vs_minimax"] + 0.01, i, f"{r['nonloss_vs_minimax']:.2f}", va="center",
                        fontsize=7, color=INK_2)
            ax.set_yticks(range(len(g)), [r["method"] for r in g])
            ax.set_xlim(0, 1.1)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{target}: non-loss vs perfect play", loc="left", fontsize=10)
        fig.tight_layout()
        paths.append(_save(fig, out_dir / "targeted_games.png"))
    return paths

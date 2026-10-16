// Landing page: interactive feasibility map and Elo ladders, drawn from the committed results.
import { fmtNum, fmtTime } from "./agents.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

const NAMES = {
  random: "Random", rule: "Rule-based", minimax: "Minimax (exact)", menace: "MENACE",
  qlearning: "Q-learning", td: "TD(0)", "alphabeta:depth=4": "Alpha-beta d4", "alphabeta:depth=2": "Alpha-beta d2",
  "alphabeta:depth=6": "Alpha-beta d6", "mcts:iterations=1000": "MCTS 1000", "mcts:iterations=200": "MCTS 200",
  alphazero: "AlphaZero-style", llm: "LLM",
};
const name = (a) => NAMES[a] ?? a;
const STATUS = {
  ok: "runs within budget", degraded: "runs with reduced settings", refused: "refuses (would not fit)",
  aborted: "aborted by its guards", unavailable: "unavailable here",
};
const fmtBytes = (b) => b == null ? "–" : b < 1024 ** 2 ? `${(b / 1024).toFixed(0)} KB` : b < 1024 ** 3 ? `${(b / 1024 ** 2).toFixed(0)} MB` : b < 1024 ** 4 ? `${(b / 1024 ** 3).toFixed(1)} GB` : `${fmtNum(b / 1024 ** 4)} TB`;

// ---------------------------------------------------------------- feasibility map
let feas = null, mapName = "standard", selected = null;

function budgetText(b) {
  return `${fmtTime(b.time_per_move)} per move, ${fmtBytes(b.memory_bytes)} memory, ${fmtTime(b.train_time)} training`;
}

function renderMap() {
  const m = feas.maps[mapName], byKey = new Map(m.cells.map((c) => [c.agent + "|" + c.config, c]));
  const t = document.createElement("table");
  t.className = "fmap";
  const head = t.createTHead().insertRow();
  head.appendChild(document.createElement("th"));
  for (const b of m.boards) { const th = document.createElement("th"); th.textContent = b; th.scope = "col"; head.appendChild(th); }
  const body = t.createTBody();
  for (const a of m.agents) {
    const tr = body.insertRow(), th = document.createElement("th");
    th.className = "agent"; th.scope = "row"; th.textContent = name(a);
    tr.appendChild(th);
    for (const b of m.boards) {
      const c = byKey.get(a + "|" + b), td = tr.insertCell();
      td.className = "s-" + c.status + (c.measured ? "" : " est");
      td.tabIndex = 0;
      td.title = `${name(a)} on ${b}: ${STATUS[c.status]}${c.measured ? "" : ", estimate only"}${c.reason ? " (" + c.reason + ")" : ""}`;
      td.setAttribute("aria-label", td.title);
      const pick = () => {
        selected = [a, b]; showDetail(c);
        document.querySelectorAll(".fmap td.sel").forEach((x) => x.classList.remove("sel")); td.classList.add("sel");
        revealDetail();
      };
      td.onclick = pick;
      td.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
      if (selected && selected[0] === a && selected[1] === b) { td.classList.add("sel"); showDetail(c); }
    }
  }
  $("#fmap").replaceChildren(t);
  $("#budget").textContent = budgetText(m.budget);
  if (!selected) showDetail(byKey.get("minimax|(5,5,4)") ?? m.cells[0]);
}

// On narrow screens the detail card stacks below the map, so a tap would otherwise update it off-screen.
function revealDetail() {
  const d = $("#fdetail"), r = d.getBoundingClientRect();
  if (r.top >= 0 && r.bottom <= innerHeight) return;
  const smooth = !matchMedia("(prefers-reduced-motion: reduce)").matches;
  d.scrollIntoView({ block: "nearest", behavior: smooth ? "smooth" : "auto" });
}

function showDetail(c) {
  const e = c.estimate ?? {}, p = c.probe ?? c.frontier_probe;
  const rows = [
    ["Estimated time per move", e.time_per_move != null ? fmtTime(e.time_per_move) : "–"],
    ["Estimated memory", fmtBytes(e.memory_bytes)],
    e.train_time ? ["Estimated training", fmtTime(e.train_time)] : null,
    e.cost_per_move ? ["Estimated cost per move", "$" + e.cost_per_move.toFixed(4)] : null,
    p && p.seconds_per_move != null ? ["Measured time per move", fmtTime(p.seconds_per_move)] : null,
    p && p.peak_rss ? ["Measured peak memory", fmtBytes(p.peak_rss)] : null,
    p && p.status === "aborted" ? ["Forced probe", "aborted: " + p.reason] : null,
    c.degraded_params && Object.keys(c.degraded_params).length ? ["Degraded to", Object.entries(c.degraded_params).map(([k, v]) => `${k}=${v}`).join(", ")] : null,
  ].filter(Boolean);
  $("#fdetail").innerHTML = `
    <div class="row" style="margin-bottom:6px"><b>${esc(name(c.agent))}</b> on <b>${esc(c.config)}</b>
      <span class="pill s-${c.status}">${c.status}</span></div>
    ${c.reason ? `<p class="small" style="margin:0 0 6px">${esc(c.reason)}</p>` : ""}
    ${e.notes ? `<p class="small muted" style="margin:0 0 6px">${esc(e.notes)}</p>` : ""}
    <table class="data small"><tbody>${rows.map(([k, v]) => `<tr><td class="muted">${k}</td><td class="num">${esc(v)}</td></tr>`).join("")}</tbody></table>
    ${c.measured ? "" : '<p class="small muted" style="margin:6px 0 0">Estimate only: this cell was not probed.</p>'}`;
}

// ---------------------------------------------------------------- Elo ladders
let elo = null;
function renderElo(idx) {
  const b = elo.boards[idx], r = b.elo.ratings, ci = b.elo.ci95 ?? {};
  const rows = Object.entries(r).sort((x, y) => y[1] - x[1]);
  const max = rows[0][1], min = Math.min(800, ...rows.map((x) => x[1]));
  $("#elo").innerHTML = rows.map(([a, v]) => {
    const c = ci[a], tip = c ? `95% CI ${Math.round(c[0])}–${Math.round(c[1])}` : "";
    return `<div class="bar" title="${tip}"><span>${esc(name(a))}</span><span class="track"><span class="fill" style="width:${(100 * (v - min) / (max - min)).toFixed(1)}%;${a === "random" ? "background:var(--unavailable)" : ""}"></span></span><span class="v">${Math.round(v)}</span></div>`;
  }).join("");
  const t = b.tournament;
  $("#elo-note").textContent = `${t.agents.length} agents, ${t.games_per_pair} games per pair, sides alternated. Random is pinned at 1000. Hover a bar for its 95% bootstrap interval.`;
}

// ---------------------------------------------------------------- hero: the boards growing
// Every board is drawn at the same cell size, so the strip shows the growth itself.
const SVGNS = "http://www.w3.org/2000/svg";
const MINI = {
  ttt: { m: 3, n: 3, x: [[0, 0], [1, 1], [2, 2]], o: [[0, 2], [1, 0]], win: [[0, 0], [1, 1], [2, 2]] },
  four: { m: 4, n: 4, x: [[1, 1], [2, 2], [1, 2]], o: [[2, 1], [0, 3], [3, 0]] },
  five: { m: 5, n: 5, x: [[2, 2], [1, 3], [3, 2]], o: [[1, 1], [2, 3], [3, 1]] },
  c4: { m: 6, n: 7, gravity: true, x: [[5, 3], [4, 3], [5, 4], [3, 2]], o: [[5, 2], [4, 2], [5, 1], [4, 4]] },
  gomoku: { m: 15, n: 15, goban: true, x: [[7, 7], [6, 8], [8, 6], [6, 6], [5, 9]], o: [[7, 8], [8, 7], [7, 6], [9, 5]] },
};
const svg = (tag, attrs) => { const e = document.createElementNS(SVGNS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

function drawMini(el, spec) {
  const { m, n } = spec;
  el.setAttribute("viewBox", `0 0 ${n} ${m}`);
  el.style.setProperty("--n", n);
  el.style.setProperty("--m", m);
  el.classList.add(spec.goban ? "goban" : "tiles");
  if (spec.gravity) el.classList.add("gravity");
  el.appendChild(svg("rect", { class: "bed", x: 0, y: 0, width: n, height: m, rx: spec.goban ? 0.3 : 0.25 }));
  for (const [r, c] of spec.win ?? []) el.appendChild(svg("rect", { class: "win", x: c + 0.06, y: r + 0.06, width: 0.88, height: 0.88, rx: 0.12 }));
  // tile boards draw the borders between cells; the goban draws lines through the cell centres
  const lines = (count, at) => { for (let i = 0; i < count; i++) at(i); };
  if (spec.goban) {
    lines(m, (r) => el.appendChild(svg("line", { x1: 0.5, x2: n - 0.5, y1: r + 0.5, y2: r + 0.5 })));
    lines(n, (c) => el.appendChild(svg("line", { y1: 0.5, y2: m - 0.5, x1: c + 0.5, x2: c + 0.5 })));
  } else {
    lines(m - 1, (r) => el.appendChild(svg("line", { x1: 0, x2: n, y1: r + 1, y2: r + 1 })));
    lines(n - 1, (c) => el.appendChild(svg("line", { y1: 0, y2: m, x1: c + 1, x2: c + 1 })));
  }
  if (spec.goban) for (const [r, c] of [[3, 3], [3, 11], [11, 3], [11, 11], [7, 7]]) el.appendChild(svg("circle", { class: "star", cx: c + 0.5, cy: r + 0.5, r: 0.13 }));
  const rad = spec.goban ? 0.46 : 0.34;
  for (const side of ["x", "o"]) for (const [r, c] of spec[side]) el.appendChild(svg("circle", { class: "stone " + side, cx: c + 0.5, cy: r + 0.5, r: rad }));
}

document.querySelectorAll("svg[data-mini]").forEach((el) => drawMini(el, MINI[el.dataset.mini]));

// One pip per agent that can run here (LLM players need an API key, so they are left out).
function renderSurvivors() {
  const m = feas.maps.standard;
  document.querySelectorAll(".growth-strip li[data-board]").forEach((li) => {
    const cells = m.cells.filter((c) => c.config === li.dataset.board && c.status !== "unavailable");
    if (!cells.length) return;
    const by = (s) => cells.filter((c) => c.status === s).length;
    const ok = by("ok"), cut = by("degraded"), gone = cells.length - ok - cut;
    const parts = [ok === cells.length ? `All ${ok} run` : `${ok} run`];
    if (cut) parts.push(`${cut} cut back`);
    if (gone) parts.push(`${gone} refuse`);
    li.querySelector(".count").textContent = parts.join(", ");
    li.querySelector(".pips").replaceChildren(...cells.map((c) => {
      const i = document.createElement("i");
      i.className = "pip p-" + (c.status === "ok" || c.status === "degraded" ? c.status : "refused");
      return i;
    }));
  });
}

async function init() {
  try {
    const [f, e, g] = await Promise.all([
      fetch("results/feasibility/feasibility.json").then((r) => r.json()),
      fetch("results/elo/elo.json").then((r) => r.json()),
      fetch("results/gomoku/gomoku.json").then((r) => r.json()).catch(() => null),
    ]);
    feas = f; elo = e;
    if (g) elo.boards.push({ config: "gomoku (15,15,5)", elo: g.elo, tournament: g.tournament });
  } catch (err) {
    $("#fmap").textContent = "Could not load the results data.";
    console.error(err);
    return;
  }
  document.querySelectorAll("#map-toggle button").forEach((btn) => {
    btn.onclick = () => {
      mapName = btn.dataset.map;
      document.querySelectorAll("#map-toggle button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
      renderMap();
    };
  });
  renderMap();
  renderSurvivors();
  const tabs = $("#elo-tabs");
  elo.boards.forEach((b, i) => {
    const btn = document.createElement("button");
    btn.type = "button"; btn.textContent = b.config;
    btn.setAttribute("aria-pressed", String(i === 1));
    btn.onclick = () => { tabs.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === btn))); renderElo(i); };
    tabs.appendChild(btn);
  });
  renderElo(Math.min(1, elo.boards.length - 1));
}

init();

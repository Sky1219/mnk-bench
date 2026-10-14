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
      td.className = "s-" + c.status;
      td.tabIndex = 0;
      td.textContent = c.measured ? "" : "·";
      td.title = `${name(a)} on ${b}: ${STATUS[c.status]}${c.reason ? " (" + c.reason + ")" : ""}`;
      td.setAttribute("aria-label", td.title);
      const pick = () => { selected = [a, b]; showDetail(c); document.querySelectorAll(".fmap td.sel").forEach((x) => x.classList.remove("sel")); td.classList.add("sel"); };
      td.onclick = pick;
      td.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
      if (selected && selected[0] === a && selected[1] === b) { td.classList.add("sel"); showDetail(c); }
    }
  }
  $("#fmap").replaceChildren(t);
  $("#budget").textContent = budgetText(m.budget);
  if (!selected) showDetail(byKey.get("minimax|(5,5,4)") ?? m.cells[0]);
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
      <span class="pill s-${c.status}" style="color:#fff">${c.status}</span></div>
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

// Arena: queue many games and play them in parallel on a pool of Web Workers.
import { getGame, X, O, DRAW } from "./mnk.js";
import { AGENTS, fmtTime } from "./agents.js";
import { PRESETS, PLAYERS, fillPlayerSelect, fillLevelSelect, playerParams, playerName, boardLabel } from "./players.js";

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const CORES = Math.max(1, navigator.hardwareConcurrency || 2);
const MAX_WORKERS = Math.min(16, CORES);

const ROUND_ROBIN = [
  { kind: "random", level: null },
  { kind: "rule", level: null },
  { kind: "menace", level: "2000" },
  { kind: "minimax", level: "5" },
  { kind: "alphabeta", level: "2" },
  { kind: "alphabeta", level: "4" },
  { kind: "mcts", level: "200" },
  { kind: "mcts", level: "1000" },
];

const A = { matchups: [], queue: [], workers: [], running: false, nextId: 1, started: 0, elapsed: 0, tick: null, dirty: false };

// ---------------------------------------------------------------- board + players setup
function readBoard() {
  const v = $("#board-preset").value;
  if (v !== "custom") return { gravity: false, exact: false, ...PRESETS.find((p) => p.id === v).cfg };
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, Math.round(+x || lo)));
  const m = clamp($("#m").value, 2, 19), n = clamp($("#n").value, 2, 19);
  return { m, n, k: clamp($("#k").value, 2, Math.max(m, n)), gravity: $("#gravity").checked, exact: $("#exact").checked };
}

function setupPlayer(prefix, kind, level) {
  const kindSel = $(`#${prefix}-kind`), levelSel = $(`#${prefix}-level`);
  fillPlayerSelect(kindSel, { human: false });
  kindSel.value = kind;
  fillLevelSelect(levelSel, kind, level);
  kindSel.onchange = () => fillLevelSelect(levelSel, kindSel.value, null);
  return () => ({ kind: kindSel.value, level: levelSel.disabled ? null : levelSel.value });
}

function addMatchup(cfg, a, b, games, alternate) {
  A.matchups.push({
    id: A.nextId++, cfg, a, b, games, alternate, seed: (Math.random() * 2 ** 31) | 0,
    done: 0, queued: 0, aWins: 0, draws: 0, bWins: 0, plies: 0,
    time: { a: 0, b: 0 }, moves: { a: 0, b: 0 }, notes: new Set(), log: [],
  });
  renderTable();
}

// ---------------------------------------------------------------- worker pool
function workerCount() { return +$("#workers").value; }

function buildQueue() {
  A.queue = [];
  for (const mu of A.matchups) {
    for (let gi = mu.done + mu.queued; gi < mu.games; gi++) A.queue.push({ mu, gi });
  }
  // interleave matchups so every row makes progress at the same time
  A.queue.sort((x, y) => x.gi - y.gi || x.mu.id - y.mu.id);
}

function run() {
  if (A.running) return;
  buildQueue();
  if (!A.queue.length) { setRunState("Nothing to run: add a matchup, or every matchup is finished."); return; }
  A.running = true;
  A.started = performance.now();
  const n = Math.min(workerCount(), A.queue.length);
  renderLiveSlots(n);
  for (let i = 0; i < n; i++) {
    const w = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    const slot = { w, i, job: null, state: null };
    w.onmessage = (e) => onMessage(slot, e.data);
    w.onerror = (e) => { console.error(e); finishJob(slot, null, `worker error: ${e.message}`); };
    A.workers.push(slot);
    dispatch(slot);
  }
  A.tick = setInterval(renderProgress, 250);
  updateButtons();
}

function dispatch(slot) {
  const job = A.queue.shift();
  if (!job || !A.running) {
    // keep the last finished game on screen
    const cap = $("#live-grid").children[slot.i]?.querySelector(".cap");
    if (cap && slot.job) cap.insertAdjacentHTML("beforeend", " · finished");
    slot.job = null;
    if (A.workers.every((s) => !s.job)) stop(true);
    return;
  }
  const { mu, gi } = job, swap = mu.alternate && gi % 2 === 1;
  mu.queued++;
  const pa = { kind: mu.a.kind, params: playerParams(mu.a) }, pb = { kind: mu.b.kind, params: playerParams(mu.b) };
  slot.job = { mu, gi, swap, id: A.nextId++ };
  slot.state = getGame(mu.cfg).newState();
  slot.w.postMessage({
    type: "game", id: slot.job.id, config: mu.cfg, seed: mu.seed + gi, live: $("#live").checked,
    players: swap ? { [X]: pb, [O]: pa } : { [X]: pa, [O]: pb },
  });
  drawSlot(slot);
}

function onMessage(slot, msg) {
  if (!slot.job || msg.id !== slot.job.id) return;
  if (msg.type === "move") {
    slot.state.push(msg.move);
    slot.dirty = true;
    scheduleDraw();
  } else if (msg.type === "done") {
    finishJob(slot, msg);
  } else if (msg.type === "error") {
    console.error(msg.error);
    finishJob(slot, null, msg.error.split("\n")[0]);
  }
}

function finishJob(slot, res, error) {
  if (slot.dirty) drawSlot(slot); // show the final position before the next game replaces it
  const { mu, swap, gi } = slot.job;
  mu.queued--;
  if (res) {
    mu.done++;
    const aSide = swap ? O : X, bSide = swap ? X : O;
    const outcome = res.winner === DRAW ? "draw" : res.winner === aSide ? "a" : "b";
    if (outcome === "a") mu.aWins++; else if (outcome === "b") mu.bWins++; else mu.draws++;
    mu.plies += res.plies;
    mu.time.a += res.time[aSide]; mu.moves.a += res.moves[aSide];
    mu.time.b += res.time[bSide]; mu.moves.b += res.moves[bSide];
    res.notes.forEach((n) => mu.notes.add(n));
    mu.log.push({ game: gi + 1, aPlays: swap ? "O" : "X", outcome, plies: res.plies });
  } else {
    mu.notes.add(error);
  }
  renderTable();
  dispatch(slot);
}

function stop(finished = false) {
  if (!A.running && !A.workers.length) return;
  for (const s of A.workers) {
    if (s.job) s.job.mu.queued--;
    s.w.terminate();
  }
  A.workers = [];
  A.queue = [];
  A.elapsed += (performance.now() - A.started) / 1000;
  A.running = false;
  clearInterval(A.tick);
  renderTable();
  renderProgress();
  setRunState(finished ? "All queued games finished." : "Stopped. Unfinished games were dropped; press Run to continue.");
  updateButtons();
}

// ---------------------------------------------------------------- live boards
let drawPending = false;
function scheduleDraw() {
  if (drawPending) return;
  drawPending = true;
  requestAnimationFrame(() => { drawPending = false; A.workers.forEach((s) => s.dirty && s.job && drawSlot(s)); });
}

function renderLiveSlots(n) {
  const wrap = $("#live-grid");
  wrap.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const card = document.createElement("div");
    card.className = "live-slot";
    card.innerHTML = `<canvas width="150" height="150"></canvas><div class="small muted cap">idle</div>`;
    wrap.appendChild(card);
  }
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawSlot(slot) {
  slot.dirty = false;
  const card = $("#live-grid").children[slot.i];
  if (!card) return;
  const cv = card.querySelector("canvas"), cap = card.querySelector(".cap");
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth || 150;
  if (cv.width !== W * dpr) { cv.width = W * dpr; cv.height = W * dpr; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, W);
  if (!slot.job) { cap.textContent = "idle"; return; }
  const s = slot.state, g = s.game, { mu, gi, swap } = slot.job;
  const cell = W / Math.max(g.m, g.n), ox = (W - cell * g.n) / 2, oy = (W - cell * g.m) / 2;
  const goban = !g.gravity && g.m >= 9 && g.n >= 9;
  ctx.fillStyle = goban ? css("--wood") : css("--surface-2");
  ctx.fillRect(ox, oy, cell * g.n, cell * g.m);
  ctx.strokeStyle = goban ? css("--grid") : css("--line");
  ctx.lineWidth = 1;
  ctx.beginPath();
  if (goban) {
    for (let r = 0; r < g.m; r++) { ctx.moveTo(ox + cell / 2, oy + (r + .5) * cell); ctx.lineTo(ox + (g.n - .5) * cell, oy + (r + .5) * cell); }
    for (let c = 0; c < g.n; c++) { ctx.moveTo(ox + (c + .5) * cell, oy + cell / 2); ctx.lineTo(ox + (c + .5) * cell, oy + (g.m - .5) * cell); }
  } else {
    for (let r = 0; r <= g.m; r++) { ctx.moveTo(ox, oy + r * cell); ctx.lineTo(ox + g.n * cell, oy + r * cell); }
    for (let c = 0; c <= g.n; c++) { ctx.moveTo(ox + c * cell, oy); ctx.lineTo(ox + c * cell, oy + g.m * cell); }
  }
  ctx.stroke();
  const xc = css("--x"), oc = css("--o"), last = s.history[s.history.length - 1];
  for (let i = 0; i < g.size; i++) {
    if (!s.cells[i]) continue;
    const cx = ox + (i % g.n + .5) * cell, cy = oy + (Math.floor(i / g.n) + .5) * cell;
    ctx.fillStyle = s.cells[i] === X ? xc : oc;
    ctx.beginPath(); ctx.arc(cx, cy, cell * .4, 0, Math.PI * 2); ctx.fill();
    if (i === last) { ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(cx, cy, cell * .12, 0, Math.PI * 2); ctx.fill(); }
  }
  const xName = playerName(swap ? mu.b : mu.a), oName = playerName(swap ? mu.a : mu.b);
  cap.innerHTML = `<span style="color:var(--x)">●</span> ${esc(xName)}<br><span style="color:var(--o)">●</span> ${esc(oName)}<br>${esc(boardLabel(mu.cfg))} · game ${gi + 1}/${mu.games}`;
}

// ---------------------------------------------------------------- tables
function score(mu) { return mu.done ? (mu.aWins + mu.draws / 2) / mu.done : null; }
function msPerMove(t, n) { return n ? fmtTime(t / n / 1000) : "–"; }

function renderTable() {
  const body = $("#matchups tbody");
  if (!A.matchups.length) {
    body.innerHTML = `<tr><td colspan="11" class="muted">No matchups yet. Add one, or a round robin, above.</td></tr>`;
  } else {
    body.innerHTML = A.matchups.map((mu) => {
      const pct = (100 * mu.done / mu.games).toFixed(0), sc = score(mu);
      return `<tr>
        <td>${esc(boardLabel(mu.cfg))}</td>
        <td><b>${esc(playerName(mu.a))}</b></td>
        <td>${esc(playerName(mu.b))}</td>
        <td><div class="progress" title="${mu.done}/${mu.games}"><span style="width:${pct}%"></span></div><span class="small muted">${mu.done}/${mu.games}${mu.queued ? " · " + mu.queued + " running" : ""}</span></td>
        <td class="num">${mu.aWins}</td><td class="num">${mu.draws}</td><td class="num">${mu.bWins}</td>
        <td class="num"><b>${sc == null ? "–" : sc.toFixed(2)}</b></td>
        <td class="num">${mu.done ? (mu.plies / mu.done).toFixed(1) : "–"}</td>
        <td class="num small">${msPerMove(mu.time.a, mu.moves.a)} / ${msPerMove(mu.time.b, mu.moves.b)}</td>
        <td><button class="btn small icon" data-remove="${mu.id}" type="button" aria-label="Remove matchup" ${mu.queued ? "disabled" : ""}>×</button></td>
      </tr>${mu.notes.size ? `<tr class="notes"><td colspan="11" class="small muted">${[...mu.notes].map(esc).join(" · ")}</td></tr>` : ""}`;
    }).join("");
  }
  renderStandings();
  renderProgress();
  updateButtons();
}

function renderStandings() {
  const rows = new Map();
  for (const mu of A.matchups) {
    if (!mu.done) continue;
    for (const [p, w, l] of [[mu.a, mu.aWins, mu.bWins], [mu.b, mu.bWins, mu.aWins]]) {
      const key = boardLabel(mu.cfg) + "|" + playerName(p);
      const r = rows.get(key) ?? { board: boardLabel(mu.cfg), name: playerName(p), g: 0, w: 0, d: 0, l: 0 };
      r.g += mu.done; r.w += w; r.d += mu.draws; r.l += l;
      rows.set(key, r);
    }
  }
  const list = [...rows.values()].sort((x, y) => x.board.localeCompare(y.board) || (y.w + y.d / 2) / y.g - (x.w + x.d / 2) / x.g);
  $("#standings-card").hidden = !list.length;
  $("#standings tbody").innerHTML = list.map((r) => `<tr><td>${esc(r.board)}</td><td>${esc(r.name)}</td><td class="num">${r.g}</td>
    <td class="num">${r.w}</td><td class="num">${r.d}</td><td class="num">${r.l}</td><td class="num"><b>${((r.w + r.d / 2) / r.g).toFixed(2)}</b></td></tr>`).join("");
}

function renderProgress() {
  const total = A.matchups.reduce((t, m) => t + m.games, 0), done = A.matchups.reduce((t, m) => t + m.done, 0);
  const secs = A.elapsed + (A.running ? (performance.now() - A.started) / 1000 : 0);
  $("#overall").textContent = total ? `${done}/${total} games · ${secs.toFixed(1)} s${A.running ? ` · ${A.workers.filter((s) => s.job).length} running in parallel` : ""}` : "";
}

function setRunState(text) { $("#run-state").textContent = text; }

function updateButtons() {
  $("#run").disabled = A.running;
  $("#stop").disabled = !A.running;
  $("#clear").disabled = A.running || !A.matchups.length;
  $("#csv").disabled = !A.matchups.some((m) => m.done);
}

function downloadCsv() {
  const head = ["board", "player_a", "player_b", "games", "a_wins", "draws", "b_wins", "a_score", "avg_plies", "a_ms_per_move", "b_ms_per_move", "notes"];
  const lines = [head.join(",")];
  for (const mu of A.matchups) {
    const q = (s) => `"${String(s).replace(/"/g, '""')}"`;
    lines.push([q(boardLabel(mu.cfg)), q(playerName(mu.a)), q(playerName(mu.b)), mu.done, mu.aWins, mu.draws, mu.bWins,
      score(mu)?.toFixed(3) ?? "", mu.done ? (mu.plies / mu.done).toFixed(1) : "",
      mu.moves.a ? (mu.time.a / mu.moves.a).toFixed(2) : "", mu.moves.b ? (mu.time.b / mu.moves.b).toFixed(2) : "",
      q([...mu.notes].join("; "))].join(","));
  }
  const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: "mnk-arena.csv" });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------------------------------------------------------------- init
function init() {
  const preset = $("#board-preset");
  PRESETS.forEach((p) => preset.appendChild(new Option(`${p.name} ${boardLabel(p.cfg)}`, p.id)));
  preset.appendChild(new Option("Custom…", "custom"));
  preset.value = "664";
  preset.onchange = () => { $("#custom").hidden = preset.value !== "custom"; };

  const readA = setupPlayer("a", "alphabeta", "4"), readB = setupPlayer("b", "mcts", "1000");
  $("#add").onclick = () => {
    addMatchup(readBoard(), readA(), readB(), Math.max(1, Math.min(1000, +$("#games").value || 10)), $("#alternate").checked);
  };

  const rr = $("#rr-list");
  ROUND_ROBIN.forEach((p, i) => {
    const lab = document.createElement("label");
    lab.innerHTML = `<input type="checkbox" value="${i}" ${[1, 2, 5, 7].includes(i) ? "checked" : ""}> ${esc(playerName(p))}`;
    rr.appendChild(lab);
  });
  $("#add-rr").onclick = () => {
    const picked = [...rr.querySelectorAll("input:checked")].map((c) => ROUND_ROBIN[+c.value]);
    if (picked.length < 2) { setRunState("Pick at least two players for a round robin."); return; }
    const cfg = readBoard(), games = Math.max(2, Math.min(1000, +$("#games").value || 10));
    for (let i = 0; i < picked.length; i++) for (let j = i + 1; j < picked.length; j++) addMatchup(cfg, picked[i], picked[j], games, true);
    setRunState(`Added ${picked.length * (picked.length - 1) / 2} matchups on ${boardLabel(cfg)}.`);
  };

  const ws = $("#workers");
  for (let i = 1; i <= MAX_WORKERS; i++) ws.appendChild(new Option(`${i}`, i));
  ws.value = Math.max(1, Math.min(MAX_WORKERS, CORES - 1, 8));
  $("#cores").textContent = `This device reports ${CORES} logical core${CORES === 1 ? "" : "s"}.`;

  $("#run").onclick = run;
  $("#stop").onclick = () => stop(false);
  $("#clear").onclick = () => { A.matchups = []; A.elapsed = 0; renderTable(); setRunState(""); $("#live-grid").innerHTML = ""; };
  $("#csv").onclick = downloadCsv;
  $("#matchups").addEventListener("click", (e) => {
    const id = e.target.closest("[data-remove]")?.dataset.remove;
    if (id) { A.matchups = A.matchups.filter((m) => m.id !== +id); renderTable(); }
  });
  $("#agent-styles").innerHTML = Object.entries(PLAYERS).filter(([k]) => k !== "human")
    .map(([k, p]) => `<li><b>${esc(p.label)}</b>: ${esc(AGENTS[k].blurb)}</li>`).join("");
  renderTable();
}

init();

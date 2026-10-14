// Play page: any m,n,k board, any mix of humans and agents. Agents think in a Web Worker.
import { getGame, X, O, DRAW, positionBound } from "./mnk.js";
import { AGENTS, minimaxEstimate, fmtNum, fmtTime } from "./agents.js";
import { PRESETS, PLAYERS, STYLES, fillPlayerSelect, fillLevelSelect, playerParams } from "./players.js";


const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// ---------------------------------------------------------------- state
const ui = {
  cfg: { ...PRESETS[4].cfg },
  players: { [X]: { kind: "human", level: null }, [O]: { kind: "alphabeta", level: "4" } },
  state: null,
  score: { [X]: 0, [O]: 0, [DRAW]: 0 },
  paused: false,
  delay: 400,
  lastInfo: null,
};
let worker = null, reqId = 0, pending = null, thinkTimer = null;

function readHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const b = h.get("b");
  if (b) {
    const m = b.match(/^(\d+),(\d+),(\d+)((?:\+\w+)*)$/);
    if (m) {
      const [mm, nn, kk] = [+m[1], +m[2], +m[3]];
      if (mm >= 1 && nn >= 1 && mm <= 19 && nn <= 19 && kk >= 2 && kk <= Math.max(mm, nn))
        ui.cfg = { m: mm, n: nn, k: kk, gravity: m[4].includes("gravity"), exact: m[4].includes("exact") };
    }
  }
  for (const [side, key] of [[X, "x"], [O, "o"]]) {
    const v = h.get(key);
    if (!v) continue;
    const [kind, level] = v.split(":");
    if (PLAYERS[kind]) ui.players[side] = { kind, level: level ?? PLAYERS[kind].def ?? null };
  }
}

function writeHash() {
  const c = ui.cfg, p = ui.players;
  const b = `${c.m},${c.n},${c.k}${c.gravity ? "+gravity" : ""}${c.exact ? "+exact" : ""}`;
  const fmt = (s) => s.level ? `${s.kind}:${s.level}` : s.kind;
  history.replaceState(null, "", `#b=${b}&x=${fmt(p[X])}&o=${fmt(p[O])}`);
}

// ---------------------------------------------------------------- worker
function ensureWorker() {
  if (!worker) {
    worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    worker.onmessage = onWorkerMessage;
    worker.onerror = (e) => { setThinking(null); showStatus(`Agent crashed: ${e.message}`); };
  }
  return worker;
}

function cancelThinking() {
  if (pending && worker) { worker.terminate(); worker = null; }
  pending = null;
  setThinking(null);
}

function requestMove() {
  const s = ui.state, side = s.toMove, p = ui.players[side];
  const id = ++reqId;
  pending = { id, side, history: s.history.slice() };
  setThinking(side);
  ensureWorker().postMessage({
    id, config: ui.cfg, history: s.history, agent: p.kind,
    params: playerParams(p),
  });
}

function onWorkerMessage(e) {
  const { id, move, info, ms, error } = e.data;
  if (!pending || id !== pending.id) return; // stale answer from a game that has moved on
  pending = null;
  setThinking(null);
  if (error) { showStatus("The agent hit an error: " + error.split("\n")[0]); console.error(error); return; }
  ui.lastInfo = { side: ui.state.toMove, kind: ui.players[ui.state.toMove].kind, info, ms };
  play(move);
}

// ---------------------------------------------------------------- game flow
function newGame() {
  cancelThinking();
  ui.state = getGame(ui.cfg).newState();
  ui.lastInfo = null;
  writeHash();
  renderBoard(true);
  renderFeasibility();
  renderInfo();
  renderLog();
  step();
}

function play(mv) {
  const s = ui.state;
  if (!s.isLegal(mv)) return;
  s.push(mv);
  renderBoard();
  renderLog();
  renderInfo();
  if (s.winner !== null) {
    ui.score[s.winner]++;
    renderScore();
  }
  step();
}

function isHuman(side) { return ui.players[side].kind === "human"; }
function bothAI() { return !isHuman(X) && !isHuman(O); }

function step() {
  const s = ui.state;
  updateStatus();
  updateButtons();
  if (s.winner !== null || isHuman(s.toMove) || pending) return;
  if (bothAI()) {
    if (ui.paused) return;
    clearTimeout(thinkTimer);
    thinkTimer = setTimeout(() => { if (!pending && ui.state === s && s.winner === null && !ui.paused) requestMove(); }, ui.delay);
  } else {
    requestMove();
  }
}

function undo() {
  cancelThinking();
  clearTimeout(thinkTimer);
  const s = ui.state;
  if (!s.history.length) return;
  if (s.winner !== null) ui.score[s.winner]--;
  // Take back to the last position where a human was to move.
  do s.pop(); while (s.history.length && !isHuman(s.toMove));
  ui.lastInfo = null;
  if (bothAI()) ui.paused = true;
  renderBoard(true);
  renderLog();
  renderInfo();
  renderScore();
  step();
}

// ---------------------------------------------------------------- rendering
const boardEl = () => $("#board");

/** Star points (hoshi): four near the corners and the centre, plus the sides on 19×19. */
function starPoints(m, n) {
  const off = Math.min(m, n) >= 13 ? 3 : 2;
  const rows = [off, m - 1 - off], cols = [off, n - 1 - off], out = new Set();
  const cr = m % 2 ? (m - 1) / 2 : null, cc = n % 2 ? (n - 1) / 2 : null;
  for (const r of rows) for (const c of cols) out.add(r * n + c);
  if (cr !== null && cc !== null) out.add(cr * n + cc);
  if (m >= 19 && n >= 19) {
    for (const r of rows) out.add(r * n + cc);
    for (const c of cols) out.add(cr * n + c);
  }
  return out;
}

function renderBoard(full = false) {
  const g = getGame(ui.cfg), s = ui.state, b = boardEl();
  if (full || b.children.length !== g.size) {
    b.innerHTML = "";
    b.style.setProperty("--cells", g.n);
    b.style.setProperty("--aspect", `${g.n} / ${g.m}`);
    const boardMax = Math.min(620, Math.max(300, g.n * 64));
    b.style.setProperty("--board-max", `${boardMax}px`);
    // size the board column to the board so the side panels sit right next to it
    document.querySelector(".play-layout").style.setProperty("--play-board", `${Math.max(440, boardMax)}px`);
    b.classList.toggle("gravity", !!g.gravity);
    // Gomoku-sized free-placement boards are drawn like a go board: stones sit on
    // the intersections of the grid lines. Small boards and gravity boards keep cells.
    const goban = !g.gravity && g.m >= 9 && g.n >= 9;
    b.classList.toggle("goban", goban);
    b.classList.toggle("small-gap", !goban && g.n > 11);
    const stars = goban ? starPoints(g.m, g.n) : new Set();
    for (let i = 0; i < g.size; i++) {
      const c = el("button", "cell");
      c.type = "button";
      c.dataset.cell = i;
      if (goban) {
        const r = Math.floor(i / g.n), col = i % g.n;
        if (col === 0) c.classList.add("el");
        if (col === g.n - 1) c.classList.add("er");
        if (r === 0) c.classList.add("et");
        if (r === g.m - 1) c.classList.add("eb");
        if (stars.has(i)) c.classList.add("star");
      }
      b.appendChild(c);
    }
  }
  const humanTurn = s.winner === null && isHuman(s.toMove) && !pending;
  const last = s.history[s.history.length - 1];
  const line = new Set(s.winningLine());
  const legal = new Set(humanTurn ? s.legalMoves() : []);
  [...b.children].forEach((c, i) => {
    const v = s.cells[i];
    const want = v === X ? "x" : v === O ? "o" : "";
    const stone = c.firstChild;
    if ((stone ? stone.dataset.p : "") !== want) {
      c.innerHTML = "";
      if (want) { const st = el("span", "stone " + want); st.dataset.p = want; c.appendChild(st); }
    }
    const playable = humanTurn && (g.gravity ? s.colTop[i % g.n] >= 0 : legal.has(i));
    c.classList.toggle("playable", playable);
    c.classList.toggle("last", i === last);
    c.classList.toggle("win", line.has(i));
    c.setAttribute("aria-label", `${g.cellName(i)}${want ? " " + want.toUpperCase() : playable ? " empty" : ""}`);
    c.disabled = !playable;
  });
  b.classList.add("drop");
}

function sideName(side) {
  const p = ui.players[side];
  if (p.kind === "human") return bothAI() ? "Human" : (isHuman(3 - side) ? `Player ${side === X ? "X" : "O"}` : "You");
  return PLAYERS[p.kind].label;
}

function showStatus(html) { $("#status").innerHTML = html; }

function updateStatus() {
  const s = ui.state;
  const dot = (side) => `<span class="dot ${side === X ? "x" : "o"}"></span>`;
  if (s.winner === DRAW) return showStatus("Draw.");
  if (s.winner) return showStatus(`${dot(s.winner)} ${sideName(s.winner)} ${sideName(s.winner) === "You" ? "win" : "wins"}!`);
  if (pending) {
    const what = ui.players[s.toMove].kind === "menace" && s.history.length < 2 ? "is training its matchboxes" : "is thinking";
    return showStatus(`${dot(s.toMove)} ${sideName(s.toMove)} ${what}… <span class="spinner" aria-hidden="true"></span>`);
  }
  if (isHuman(s.toMove)) {
    const verb = ui.cfg.gravity ? "drop a stone in a column" : "place a stone";
    return showStatus(`${dot(s.toMove)} ${sideName(s.toMove) === "You" ? "Your" : sideName(s.toMove) + "'s"} move: ${verb}.`);
  }
  return showStatus(`${dot(s.toMove)} ${sideName(s.toMove)} to move${ui.paused ? " (paused)" : ""}.`);
}

let thinkStart = 0, thinkTick = null;
function setThinking(side) {
  clearInterval(thinkTick);
  const t = $("#think");
  if (side == null) { t.textContent = ""; if (ui.state) renderBoard(); return; }
  thinkStart = performance.now();
  thinkTick = setInterval(() => { t.textContent = `${((performance.now() - thinkStart) / 1000).toFixed(1)} s`; }, 100);
  renderBoard();
  updateStatus();
}

function updateButtons() {
  const s = ui.state, ai = bothAI();
  $("#undo").disabled = !s.history.length || ai;
  $("#pause").hidden = !ai;
  $("#step").hidden = !ai;
  $("#pause").textContent = ui.paused ? "Resume" : "Pause";
  $("#step").disabled = !ui.paused || s.winner !== null || !!pending;
}

function renderLog() {
  const s = ui.state, g = s.game, log = $("#log");
  log.innerHTML = "";
  s.history.forEach((mv, i) => {
    const sp = el("span", null, `${i % 2 === 0 ? Math.floor(i / 2) + 1 + "." : ""}${g.cellName(mv)}`);
    sp.style.color = i % 2 === 0 ? "var(--x)" : "var(--o)";
    log.appendChild(sp);
  });
  if (!s.history.length) log.textContent = "No moves yet. Cells are named column letter + row number (a1 is top left).";
  log.scrollTop = log.scrollHeight;
}

function renderScore() {
  $("#score-x").textContent = ui.score[X];
  $("#score-d").textContent = ui.score[DRAW];
  $("#score-o").textContent = ui.score[O];
}

function renderInfo() {
  const box = $("#info"), li = ui.lastInfo;
  if (!li) { box.innerHTML = '<span class="muted">After an agent moves, what it saw (nodes searched, playouts, verdict) shows up here.</span>'; return; }
  const { info, ms, kind } = li, parts = [];
  parts.push(info.trainMs ? `<b>${PLAYERS[kind].label}</b> trained for ${fmtTime(info.trainMs / 1000)}, then took ${fmtTime((ms - info.trainMs) / 1000)} to move`
    : `<b>${PLAYERS[kind].label}</b> took ${fmtTime(ms / 1000)}`);
  if (info.verdict) parts.push(`verdict for the side to move: <b>${info.verdict}</b> with perfect play`);
  if (info.nodes) parts.push(`${fmtNum(info.nodes)} nodes searched`);
  if (info.depth) parts.push(`depth ${info.depth}`);
  if (info.playouts) parts.push(`${fmtNum(info.playouts)} playouts`);
  if (info.winRate != null) parts.push(`estimated win rate ${(info.winRate * 100).toFixed(0)}%`);
  if (info.rule) parts.push(`rule: ${info.rule}`);
  if (info.trained != null) {
    parts.push(`${fmtNum(info.boxes)} matchboxes after ${fmtNum(info.trained)} training games (teacher: ${info.teacher === "minimax" ? "random + perfect play" : "random + rule-based"})`);
    parts.push(info.unseen ? "no box for this position" : "drew from this position's matchbox");
    if (info.trainNote) parts.push(info.trainNote);
  }
  const big = info.refused || info.aborted || info.unseen;
  if (info.note && !big) parts.push(info.note);
  let html = parts.join(" · ");
  if (big) html = `<div class="note warn">${info.note}</div>` + html;
  box.innerHTML = html;
}

function renderFeasibility() {
  const g = getGame(ui.cfg), est = minimaxEstimate(g), pos = positionBound(g);
  const solvable = est.seconds <= 5;
  const usesMinimax = ui.players[X].kind === "minimax" || ui.players[O].kind === "minimax";
  $("#feas").innerHTML = `
    <div><b>${g.label}</b>: ${g.size} cells, up to ${fmtNum(pos)} positions.</div>
    <div>Solving it with minimax: ~${fmtNum(est.nodes)} nodes, ${fmtTime(est.seconds)} in this browser.</div>
    <div class="note ${solvable ? "" : usesMinimax ? "warn" : ""}" style="margin-top:8px">${solvable
      ? "Small enough to solve exactly: minimax plays perfectly here."
      : `Too big to solve: minimax refuses this board and plays depth-limited alpha-beta instead. ` +
        `MENACE would need up to ${fmtNum(pos / (g.gravity ? 2 : g.m === g.n ? 8 : 4))} matchboxes.`}</div>`;
}

// ---------------------------------------------------------------- controls
function buildPresets() {
  const wrap = $("#presets");
  PRESETS.forEach((p) => {
    const btn = el("button", "btn", p.name);
    btn.type = "button";
    btn.dataset.id = p.id;
    btn.onclick = () => { ui.cfg = { gravity: false, exact: false, ...p.cfg }; syncCustom(); newGame(); };
    wrap.appendChild(btn);
  });
}

function syncPresetButtons() {
  const c = ui.cfg;
  document.querySelectorAll("#presets button").forEach((b) => {
    const p = PRESETS.find((x) => x.id === b.dataset.id).cfg;
    b.setAttribute("aria-pressed", String(p.m === c.m && p.n === c.n && p.k === c.k && !!p.gravity === !!c.gravity && !!p.exact === !!c.exact));
  });
}

function syncCustom() {
  $("#m").value = ui.cfg.m; $("#n").value = ui.cfg.n; $("#k").value = ui.cfg.k;
  $("#gravity").checked = !!ui.cfg.gravity; $("#exact").checked = !!ui.cfg.exact;
  syncPresetButtons();
}

function readCustom() {
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, Math.round(+v || lo)));
  const m = clamp($("#m").value, 2, 19), n = clamp($("#n").value, 2, 19);
  const k = clamp($("#k").value, 2, Math.max(m, n));
  ui.cfg = { m, n, k, gravity: $("#gravity").checked, exact: $("#exact").checked };
  syncCustom();
  newGame();
}

function buildPlayerSelect(side) {
  const kindSel = $(`#p${side}-kind`), levelSel = $(`#p${side}-level`);
  fillPlayerSelect(kindSel);
  const syncLevel = () => {
    const p = ui.players[side];
    kindSel.value = p.kind;
    p.level = fillLevelSelect(levelSel, p.kind, p.level);
    const style = STYLES.find(([k]) => k === PLAYERS[p.kind].style)[1];
    $(`#p${side}-blurb`).innerHTML = p.kind === "human" ? "Click the board to move."
      : `<span class="pill">${style}</span> ${AGENTS[p.kind].blurb}`;
  };
  kindSel.onchange = () => { ui.players[side] = { kind: kindSel.value, level: PLAYERS[kindSel.value].def ?? null }; syncLevel(); playersChanged(); };
  levelSel.onchange = () => { ui.players[side].level = levelSel.value; playersChanged(); };
  return syncLevel;
}

function playersChanged() {
  writeHash();
  renderFeasibility();
  // If the side to move just became an agent (or stopped being one), restart the turn.
  cancelThinking();
  clearTimeout(thinkTimer);
  if (bothAI()) ui.paused = false;
  renderBoard();
  step();
}

function init() {
  readHash();
  buildPresets();
  const syncX = buildPlayerSelect(X), syncO = buildPlayerSelect(O);
  syncX(); syncO();
  syncCustom();

  boardEl().addEventListener("click", (e) => {
    const c = e.target.closest(".cell");
    if (!c || c.disabled) return;
    const s = ui.state, g = s.game;
    let mv = +c.dataset.cell;
    if (g.gravity) mv = s.colTop[mv % g.n] * g.n + (mv % g.n);
    if (s.winner === null && isHuman(s.toMove) && !pending) play(mv);
  });
  $("#new").onclick = newGame;
  $("#undo").onclick = undo;
  $("#swap").onclick = () => {
    [ui.players[X], ui.players[O]] = [ui.players[O], ui.players[X]];
    ui.score = { [X]: ui.score[O], [O]: ui.score[X], [DRAW]: ui.score[DRAW] };
    syncX(); syncO(); renderScore(); newGame();
  };
  $("#pause").onclick = () => { ui.paused = !ui.paused; if (!ui.paused) step(); else { clearTimeout(thinkTimer); updateStatus(); updateButtons(); } };
  $("#step").onclick = () => { if (!pending && ui.state.winner === null) requestMove(); };
  $("#apply").onclick = readCustom;
  $("#delay").oninput = (e) => { ui.delay = +e.target.value; };
  $("#reset-score").onclick = () => { ui.score = { [X]: 0, [O]: 0, [DRAW]: 0 }; renderScore(); };
  window.addEventListener("hashchange", () => { readHash(); syncX(); syncO(); syncCustom(); newGame(); });
  renderScore();
  newGame();
}

init();

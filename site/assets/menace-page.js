// MENACE page: train a matchbox learner live, watch its beads, then play it.
import { getGame, X, O, DRAW, positionBound } from "./mnk.js";
import { AGENTS, makeRng, fmtNum } from "./agents.js";
import { Menace } from "./menace.js";

const BOARDS = {
  "3,3,3": { m: 3, n: 3, k: 3 },
  "3,4,3": { m: 3, n: 4, k: 3 },
  "4,4,4": { m: 4, n: 4, k: 4 },
};
const OPPONENTS = {
  random: "Random",
  minimax: "Perfect play (minimax)",
  mixed: "Mix: random half the time, perfect half",
  rule: "Rule-based",
};
const WINDOW = 200, SAMPLE = 25;

const $ = (s) => document.querySelector(s);
const rng = makeRng(Date.now() & 0xffff);
const S = {
  board: "3,3,3", opponent: "mixed", side: X, menace: null, games: 0,
  outcomes: [], curve: [], running: false, raf: 0, watching: null, human: null,
};

function cfg() { return BOARDS[S.board]; }

function reset() {
  stop();
  S.menace = new Menace(getGame(cfg()), { rng: () => rng() });
  S.games = 0; S.outcomes = []; S.curve = [];
  S.watching = null; S.human = null;
  showPosition(getGame(cfg()).newState(), "The empty board. MENACE has no matchbox for it yet: press Train.");
  renderStats(); drawChart();
}

function opponentMove(state) {
  const kind = S.opponent === "mixed" ? (rng() < 0.5 ? "random" : "minimax") : S.opponent;
  return AGENTS[kind].select(state, { budget: 5 }, rng).move;
}

/** One complete training game. Returns the finished state. */
function trainingGame() {
  const s = getGame(cfg()).newState();
  while (s.winner === null) s.push(s.toMove === S.side ? S.menace.select(s).move : opponentMove(s));
  record(s);
  return s;
}

function record(s) {
  S.menace.learn(s, S.side);
  S.games++;
  S.outcomes.push(s.winner === DRAW ? 0 : s.winner === S.side ? 1 : -1);
  if (S.outcomes.length > WINDOW) S.outcomes.shift();
  if (S.games % SAMPLE === 0) S.curve.push([S.games, ...rates()]);
}

function rates() {
  const n = S.outcomes.length || 1;
  let w = 0, d = 0, l = 0;
  for (const o of S.outcomes) { if (o > 0) w++; else if (o === 0) d++; else l++; }
  return [w / n, d / n, l / n];
}

// ---------------------------------------------------------------- training loop
function loop() {
  // 1, 10 or 100 games per frame (about 60, 600 or 6,000 games a second), capped at 12 ms of work.
  const t0 = performance.now(), perFrame = [1, 10, 100][+$("#speed").value];
  let last = null;
  for (let i = 0; i < perFrame && (i === 0 || performance.now() - t0 < 12); i++) last = trainingGame();
  if (last) showPosition(getGame(cfg()).newState(), "The opening matchbox: beads per first move (more beads = likelier).");
  renderStats(); drawChart();
  if (S.running) S.raf = requestAnimationFrame(loop);
}

function start() {
  if (S.running) return;
  S.human = null; S.watching = null;
  S.running = true;
  $("#train").textContent = "Pause";
  S.raf = requestAnimationFrame(loop);
}
function stop() {
  S.running = false;
  cancelAnimationFrame(S.raf);
  $("#train").textContent = S.games ? "Resume training" : "Train";
}

// ---------------------------------------------------------------- watch one game slowly
async function watch() {
  stop();
  S.human = null;
  const token = {};
  S.watching = token;
  const s = getGame(cfg()).newState();
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  while (s.winner === null && S.watching === token) {
    if (s.toMove === S.side) {
      const { move, box } = S.menace.select(s);
      showPosition(s, box ? "MENACE opens the matchbox for this position and draws a bead…" : "Only one distinct move here: no matchbox needed.", move);
      await sleep(900);
      if (S.watching !== token) return;
      s.push(move);
    } else {
      s.push(opponentMove(s));
    }
    showPosition(s, "");
    await sleep(450);
  }
  if (S.watching !== token) return;
  record(s);
  const res = s.winner === DRAW ? "a draw (+1 bead for each move it made)" : s.winner === S.side ? "a win (+3 beads for each move it made)" : "a loss (−1 bead for each move it made)";
  showPosition(s, `Game over: ${res}.`);
  S.watching = null;
  renderStats(); drawChart();
}

// ---------------------------------------------------------------- play against it
function playHuman() {
  stop();
  S.watching = null;
  const s = getGame(cfg()).newState();
  S.human = s;
  humanStep();
}

function humanStep() {
  const s = S.human;
  if (!s) return;
  if (s.winner !== null) {
    record(s);
    const msg = s.winner === DRAW ? "Draw. MENACE adds a bead to each move it made." :
      s.winner === S.side ? "MENACE wins and adds 3 beads to each move it made." : "You win! MENACE takes a bead away from each move it made.";
    showPosition(s, msg);
    S.human = null;
    renderStats(); drawChart();
    return;
  }
  if (s.toMove === S.side) {
    const { move } = S.menace.select(s);
    setTimeout(() => { if (S.human === s) { s.push(move); humanStep(); } }, 350);
    showPosition(s, "MENACE is drawing a bead…");
  } else {
    showPosition(s, "Your move. MENACE learns from this game too.", null, true);
  }
}

// ---------------------------------------------------------------- rendering
function showPosition(s, caption, chosen = null, clickable = false) {
  const g = s.game, b = $("#mboard");
  b.style.setProperty("--cells", g.n);
  b.style.setProperty("--aspect", `${g.n} / ${g.m}`);
  b.style.setProperty("--board-max", `${g.n * 92}px`);
  if (b.children.length !== g.size) {
    b.innerHTML = "";
    for (let i = 0; i < g.size; i++) {
      const c = document.createElement("button");
      c.type = "button"; c.className = "cell"; c.dataset.cell = i;
      b.appendChild(c);
    }
  }
  const beads = s.winner === null && s.toMove === S.side ? S.menace.beadsFor(s) : null;
  const max = beads ? Math.max(1, ...beads.values()) : 1;
  const line = new Set(s.winningLine());
  [...b.children].forEach((c, i) => {
    c.innerHTML = "";
    c.classList.toggle("win", line.has(i));
    c.classList.toggle("playable", clickable && s.isLegal(i));
    c.disabled = !(clickable && s.isLegal(i));
    if (s.cells[i]) {
      const st = document.createElement("span");
      st.className = "stone " + (s.cells[i] === X ? "x" : "o");
      c.appendChild(st);
    } else if (beads && beads.has(i)) {
      const n = beads.get(i), heat = document.createElement("span");
      heat.className = "heat";
      heat.style.background = `color-mix(in srgb, var(--${S.side === X ? "x" : "o"}) ${Math.round(8 + 50 * n / max)}%, transparent)`;
      if (i === chosen) heat.style.outline = "3px solid var(--text)";
      const lab = document.createElement("span");
      lab.className = "beads"; lab.textContent = n;
      c.append(heat, lab);
    }
  });
  $("#caption").textContent = caption;
}

function renderStats() {
  const [w, d, l] = rates(), g = getGame(cfg());
  $("#st-games").textContent = fmtNum(S.games);
  $("#st-boxes").textContent = fmtNum(S.menace.boxes.size);
  $("#st-w").textContent = S.outcomes.length ? Math.round(w * 100) + "%" : "–";
  $("#st-d").textContent = S.outcomes.length ? Math.round(d * 100) + "%" : "–";
  $("#st-l").textContent = S.outcomes.length ? Math.round(l * 100) + "%" : "–";
  const sym = g.gravity ? 2 : g.m === g.n ? 8 : 4;
  $("#boxes-note").textContent = S.board === "3,3,3"
    ? "Michie's machine needed 304 for X with symmetries (this demo reproduces that count exactly)."
    : `Up to ~${fmtNum(positionBound(g) / sym)} positions after symmetry on this board.`;
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawChart() {
  const cv = $("#chart"), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);
  const pad = { l: 40, r: 10, t: 10, b: 26 }, pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--line"); ctx.lineWidth = 1;
  for (const f of [0, .25, .5, .75, 1]) {
    const y = pad.t + ph * (1 - f);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(pad.l + pw, y); ctx.stroke();
    ctx.textAlign = "right"; ctx.fillText(`${f * 100}%`, pad.l - 6, y + 4);
  }
  const pts = S.curve;
  const maxG = Math.max(500, pts.length ? pts[pts.length - 1][0] : 0);
  ctx.textAlign = "center";
  for (let i = 0; i <= 4; i++) ctx.textAlign = i === 4 ? "right" : i === 0 ? "left" : "center", ctx.fillText(fmtNum(Math.round(maxG * i / 4)), pad.l + pw * i / 4, H - 8);
  if (pts.length < 2) {
    ctx.fillText(`Rolling outcome rates (last ${WINDOW} games) appear here once training starts.`, pad.l + pw / 2, pad.t + ph / 2);
    return;
  }
  const x = (gm) => pad.l + pw * gm / maxG, y = (v) => pad.t + ph * (1 - v);
  // stacked areas: loss at the bottom, then draw, then win
  const layers = [[3, css("--refused")], [2, css("--unavailable")], [1, css("--ok")]];
  let base = pts.map(() => 0);
  for (const [idx, color] of layers) {
    const top = pts.map((p, i) => base[i] + p[idx]);
    ctx.beginPath();
    pts.forEach((p, i) => (i ? ctx.lineTo : ctx.moveTo).call(ctx, x(p[0]), y(top[i])));
    for (let i = pts.length - 1; i >= 0; i--) ctx.lineTo(x(pts[i][0]), y(base[i]));
    ctx.closePath();
    ctx.fillStyle = color; ctx.globalAlpha = .85; ctx.fill(); ctx.globalAlpha = 1;
    base = top;
  }
}

// ---------------------------------------------------------------- init
function init() {
  const bs = $("#board-select"), os = $("#opp-select");
  for (const k of Object.keys(BOARDS)) bs.appendChild(new Option(`(${k})${k === "3,3,3" ? " tic-tac-toe" : ""}`, k));
  for (const [k, v] of Object.entries(OPPONENTS)) os.appendChild(new Option(v, k));
  bs.value = S.board; os.value = S.opponent;
  bs.onchange = () => { S.board = bs.value; reset(); };
  os.onchange = () => { S.opponent = os.value; };
  $("#side-select").onchange = (e) => { S.side = +e.target.value; reset(); };
  $("#train").onclick = () => (S.running ? stop() : start());
  $("#reset").onclick = reset;
  $("#watch").onclick = watch;
  $("#play").onclick = playHuman;
  $("#mboard").addEventListener("click", (e) => {
    const c = e.target.closest(".cell"), s = S.human;
    if (!c || !s || s.toMove === S.side || !s.isLegal(+c.dataset.cell)) return;
    s.push(+c.dataset.cell);
    humanStep();
  });
  window.addEventListener("resize", drawChart);
  matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", drawChart);
  reset();
}

init();

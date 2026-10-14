// Agents for the browser, ported from src/mnkbench/agents/*.py.
// Every agent: select(state, params, rng) -> { move, info } where info has
// human-readable stats (nodes searched, playouts, depth, notes).

import { DRAW, EMPTY, positionBound } from "./mnk.js";

export class BudgetExceeded extends Error {}

// Seedable PRNG (mulberry32) so games can be replayed.
export function makeRng(seed = Date.now()) {
  let a = seed >>> 0;
  const rng = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  rng.choice = (arr) => arr[Math.floor(rng() * arr.length)];
  return rng;
}

// ------------------------------------------------------------------ random
function randomAgent(state, params, rng) {
  return { move: rng.choice(state.legalMoves()), info: {} };
}

// -------------------------------------------------------------- rule-based
function threatCells(state, cell, p) {
  const g = state.game, cells = state.cells, found = new Set();
  cells[cell] = p;
  for (const wi of g.cellWindows[cell]) {
    let own = 0, empties = 0, gap = -1, blocked = false;
    for (const c of g.windows[wi]) {
      const v = cells[c];
      if (v === p) own++;
      else if (v === EMPTY) { empties++; gap = c; } else { blocked = true; break; }
    }
    if (!blocked && own === g.k - 1 && empties === 1 && !found.has(gap) && state.wouldWin(gap, p)) {
      if (!g.gravity) found.add(gap);
      else {
        const col = gap % g.n;
        const top = state.colTop[col] - (cell % g.n === col ? 1 : 0);
        if (Math.floor(gap / g.n) === top) found.add(gap);
      }
    }
  }
  cells[cell] = EMPTY;
  return found;
}

function linePotential(state, cell, me) {
  const g = state.game, cells = state.cells;
  let score = 0;
  for (const wi of g.cellWindows[cell]) {
    let own = 0, opp = 0;
    for (const c of g.windows[wi]) {
      const v = cells[c];
      if (v === me) own++;
      else if (v !== EMPTY) opp++;
    }
    if (opp === 0) score += 1 + 4 ** own;
    else if (own === 0) score += 0.5 + 0.8 * 4 ** opp;
  }
  return score;
}

function nearCandidates(state, radius = 2) {
  const g = state.game, moves = state.legalMoves();
  if (g.gravity || g.size <= 25 || !state.history.length) return moves;
  const near = g.neighbors[radius - 1], set = new Set();
  for (const mv of state.history) for (const c of near[mv]) if (state.cells[c] === EMPTY) set.add(c);
  return set.size ? [...set].sort((a, b) => a - b) : moves;
}

function bestByPotential(state, moves, me, rng) {
  let best = -Infinity, out = [];
  for (const mv of moves) {
    const s = linePotential(state, mv, me);
    if (s > best + 1e-9) { best = s; out = [mv]; } else if (Math.abs(s - best) <= 1e-9) out.push(mv);
  }
  return rng.choice(out);
}

function ruleAgent(state, params, rng) {
  const me = state.toMove, opp = 3 - me, moves = state.legalMoves();
  const wins = moves.filter((mv) => state.wouldWin(mv, me));
  if (wins.length) return { move: rng.choice(wins), info: { rule: "win" } };
  const blocks = moves.filter((mv) => state.wouldWin(mv, opp));
  if (blocks.length) return { move: rng.choice(blocks), info: { rule: "block" } };
  const cands = nearCandidates(state);
  if (state.game.k >= 2) {
    const forks = cands.filter((mv) => threatCells(state, mv, me).size >= 2);
    if (forks.length) return { move: bestByPotential(state, forks, me, rng), info: { rule: "fork" } };
    const oppForks = cands.filter((mv) => threatCells(state, mv, opp).size >= 2);
    if (oppForks.length === 1) return { move: oppForks[0], info: { rule: "block fork" } };
    if (oppForks.length) {
      const forkSet = new Set(oppForks);
      const safe = cands.filter((mv) => {
        const t = threatCells(state, mv, me);
        return t.size && ![...t].some((c) => forkSet.has(c));
      });
      if (safe.length) return { move: bestByPotential(state, safe, me, rng), info: { rule: "force" } };
      return { move: bestByPotential(state, oppForks, me, rng), info: { rule: "block fork" } };
    }
  }
  return { move: bestByPotential(state, cands, me, rng), info: { rule: "line potential" } };
}

// ----------------------------------------------------------------- minimax
// Negamax + alpha-beta + transposition table. Score = 1 + empties left after
// a win (faster wins score higher); draw = 0.
const EXACT = 0, LOWER = 1, UPPER = 2;
export const NODE_EXPONENT = 0.68;
export const SECONDS_PER_NODE = 3e-6; // measured in V8: 2.6-3.3 us/node on (4,4,4), (4,5,4), (5,4,4)+gravity

export function minimaxEstimate(game) {
  const nodes = Math.max(10, positionBound(game) ** NODE_EXPONENT);
  return { nodes, seconds: nodes * SECONDS_PER_NODE * (1 + game.size / 40), memory: nodes * 120 };
}

class Solver {
  constructor(game, maxNodes = Infinity, deadline = Infinity) {
    this.game = game;
    this.tt = new Map();
    this.nodes = 0;
    this.maxNodes = maxNodes;
    this.deadline = deadline;
  }

  search(state, alpha, beta) {
    this.nodes++;
    if ((this.nodes & 4095) === 0) {
      if (this.nodes > this.maxNodes) throw new BudgetExceeded(`more than ${this.maxNodes} nodes`);
      if (performance.now() > this.deadline) throw new BudgetExceeded("out of time");
    }
    const empty = state.emptyCount, me = state.toMove, opp = 3 - me;
    let moves = state.legalMoves();
    for (const mv of moves) if (state.wouldWin(mv, me)) return empty;
    const threats = moves.filter((mv) => state.wouldWin(mv, opp));
    if (threats.length >= 2) return -(empty - 1);
    const key = state.key() + me;
    const e = this.tt.get(key);
    if (e) {
      if (e[1] === EXACT) return e[0];
      if (e[1] === LOWER && e[0] >= beta) return e[0];
      if (e[1] === UPPER && e[0] <= alpha) return e[0];
    }
    const hi = empty - 2, lo = -(empty - 1);
    if (beta > hi) beta = hi;
    if (alpha < lo) alpha = lo;
    if (alpha >= beta) return alpha;
    if (threats.length) moves = threats;
    else { const rank = this.game.rank; moves.sort((a, b) => rank[a] - rank[b]); }
    const a0 = alpha;
    let best = -Infinity;
    for (const mv of moves) {
      state.push(mv);
      const v = state.winner === DRAW ? 0 : -this.search(state, -beta, -alpha);
      state.pop();
      if (v > best) {
        best = v;
        if (v > alpha) { alpha = v; if (alpha >= beta) break; }
      }
    }
    this.tt.set(key, [best, best <= a0 ? UPPER : best >= beta ? LOWER : EXACT]);
    return best;
  }

  value(state) {
    if (state.winner !== null) return state.winner === DRAW ? 0 : -(state.emptyCount + 1);
    return this.search(state.copy(), -Infinity, Infinity);
  }

  bestMoves(state) {
    const v = this.value(state), work = state.copy(), best = [];
    for (const mv of state.legalMoves()) {
      work.push(mv);
      let ok;
      if (work.winner === DRAW) ok = v === 0;
      else if (work.winner !== null) ok = work.emptyCount + 1 >= v;
      else ok = -this.search(work, -v, -v + 1) >= v;
      work.pop();
      if (ok) best.push(mv);
    }
    return { value: v, moves: best };
  }
}

const solvers = new Map();
function minimaxAgent(state, params, rng) {
  const g = state.game, est = minimaxEstimate(g), budget = params.budget ?? 5;
  if (est.seconds > budget) {
    const r = alphabetaAgent(state, { depth: 4, timeLimit: Math.min(2, budget) }, rng);
    r.info.note = `Minimax refused ${g.label}: solving it needs ~${fmtNum(est.nodes)} nodes ` +
      `(~${fmtTime(est.seconds)}). Degraded to depth-limited alpha-beta.`;
    r.info.refused = true;
    return r;
  }
  if (!solvers.has(g.label)) solvers.set(g.label, new Solver(g));
  const s = solvers.get(g.label);
  const before = s.nodes;
  s.deadline = performance.now() + budget * 1000 * 2; // the estimate can be ~10x low on hard draws
  try {
    const { value, moves } = s.bestMoves(state);
    const verdict = value > 0 ? "win" : value < 0 ? "loss" : "draw";
    return { move: rng.choice(moves), info: { nodes: s.nodes - before, verdict, tt: s.tt.size } };
  } catch (err) {
    if (!(err instanceof BudgetExceeded)) throw err;
    solvers.delete(g.label);
    const r = alphabetaAgent(state, { depth: 4, timeLimit: 1 }, rng);
    r.info.note = `Minimax aborted (${err.message}); fell back to alpha-beta for this move.`;
    r.info.aborted = true;
    return r;
  }
}

// --------------------------------------------------------------- alphabeta
// Depth-limited alpha-beta with an incremental threat evaluation and tracked
// "hot" windows (k-1 stones + 1 empty = an immediate threat).
const WIN = 1e9;

class Board {
  constructor(state, base = 8, radius = 2) {
    const g = state.game;
    this.g = g;
    this.st = state.copy();
    this.w = [0];
    for (let i = 1; i <= g.k; i++) this.w.push(base ** i);
    this.wx = new Int16Array(g.windows.length);
    this.wo = new Int16Array(g.windows.length);
    this.sx = 0; this.so = 0;
    this.near = new Int16Array(g.size);
    this.neigh = g.neighbors[radius <= 1 ? 0 : 1];
    this.hot = { 1: new Set(), 2: new Set() };
    const cells = this.st.cells;
    g.windows.forEach((win, wi) => {
      for (const c of win) { if (cells[c] === 1) this.wx[wi]++; else if (cells[c] === 2) this.wo[wi]++; }
      const x = this.wx[wi], o = this.wo[wi];
      if (o === 0) this.sx += this.w[x]; else if (x === 0) this.so += this.w[o];
      this.updHot(wi);
    });
    for (const mv of this.st.history) for (const c of this.neigh[mv]) this.near[c]++;
  }

  updHot(wi) {
    const x = this.wx[wi], o = this.wo[wi], km1 = this.g.k - 1;
    if (o === 0 && x === km1) this.hot[1].add(wi); else this.hot[1].delete(wi);
    if (x === 0 && o === km1) this.hot[2].add(wi); else this.hot[2].delete(wi);
  }

  push(mv) {
    const p = this.st.toMove, own = p === 1 ? this.wx : this.wo, opp = p === 1 ? this.wo : this.wx, w = this.w;
    let gain = 0, loss = 0;
    for (const wi of this.g.cellWindows[mv]) {
      const o = opp[wi];
      if (o === 0) gain += w[own[wi] + 1] - w[own[wi]];
      else if (own[wi] === 0) loss += w[o];
      own[wi]++;
      this.updHot(wi);
    }
    if (p === 1) { this.sx += gain; this.so -= loss; } else { this.so += gain; this.sx -= loss; }
    for (const c of this.neigh[mv]) this.near[c]++;
    this.st.push(mv);
  }

  pop() {
    const mv = this.st.pop(), p = this.st.toMove;
    const own = p === 1 ? this.wx : this.wo, opp = p === 1 ? this.wo : this.wx, w = this.w;
    let gain = 0, loss = 0;
    for (const wi of this.g.cellWindows[mv]) {
      own[wi]--;
      this.updHot(wi);
      const o = opp[wi];
      if (o === 0) gain += w[own[wi] + 1] - w[own[wi]];
      else if (own[wi] === 0) loss += w[o];
    }
    if (p === 1) { this.sx -= gain; this.so += loss; } else { this.so -= gain; this.sx += loss; }
    for (const c of this.neigh[mv]) this.near[c]--;
  }

  evaluate(tempo) {
    return this.st.toMove === 1 ? tempo * this.sx - this.so : tempo * this.so - this.sx;
  }

  winCells(p) {
    const out = [], st = this.st, check = this.g.exact || this.g.gravity;
    for (const wi of this.hot[p]) {
      for (const c of this.g.windows[wi]) {
        if (st.cells[c] === EMPTY) {
          if (!out.includes(c) && (!check || (st.isLegal(c) && st.wouldWin(c, p)))) out.push(c);
          break;
        }
      }
    }
    return out;
  }

  candidates() {
    const st = this.st;
    if (this.g.gravity || !st.history.length) return st.legalMoves();
    const out = [];
    for (let c = 0; c < this.g.size; c++) if (this.near[c] && st.cells[c] === EMPTY) out.push(c);
    return out;
  }

  moveScore(mv, p) {
    const own = p === 1 ? this.wx : this.wo, opp = p === 1 ? this.wo : this.wx, w = this.w;
    let s = 0;
    for (const wi of this.g.cellWindows[mv]) {
      if (opp[wi] === 0) s += w[own[wi] + 1];
      if (own[wi] === 0) s += 0.9 * w[opp[wi] + 1];
    }
    return s;
  }
}

class Timeout extends Error {}

function alphabetaAgent(state, params, rng) {
  const depthMax = params.depth ?? 4, maxCand = params.maxCandidates ?? 12, tempo = params.tempo ?? 2;
  const legal = state.legalMoves(), me = state.toMove, opp = 3 - me;
  if (legal.length === 1) return { move: legal[0], info: { depth: 0, nodes: 0 } };
  const wins = legal.filter((mv) => state.wouldWin(mv, me));
  if (wins.length) return { move: rng.choice(wins), info: { depth: 1, nodes: 0, note: "immediate win" } };
  const blocks = legal.filter((mv) => state.wouldWin(mv, opp));
  if (blocks.length === 1) return { move: blocks[0], info: { depth: 1, nodes: 0, note: "forced block" } };
  if (!state.history.length && !state.game.gravity) return { move: state.game.centerOrder[0], info: { depth: 0, nodes: 0 } };

  const b = new Board(state);
  const tt = new Map();
  let nodes = 0;
  const deadline = params.timeLimit ? performance.now() + params.timeLimit * 1000 : Infinity;
  const ordered = (moves, p, ttMove, limit) => {
    const sc = new Map(moves.map((mv) => [mv, b.moveScore(mv, p)]));
    let out = moves.slice().sort((x, y) => sc.get(y) - sc.get(x));
    if (ttMove !== undefined && out.includes(ttMove)) out = [ttMove, ...out.filter((m) => m !== ttMove)];
    return limit && out.length > maxCand ? out.slice(0, maxCand) : out;
  };
  const search = (depth, alpha, beta, ply) => {
    nodes++;
    if ((nodes & 255) === 0 && performance.now() > deadline) throw new Timeout();
    const st = b.st, p = st.toMove, o = 3 - p;
    if (b.winCells(p).length) return WIN - ply;
    const threats = b.winCells(o);
    if (threats.length >= 2) return -(WIN - ply - 1);
    if (depth <= 0) return b.evaluate(tempo);
    const key = st.key();
    const e = tt.get(key);
    let ttMove;
    if (e) {
      ttMove = e[3];
      if (e[0] >= depth) {
        if (e[2] === EXACT) return e[1];
        if (e[2] === LOWER && e[1] >= beta) return e[1];
        if (e[2] === UPPER && e[1] <= alpha) return e[1];
      }
    }
    const moves = threats.length ? threats : ordered(b.candidates(), p, ttMove, true);
    const a0 = alpha;
    let best = -2 * WIN, bestMv = moves[0];
    for (const mv of moves) {
      b.push(mv);
      const v = st.winner === DRAW ? 0 : -search(depth - 1, -beta, -alpha, ply + 1);
      b.pop();
      if (v > best) { best = v; bestMv = mv; if (v > alpha) { alpha = v; if (alpha >= beta) break; } }
    }
    tt.set(key, [depth, best, best <= a0 ? UPPER : best >= beta ? LOWER : EXACT, bestMv]);
    return best;
  };
  let order = ordered(blocks.length ? blocks : b.candidates().sort(() => rng() - 0.5), me, undefined, false);
  if (order.length > 2 * maxCand) order = order.slice(0, 2 * maxCand);
  let bestMv = order[0], reached = 0, bestVal = 0;
  try {
    for (let d = 1; d <= depthMax; d++) {
      let alpha = -2 * WIN, best = -2 * WIN, mvBest = order[0];
      const scores = [];
      for (const mv of order) {
        b.push(mv);
        let v;
        if (b.st.winner === DRAW) v = 0;
        else if (b.st.winner !== null) v = WIN;
        else v = -search(d - 1, -2 * WIN, -alpha, 1);
        b.pop();
        scores.push([v, mv]);
        if (v > best) { best = v; mvBest = mv; alpha = Math.max(alpha, v); }
      }
      bestMv = mvBest; reached = d; bestVal = best;
      if (Math.abs(best) >= WIN - 100) break;
      order = scores.sort((x, y) => y[0] - x[0]).map((s) => s[1]);
    }
  } catch (err) {
    if (!(err instanceof Timeout)) throw err;
  }
  const note = bestVal >= WIN - 100 ? "found a forced win" : bestVal <= -(WIN - 100) ? "sees a forced loss" : undefined;
  return { move: bestMv, info: { depth: reached, nodes, note } };
}

// -------------------------------------------------------------------- MCTS
export function randomPlayout(state, rng) {
  if (state.winner !== null) return state.winner;
  const g = state.game, cells = state.cells.slice(), rays = g.rays, k = g.k, exact = g.exact;
  let p = state.toMove;
  const wins = (mv) => {
    for (const [fwd, bwd] of rays[mv]) {
      let cnt = 1;
      for (const c of fwd) { if (cells[c] !== p) break; cnt++; }
      for (const c of bwd) { if (cells[c] !== p) break; cnt++; }
      if (cnt === k || (cnt > k && !exact)) return true;
    }
    return false;
  };
  if (g.gravity) {
    const tops = state.colTop.slice(), open = [];
    for (let c = 0; c < g.n; c++) if (tops[c] >= 0) open.push(c);
    while (open.length) {
      const i = Math.floor(rng() * open.length), col = open[i], mv = tops[col] * g.n + col;
      if (--tops[col] < 0) { open[i] = open[open.length - 1]; open.pop(); }
      cells[mv] = p;
      if (wins(mv)) return p;
      p = 3 - p;
    }
    return DRAW;
  }
  const empties = [];
  for (let i = 0; i < cells.length; i++) if (cells[i] === EMPTY) empties.push(i);
  let remaining = empties.length;
  while (remaining) {
    const i = Math.floor(rng() * remaining), mv = empties[i];
    empties[i] = empties[--remaining];
    cells[mv] = p;
    if (wins(mv)) return p;
    p = 3 - p;
  }
  return DRAW;
}

function mctsAgent(state, params, rng) {
  const legal = state.legalMoves(), me = state.toMove;
  if (legal.length === 1) return { move: legal[0], info: { playouts: 0 } };
  const wins = legal.filter((mv) => state.wouldWin(mv, me));
  if (wins.length) return { move: rng.choice(wins), info: { playouts: 0, note: "immediate win" } };
  const blocks = legal.filter((mv) => state.wouldWin(mv, 3 - me));
  if (blocks.length === 1) return { move: blocks[0], info: { playouts: 0, note: "forced block" } };
  const g = state.game;
  if (!state.history.length && !g.gravity && g.size > 49) return { move: g.centerOrder[0], info: { playouts: 0 } };
  const radius = !g.gravity && g.size > 49 ? 2 : 0;
  const movesOf = (s) => {
    const ms = s.legalMoves();
    if (!radius || !s.history.length) return ms;
    const set = new Set(), near = g.neighbors[1];
    for (const mv of s.history) for (const c of near[mv]) if (s.cells[c] === EMPTY) set.add(c);
    return set.size ? [...set] : ms;
  };
  const maxIters = params.iterations ?? 2000, c = params.c ?? 1.4;
  const deadline = performance.now() + (params.timeLimit ?? 5) * 1000;
  const root = { move: -1, parent: null, children: [], untried: movesOf(state), n: 0, w: 0, player: 3 - me };
  let it = 0;
  while (it < maxIters && (it === 0 || performance.now() < deadline || (it & 63) !== 0)) {
    if (it >= maxIters) break;
    it++;
    let node = root;
    const sim = state.copy();
    while (!node.untried.length && node.children.length) {
      const lnN = Math.log(node.n);
      let best = null, bv = -1;
      for (const ch of node.children) {
        const v = ch.w / ch.n + c * Math.sqrt(lnN / ch.n);
        if (v > bv) { bv = v; best = ch; }
      }
      node = best;
      sim.push(node.move);
    }
    if (node.untried.length && sim.winner === null) {
      const i = Math.floor(rng() * node.untried.length);
      const mv = node.untried[i];
      node.untried[i] = node.untried[node.untried.length - 1];
      node.untried.pop();
      const player = sim.toMove;
      sim.push(mv);
      const child = { move: mv, parent: node, children: [], untried: sim.winner === null ? movesOf(sim) : [], n: 0, w: 0, player };
      node.children.push(child);
      node = child;
    }
    const winner = sim.winner !== null ? sim.winner : randomPlayout(sim, rng);
    for (let nd = node; nd; nd = nd.parent) {
      nd.n++;
      if (winner === nd.player) nd.w += 1; else if (winner === DRAW) nd.w += 0.5;
    }
    if (performance.now() > deadline) break;
  }
  const best = root.children.reduce((a, b) => (b.n > a.n || (b.n === a.n && b.w > a.w) ? b : a));
  return { move: best.move, info: { playouts: it, winRate: best.w / best.n } };
}

// ------------------------------------------------------------------ registry
export const AGENTS = {
  random: { label: "Random", select: randomAgent, blurb: "Uniformly random legal moves: the floor." },
  rule: { label: "Rule-based", select: ruleAgent, blurb: "Win, block, make a fork, block a fork, else the cell with the best line potential." },
  minimax: { label: "Minimax (perfect)", select: minimaxAgent, blurb: "Full alpha-beta search to the end of the game. Perfect, but refuses boards it cannot solve in time." },
  alphabeta: { label: "Alpha-beta (depth-limited)", select: alphabetaAgent, blurb: "Looks a few moves ahead and scores leaves with a threat evaluation." },
  mcts: { label: "MCTS", select: mctsAgent, blurb: "Monte Carlo Tree Search with random playouts." },
};

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const pow10 = (e) => "×10" + String(e).split("").map((d) => SUP[+d]).join("");

export function fmtNum(x) {
  if (!isFinite(x)) return "∞";
  if (x < 1e4) return Math.round(x).toLocaleString("en-US");
  const e = Math.floor(Math.log10(x));
  if (e < 9) return (x / 1e6 >= 1 ? (x / 1e6).toFixed(1) + " million" : Math.round(x / 1e3) + " thousand");
  return (x / 10 ** e).toFixed(1) + pow10(e);
}

export function fmtTime(s) {
  if (!isFinite(s)) return "forever";
  if (s < 1e-3) return Math.max(1, Math.round(s * 1e6)) + " µs";
  if (s < 0.01) return (s * 1000).toFixed(1) + " ms";
  if (s < 1) return Math.round(s * 1000) + " ms";
  if (s < 120) return s.toFixed(1) + " s";
  if (s < 7200) return Math.round(s / 60) + " min";
  if (s < 86400 * 2) return Math.round(s / 3600) + " hours";
  if (s < 86400 * 365 * 2) return Math.round(s / 86400) + " days";
  const y = s / (86400 * 365);
  return y < 1e6 ? Math.round(y).toLocaleString("en-US") + " years" : (y / 10 ** Math.floor(Math.log10(y))).toFixed(1) + pow10(Math.floor(Math.log10(y))) + " years";
}

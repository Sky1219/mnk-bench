// MENACE in the browser: a port of src/mnkbench/agents/menace.py.
// One matchbox per (canonical) position, one bead colour per distinct move.
// Beads are drawn at random; after the game every move gets +3 beads for a
// win, +1 for a draw and -1 for a loss (Michie's 4/3/2/1 starting schedule).
import { DRAW } from "./mnk.js";

/** Board symmetries as cell permutations: perm[cell] = image of cell. */
export function symmetries(game) {
  const { m, n, gravity } = game, maps = [];
  const add = (f) => maps.push(Int16Array.from({ length: m * n }, (_, cell) => {
    const [r, c] = f(Math.floor(cell / n), cell % n);
    return r * n + c;
  }));
  add((r, c) => [r, c]);
  add((r, c) => [r, n - 1 - c]);
  if (!gravity) {
    add((r, c) => [m - 1 - r, c]);
    add((r, c) => [m - 1 - r, n - 1 - c]);
    if (m === n) {
      add((r, c) => [c, r]);
      add((r, c) => [c, n - 1 - r]);
      add((r, c) => [n - 1 - c, r]);
      add((r, c) => [n - 1 - c, n - 1 - r]);
    }
  }
  return maps;
}

export class Canonicalizer {
  constructor(game, useSymmetry = true) {
    this.perms = useSymmetry ? symmetries(game) : symmetries(game).slice(0, 1);
    this.inv = this.perms.map((p) => {
      const q = new Int16Array(p.length);
      p.forEach((to, from) => { q[to] = from; });
      return q;
    });
    this.buf = new Int8Array(game.size);
  }

  /** Smallest image of the position under the symmetries: [key, symIndex]. */
  canonical(cells) {
    let best = null, bestSym = 0;
    this.perms.forEach((p, s) => {
      const b = this.buf;
      for (let i = 0; i < cells.length; i++) b[p[i]] = cells[i];
      const key = String.fromCharCode.apply(null, b);
      if (best === null || key < best) { best = key; bestSym = s; }
    });
    return [best, bestSym];
  }

  toCanon(sym, mv) { return this.perms[sym][mv]; }
  fromCanon(sym, cm) { return this.inv[sym][cm]; }

  /** One representative per orbit of moves under the canonical position's own symmetries. */
  orbitReps(key, moves) {
    const stab = this.perms.filter((p) => {
      for (let i = 0; i < key.length; i++) if (key.charCodeAt(p[i]) !== key.charCodeAt(i)) return false;
      return true;
    });
    const reps = new Set();
    for (const mv of moves) reps.add(Math.min(...stab.map((p) => p[mv])));
    return [...reps].sort((a, b) => a - b);
  }
}

export class Menace {
  constructor(game, { useSymmetry = true, maxBeads = 4, win = 3, draw = 1, loss = 1, rng = Math.random } = {}) {
    Object.assign(this, { game, maxBeads, win, draw, loss, rng });
    this.canon = new Canonicalizer(game, useSymmetry);
    this.boxes = new Map(); // key -> Map(canonMove -> beads)
    this.trail = { 1: [], 2: [] };
    this.training = true;
  }

  initialBeads(stones) { return Math.max(1, this.maxBeads - (stones >> 1)); }

  /** Picks a move for state.toMove. Returns { move, box, canonMove, sym } (box is null for forced moves). */
  select(state) {
    const [key, sym] = this.canon.canonical(state.cells);
    let box = this.boxes.get(key);
    if (!box) {
      const moves = this.canon.orbitReps(key, state.legalMoves().map((mv) => this.canon.toCanon(sym, mv)));
      if (moves.length === 1) return { move: this.canon.fromCanon(sym, moves[0]), box: null };
      const beads = this.initialBeads(state.history.length);
      box = new Map(moves.map((mv) => [mv, beads]));
      if (this.training) this.boxes.set(key, box);
    }
    let total = 0;
    for (const b of box.values()) total += b;
    if (total <= 0) { // an empty box: refill it (Michie's machine would resign)
      const beads = this.initialBeads(state.history.length);
      for (const mv of box.keys()) box.set(mv, beads);
      total = beads * box.size;
    }
    let cm;
    if (this.training) {
      let pick = this.rng() * total;
      for (const [mv, b] of box) { cm = mv; pick -= b; if (pick < 0) break; }
      this.trail[state.toMove].push([key, cm]);
    } else {
      const top = Math.max(...box.values()), best = [...box].filter(([, b]) => b === top).map(([mv]) => mv);
      cm = best[Math.floor(this.rng() * best.length)];
    }
    return { move: this.canon.fromCanon(sym, cm), box, canonMove: cm, sym };
  }

  /** Reinforce the moves `player` made in the finished game. */
  learn(state, player) {
    const trail = this.trail[player];
    if (this.training && trail.length) {
      const delta = state.winner === DRAW ? this.draw : state.winner === player ? this.win : -this.loss;
      for (const [key, cm] of trail) {
        const box = this.boxes.get(key);
        box.set(cm, Math.max(0, box.get(cm) + delta));
      }
    }
    this.trail[player] = [];
  }

  /** Beads for each real cell of the given position (null if MENACE has no box for it). */
  beadsFor(state) {
    const [key, sym] = this.canon.canonical(state.cells);
    const box = this.boxes.get(key);
    if (!box) return null;
    const stab = this.canon.perms.filter((p) => {
      for (let i = 0; i < key.length; i++) if (key.charCodeAt(p[i]) !== key.charCodeAt(i)) return false;
      return true;
    });
    const out = new Map();
    for (const mv of state.legalMoves()) {
      const cm = this.canon.toCanon(sym, mv), rep = Math.min(...stab.map((p) => p[cm]));
      if (box.has(rep)) out.set(mv, box.get(rep));
    }
    return out;
  }
}

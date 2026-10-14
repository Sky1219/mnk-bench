// mnk-bench in the browser: a JavaScript port of the m,n,k game engine.
// Mirrors src/mnkbench/game.py: cells are row-major indices (r * n + c),
// X = 1 moves first, O = 2; State.winner is null while running, 1/2 for a win, 0 for a draw.

export const EMPTY = 0, X = 1, O = 2, DRAW = 0;
const AXES = [[0, 1], [1, 0], [1, 1], [1, -1]];
export const LETTERS = "abcdefghijklmnopqrstuvwxyz";

export class Game {
  constructor(m, n, k, gravity = false, exact = false) {
    Object.assign(this, { m, n, k, gravity, exact });
    this.size = m * n;
    // rays[cell] = [[fwd cells], [bwd cells]] for each of the 4 axes (up to k long)
    this.rays = [];
    for (let cell = 0; cell < this.size; cell++) {
      const r = Math.floor(cell / n), c = cell % n, perAxis = [];
      for (const [dr, dc] of AXES) {
        const fwd = [], bwd = [];
        for (let s = 1; s <= k; s++) {
          const rr = r + dr * s, cc = c + dc * s;
          if (rr < 0 || rr >= m || cc < 0 || cc >= n) break;
          fwd.push(rr * n + cc);
        }
        for (let s = 1; s <= k; s++) {
          const rr = r - dr * s, cc = c - dc * s;
          if (rr < 0 || rr >= m || cc < 0 || cc >= n) break;
          bwd.push(rr * n + cc);
        }
        perAxis.push([fwd, bwd]);
      }
      this.rays.push(perAxis);
    }
    // every k-cell window, and the windows through each cell
    this.windows = [];
    for (let r = 0; r < m; r++) for (let c = 0; c < n; c++) for (const [dr, dc] of AXES) {
      const er = r + dr * (k - 1), ec = c + dc * (k - 1);
      if (er >= 0 && er < m && ec >= 0 && ec < n) {
        const w = [];
        for (let i = 0; i < k; i++) w.push((r + dr * i) * n + (c + dc * i));
        this.windows.push(w);
      }
    }
    this.cellWindows = Array.from({ length: this.size }, () => []);
    this.windows.forEach((w, wi) => w.forEach((cell) => this.cellWindows[cell].push(wi)));
    this.neighbors = [1, 2].map((rad) => Array.from({ length: this.size }, (_, cell) => {
      const r = Math.floor(cell / n), c = cell % n, out = [];
      for (let dr = -rad; dr <= rad; dr++) for (let dc = -rad; dc <= rad; dc++) {
        if (!dr && !dc) continue;
        const rr = r + dr, cc = c + dc;
        if (rr >= 0 && rr < m && cc >= 0 && cc < n) out.push(rr * n + cc);
      }
      return out;
    }));
    const cr = (m - 1) / 2, cc = (n - 1) / 2;
    this.centerOrder = [...Array(this.size).keys()].sort((a, b) =>
      (Math.abs(Math.floor(a / n) - cr) + Math.abs((a % n) - cc)) -
      (Math.abs(Math.floor(b / n) - cr) + Math.abs((b % n) - cc)) || a - b);
    this.rank = new Array(this.size);
    this.centerOrder.forEach((cell, i) => { this.rank[cell] = gravity ? Math.abs((cell % n) - cc) : i; });
  }

  get label() {
    return `(${this.m},${this.n},${this.k})` + (this.gravity ? "+gravity" : "") + (this.exact ? "+exact" : "");
  }

  cellName(cell) { return LETTERS[cell % this.n] + (Math.floor(cell / this.n) + 1); }

  newState() { return new State(this); }
}

const games = new Map();
export function getGame({ m, n, k, gravity = false, exact = false }) {
  const key = `${m},${n},${k},${+gravity},${+exact}`;
  if (!games.has(key)) games.set(key, new Game(m, n, k, gravity, exact));
  return games.get(key);
}

export class State {
  constructor(game) {
    this.game = game;
    this.cells = new Int8Array(game.size);
    this.toMove = X;
    this.history = [];
    this.winner = null;
    this.colTop = new Int8Array(game.n).fill(game.m - 1);
  }

  copy() {
    const s = new State(this.game);
    s.cells.set(this.cells);
    s.toMove = this.toMove;
    s.history = this.history.slice();
    s.winner = this.winner;
    s.colTop.set(this.colTop);
    return s;
  }

  get emptyCount() { return this.game.size - this.history.length; }

  key() { return String.fromCharCode.apply(null, this.cells); }

  legalMoves() {
    if (this.winner !== null) return [];
    const g = this.game, out = [];
    if (g.gravity) {
      for (let c = 0; c < g.n; c++) if (this.colTop[c] >= 0) out.push(this.colTop[c] * g.n + c);
    } else {
      for (let i = 0; i < g.size; i++) if (this.cells[i] === EMPTY) out.push(i);
    }
    return out;
  }

  isLegal(mv) {
    if (this.winner !== null || mv < 0 || mv >= this.game.size || this.cells[mv] !== EMPTY) return false;
    return !this.game.gravity || Math.floor(mv / this.game.n) === this.colTop[mv % this.game.n];
  }

  wouldWin(cell, p) {
    const { k, exact, rays } = this.game, cells = this.cells;
    for (const [fwd, bwd] of rays[cell]) {
      let cnt = 1;
      for (const c of fwd) { if (cells[c] !== p) break; cnt++; }
      for (const c of bwd) { if (cells[c] !== p) break; cnt++; }
      if (cnt === k || (cnt > k && !exact)) return true;
    }
    return false;
  }

  winningMoves(p) { return this.legalMoves().filter((mv) => this.wouldWin(mv, p)); }

  push(mv) {
    const p = this.toMove, g = this.game;
    this.cells[mv] = p;
    this.history.push(mv);
    if (g.gravity) this.colTop[mv % g.n]--;
    if (this.wouldWin(mv, p)) this.winner = p;
    else if (this.history.length === g.size) this.winner = DRAW;
    this.toMove = 3 - p;
  }

  pop() {
    const mv = this.history.pop();
    this.cells[mv] = EMPTY;
    if (this.game.gravity) this.colTop[mv % this.game.n]++;
    this.winner = null;
    this.toMove = 3 - this.toMove;
    return mv;
  }

  /** Cells of the winning line through the last move (for highlighting). */
  winningLine() {
    if (!this.winner) return [];
    const mv = this.history[this.history.length - 1], p = this.cells[mv];
    for (const [fwd, bwd] of this.game.rays[mv]) {
      const line = [mv];
      for (const c of fwd) { if (this.cells[c] !== p) break; line.push(c); }
      for (const c of bwd) { if (this.cells[c] !== p) break; line.push(c); }
      if (line.length === this.game.k || (line.length > this.game.k && !this.game.exact)) return line;
    }
    return [];
  }
}

/** Upper bound on reachable positions (same as complexity.position_bound). */
export function positionBound(game) {
  const N = game.size;
  let free = 0, comb = 1; // comb = C(N, i)
  for (let i = 0; i <= N; i++) {
    if (i > 0) comb = comb * (N - i + 1) / i;
    let half = 1; // C(i, floor(i/2))
    const h = Math.floor(i / 2);
    for (let j = 1; j <= h; j++) half = half * (i - h + j) / j;
    free += comb * half;
  }
  if (game.gravity) return Math.min(free, Math.pow(2 ** (game.m + 1) - 1, game.n));
  return free;
}

// Board presets and player specs shared by the Play and Arena pages.

export const PRESETS = [
  { id: "ttt", name: "Tic-tac-toe", cfg: { m: 3, n: 3, k: 3 } },
  { id: "444", name: "4×4, 4 in a row", cfg: { m: 4, n: 4, k: 4 } },
  { id: "554", name: "5×5, 4", cfg: { m: 5, n: 5, k: 4 } },
  { id: "664", name: "6×6, 4", cfg: { m: 6, n: 6, k: 4 } },
  { id: "c4", name: "Connect Four", cfg: { m: 6, n: 7, k: 4, gravity: true } },
  { id: "775", name: "7×7, 5", cfg: { m: 7, n: 7, k: 5 } },
  { id: "gomoku", name: "Gomoku", cfg: { m: 15, n: 15, k: 5 } },
  { id: "gomoku-exact", name: "Gomoku (exact 5)", cfg: { m: 15, n: 15, k: 5, exact: true } },
  { id: "go", name: "19×19, 5", cfg: { m: 19, n: 19, k: 5 } },
];

/** Agent families, in the order the menus show them. */
export const STYLES = [
  ["you", "You"],
  ["baseline", "Baselines"],
  ["learning", "Learning: MENACE-style matchboxes"],
  ["exact", "Exact search"],
  ["heuristic", "Heuristic search"],
  ["sampling", "Sampling"],
];

export const PLAYERS = {
  human: { label: "Human", style: "you" },
  random: { label: "Random", style: "baseline" },
  rule: { label: "Rule-based", style: "baseline" },
  menace: {
    label: "MENACE (matchbox learner)", short: "MENACE", style: "learning",
    levels: [["300", "trained 300 games"], ["2000", "trained 2,000 games"], ["10000", "trained 10,000 games"]], def: "2000",
    params: (v) => ({ games: +v, trainSeconds: 8 }),
  },
  minimax: {
    label: "Minimax (perfect)", short: "Minimax", style: "exact",
    levels: [["2", "2 s budget"], ["5", "5 s budget"], ["15", "15 s budget"]], def: "5",
    params: (v) => ({ budget: +v }),
  },
  alphabeta: {
    label: "Alpha-beta", style: "heuristic",
    levels: [["1", "depth 1"], ["2", "depth 2"], ["3", "depth 3"], ["4", "depth 4"], ["5", "depth 5"], ["6", "depth 6"]], def: "4",
    params: (v) => ({ depth: +v, timeLimit: 8 }),
  },
  mcts: {
    label: "MCTS", style: "sampling",
    levels: [["200", "200 playouts"], ["1000", "1,000 playouts"], ["5000", "5,000 playouts"], ["t2", "2 s thinking"], ["t5", "5 s thinking"]], def: "1000",
    params: (v) => v[0] === "t" ? { iterations: 1e9, timeLimit: +v.slice(1) } : { iterations: +v, timeLimit: 10 },
  },
};

/** Fill a <select> with players grouped by style. */
export function fillPlayerSelect(sel, { human = true } = {}) {
  sel.innerHTML = "";
  for (const [style, title] of STYLES) {
    const kinds = Object.entries(PLAYERS).filter(([k, p]) => p.style === style && (human || k !== "human"));
    if (!kinds.length) continue;
    const group = document.createElement("optgroup");
    group.label = title;
    for (const [k, p] of kinds) group.appendChild(new Option(p.label, k));
    sel.appendChild(group);
  }
}

/** Fill a strength <select> for a player kind; returns the chosen level (or null). */
export function fillLevelSelect(sel, kind, level) {
  const spec = PLAYERS[kind];
  sel.innerHTML = "";
  sel.disabled = !spec.levels;
  if (!spec.levels) { sel.appendChild(new Option("—", "")); return null; }
  spec.levels.forEach(([v, l]) => sel.appendChild(new Option(l, v)));
  if (!spec.levels.some(([v]) => v === level)) level = spec.def;
  sel.value = level;
  return level;
}

export function playerParams(p) {
  const spec = PLAYERS[p.kind];
  return spec.params ? spec.params(p.level) : {};
}

/** Short display name such as "Alpha-beta d4" or "MENACE 2,000". */
export function playerName(p) {
  const spec = PLAYERS[p.kind], base = spec.short ?? spec.label;
  if (!spec.levels) return base;
  const lvl = spec.levels.find(([v]) => v === p.level)?.[1] ?? "";
  return `${base} (${lvl})`;
}

export function boardLabel(c) {
  return `(${c.m},${c.n},${c.k})${c.gravity ? "+gravity" : ""}${c.exact ? "+exact" : ""}`;
}

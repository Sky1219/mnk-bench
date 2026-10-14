import { test } from "node:test";
import assert from "node:assert/strict";
import { getGame, X, O, DRAW, positionBound } from "../assets/mnk.js";
import { AGENTS, makeRng, minimaxEstimate } from "../assets/agents.js";

function play(cfg, a, b, seed = 1, params = {}) {
  const s = getGame(cfg).newState(), rng = makeRng(seed);
  while (s.winner === null) {
    const name = s.toMove === X ? a : b;
    const { move } = AGENTS[name].select(s, params[name] ?? {}, rng);
    assert.ok(s.isLegal(move), `${name} played illegal ${move}`);
    s.push(move);
  }
  return s.winner;
}

test("row, column and diagonals win; exact-k rejects overlines", () => {
  const s = getGame({ m: 3, n: 3, k: 3 }).newState();
  for (const mv of [0, 3, 4, 5, 8]) s.push(mv);
  assert.equal(s.winner, X);
  assert.deepEqual(s.winningLine().sort(), [0, 4, 8]);
  const e = getGame({ m: 1, n: 7, k: 3, exact: true }).newState();
  // X at 0,1,3,4 then 2 makes five in a row: not a win under exact-k
  e.cells.set([1, 1, 0, 1, 1, 0, 0]);
  assert.equal(e.wouldWin(2, X), false);
  assert.equal(e.wouldWin(5, X), true);
});

test("gravity only allows the lowest empty cell per column", () => {
  const s = getGame({ m: 3, n: 3, k: 3, gravity: true }).newState();
  assert.deepEqual(s.legalMoves(), [6, 7, 8]);
  s.push(7);
  assert.ok(s.isLegal(4) && !s.isLegal(1));
  s.pop();
  assert.deepEqual(s.legalMoves(), [6, 7, 8]);
});

test("perfect play draws tic-tac-toe and never loses to the others", () => {
  assert.equal(play({ m: 3, n: 3, k: 3 }, "minimax", "minimax"), DRAW);
  for (let seed = 0; seed < 10; seed++) {
    assert.notEqual(play({ m: 3, n: 3, k: 3 }, "random", "minimax", seed), X);
    assert.notEqual(play({ m: 3, n: 3, k: 3 }, "minimax", "mcts", seed, { mcts: { iterations: 300 } }), O);
  }
});

test("minimax finds the first-player win on (3,4,3)", () => {
  const s = getGame({ m: 3, n: 4, k: 3 }).newState();
  assert.equal(AGENTS.minimax.select(s, {}, makeRng(1)).info.verdict, "win");
});

test("minimax refuses boards it cannot solve and degrades gracefully", () => {
  const g = getGame({ m: 9, n: 9, k: 5 });
  assert.ok(minimaxEstimate(g).seconds > 1e6);
  const r = AGENTS.minimax.select(g.newState(), {}, makeRng(1));
  assert.ok(r.info.refused && g.newState().isLegal(r.move));
});

test("alpha-beta takes immediate wins and blocks", () => {
  const g = getGame({ m: 7, n: 7, k: 4 }), s = g.newState();
  for (const mv of [24, 0, 25, 6, 26]) s.push(mv); // X threatens 23 or 27
  const block = AGENTS.alphabeta.select(s, {}, makeRng(1)).move;
  assert.ok([23, 27].includes(block));
  s.push(48);
  const win = AGENTS.alphabeta.select(s, {}, makeRng(1)).move;
  assert.ok([23, 27].includes(win));
});

test("stronger agents beat random on (7,7,4)", () => {
  let wins = 0;
  for (let seed = 0; seed < 6; seed++) {
    if (play({ m: 7, n: 7, k: 4 }, "alphabeta", "random", seed, { alphabeta: { depth: 2 } }) === X) wins++;
  }
  assert.ok(wins >= 5, `alphabeta won ${wins}/6`);
});

test("position bound matches the Python implementation", () => {
  assert.equal(positionBound(getGame({ m: 3, n: 3, k: 3 })), 6046);
  assert.equal(positionBound(getGame({ m: 4, n: 4, k: 3 })), 10165779);
  const c4 = positionBound(getGame({ m: 6, n: 7, k: 4, gravity: true }));
  assert.ok(Math.abs(c4 / 532875860165503 - 1) < 1e-9);
});

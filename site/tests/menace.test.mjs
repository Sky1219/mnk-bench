import { test } from "node:test";
import assert from "node:assert/strict";
import { getGame, X, O, DRAW } from "../assets/mnk.js";
import { Canonicalizer, Menace, symmetries } from "../assets/menace.js";
import { AGENTS, makeRng } from "../assets/agents.js";

function countBoxes(cfg, useSymmetry) {
  const g = getGame(cfg), canon = new Canonicalizer(g, useSymmetry), s = g.newState();
  const seen = new Set(), boxes = new Set();
  (function visit() {
    const [key, sym] = canon.canonical(s.cells);
    if (seen.has(key)) return;
    seen.add(key);
    if (s.winner !== null) return;
    if (s.toMove === X && canon.orbitReps(key, s.legalMoves().map((mv) => canon.toCanon(sym, mv))).length > 1) boxes.add(key);
    for (const mv of s.legalMoves()) { s.push(mv); visit(); s.pop(); }
  })();
  return boxes.size;
}

test("symmetry groups have the right sizes and are permutations", () => {
  assert.equal(symmetries(getGame({ m: 3, n: 3, k: 3 })).length, 8);
  assert.equal(symmetries(getGame({ m: 3, n: 4, k: 3 })).length, 4);
  assert.equal(symmetries(getGame({ m: 6, n: 7, k: 4, gravity: true })).length, 2);
  for (const p of symmetries(getGame({ m: 3, n: 3, k: 3 }))) assert.equal(new Set(p).size, 9);
});

test("MENACE needs Michie's 304 matchboxes for X on tic-tac-toe", () => {
  assert.equal(countBoxes({ m: 3, n: 3, k: 3 }, true), 304);
});

test("MENACE learns to stop losing to perfect play", () => {
  const g = getGame({ m: 3, n: 3, k: 3 }), rng = makeRng(7);
  const menace = new Menace(g, { rng });
  let lastLosses = 0;
  for (let i = 0; i < 3000; i++) {
    const s = g.newState();
    while (s.winner === null) {
      s.push(s.toMove === X ? menace.select(s).move : AGENTS.minimax.select(s, {}, rng).move);
    }
    menace.learn(s, X);
    if (i >= 2500 && s.winner === O) lastLosses++;
  }
  assert.ok(menace.boxes.size <= 304);
  assert.ok(lastLosses <= 25, `lost ${lastLosses}/500 late games`);
  const beads = menace.beadsFor(g.newState());
  assert.equal(beads.size, 9); // every opening cell maps to one of the 3 distinct opening moves
});

test("the MENACE agent trains on first use and never loses tic-tac-toe after training", () => {
  const g = getGame({ m: 3, n: 3, k: 3 });
  for (let seed = 0; seed < 20; seed++) {
    const s = g.newState(), rng = makeRng(seed);
    while (s.winner === null) {
      const r = s.toMove === X ? AGENTS.menace.select(s, { games: 3000 }, rng) : AGENTS.random.select(s, {}, rng);
      assert.ok(s.isLegal(r.move));
      s.push(r.move);
    }
    assert.notEqual(s.winner, O, `lost on seed ${seed}`);
  }
});

test("MENACE on a big board reports unseen positions instead of failing", () => {
  const s = getGame({ m: 9, n: 9, k: 5 }).newState();
  s.push(0); s.push(80); s.push(8); s.push(72);
  const r = AGENTS.menace.select(s, { games: 50, trainSeconds: 2 }, makeRng(1));
  assert.ok(s.isLegal(r.move) && r.info.unseen && r.info.trained > 0);
});

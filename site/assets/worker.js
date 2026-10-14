// Runs agents off the main thread so the page stays responsive while they think.
//
// One move (Play page):
//   in:  { id, config, history, agent, params, seed }
//   out: { id, move, info, ms } or { id, error }
// A whole game (Arena page):
//   in:  { type: "game", id, config, players: { 1: { kind, params }, 2: { kind, params } }, seed, live }
//   out: { type: "move", id, move } after every move when live, then
//        { type: "done", id, winner, plies, time: { 1, 2 }, moves: { 1, 2 }, notes } or { type: "error", id, error }
import { getGame } from "./mnk.js";
import { AGENTS, makeRng } from "./agents.js";

function playGame({ id, config, players, seed, live }) {
  const state = getGame(config).newState(), rng = makeRng(seed);
  const time = { 1: 0, 2: 0 }, moves = { 1: 0, 2: 0 }, notes = new Set();
  while (state.winner === null) {
    const side = state.toMove, p = players[side], t0 = performance.now();
    const { move, info } = AGENTS[p.kind].select(state, p.params ?? {}, rng);
    time[side] += performance.now() - t0 - (info.trainMs ?? 0); // MENACE's one-off training is not thinking time
    moves[side]++;
    if (info.refused) notes.add(`${AGENTS[p.kind].label} refused this board and played alpha-beta`);
    if (info.aborted) notes.add(`${AGENTS[p.kind].label} hit its time guard`);
    if (info.unseen) notes.add("MENACE met positions it never saw in training");
    state.push(move);
    if (live) self.postMessage({ type: "move", id, move });
  }
  self.postMessage({ type: "done", id, winner: state.winner, plies: state.history.length, time, moves, notes: [...notes] });
}

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "game") {
    try { playGame(msg); } catch (err) { self.postMessage({ type: "error", id: msg.id, error: String(err && err.stack || err) }); }
    return;
  }
  const { id, config, history, agent, params = {}, seed } = msg;
  try {
    const state = getGame(config).newState();
    for (const mv of history) state.push(mv);
    const t0 = performance.now();
    const { move, info } = AGENTS[agent].select(state, params, makeRng(seed ?? (Date.now() ^ (id * 2654435761))));
    self.postMessage({ id, move, info, ms: performance.now() - t0 });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.stack || err) });
  }
};

// Runs agents off the main thread so the page stays responsive while they think.
// Message in:  { id, config, history, agent, params, seed }
// Message out: { id, move, info, ms } or { id, error }
import { getGame } from "./mnk.js";
import { AGENTS, makeRng } from "./agents.js";

self.onmessage = (e) => {
  const { id, config, history, agent, params = {}, seed } = e.data;
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

# mnk-bench

**Where does each game-playing AI fall apart as the board gets bigger?**

`mnk-bench` is a Python framework for pitting different kinds of AI agents against each other on any *m,n,k* game, from tic-tac-toe to gomoku, and measuring exactly how their strength, speed, and cost change as the game grows.

> Status: early development. The MVP (tic-tac-toe with random, minimax, and MENACE agents) is in progress.

## Why I'm building this

After a club meeting one day, some friends and I went to our college's honors lounge to play a few games, and one friend suggested "Japanese Connect 5," better known as gomoku. I'd never played it, but a few moves in I realized it was just tic-tac-toe on a much bigger board.

That reminded me of one of my favorite stories from early machine learning: **MENACE**, the Machine Educable Noughts And Crosses Engine. In 1961, when computers were still hard to come by, Donald Michie built a tic-tac-toe learner out of about 300 matchboxes filled with colored beads. Each matchbox stood for a board position and each bead color for a move. After every game, beads were added for moves that led to a win and taken away for moves that led to a loss. Over a few hundred games, a pile of matchboxes learned to play tic-tac-toe well.

Tic-tac-toe and gomoku are both **m,n,k games**: two players take turns placing stones on an m×n board, and the first to get k in a row wins. Tic-tac-toe is (3,3,3); gomoku is usually (15,15,5).

While telling my friends the MENACE story, it was obvious that a matchbox for every position could never work on a gomoku board. There are far more positions than atoms in the universe. But that raised a better question: **where exactly does it fall apart?** At 4×4? At 6×6? And what about other approaches, like exhaustive search, sampling, or neural networks? Each one should break somewhere different, and for a different reason.

This project is my attempt to find out: one general m,n,k framework, many kinds of agents, and measurements of how each one's performance changes as the game gets harder.

## Goals

- **One framework for any (m,n,k).** Tic-tac-toe, gomoku, and everything in between, with rule flags for variants like Connect Four (gravity) and exact-k win rules.
- **Many kinds of agents behind one interface**, including:
  - Baselines: random and rule-based
  - Exact search: full minimax with alpha-beta pruning (perfect play on small boards)
  - Tabular learning: MENACE, TD learning, and Q-learning
  - Heuristic search: depth-limited alpha-beta with a threat-based evaluation
  - Sampling: Monte Carlo Tree Search
  - Stretch goals: an AlphaZero-style network-guided MCTS and LLM players
- **Compare them on strength, speed, and cost:** win/draw/loss rates, Elo ratings, optimal-move rate against perfect play, time per move, memory, and games needed to learn.
- **Know when a method won't fit.** Each agent estimates whether it can run on a given board within a time and memory budget, and refuses, degrades, or aborts cleanly when it can't.
- **Graphs generated automatically**: learning curves, scaling plots, head-to-head heatmaps, and a "feasibility map" showing which methods survive at which board sizes.
- **Reproducible results**: seeded runs, with results saved as data so plots can be regenerated without re-running experiments.

## Roadmap

1. **MVP:** m,n,k game engine, random/minimax/MENACE agents, tournament runner, and a MENACE learning-curve plot on tic-tac-toe.
2. Rule-based agent and a head-to-head results table.
3. Board symmetries for MENACE (the trick that got Michie down to about 300 matchboxes).
4. Q-learning and TD learning, compared against MENACE.
5. Bigger boards such as (6,6,4), where tabular methods start to choke; depth-limited search.
6. Monte Carlo Tree Search and Elo ratings.
7. Automatic feasibility checks and the feasibility map.
8. Performance work and gomoku (15,15,5).
9. CLI, HTML reports, packaging, and CI.

## Related work

- Donald Michie's original MENACE (1961) and its many recreations, including Matt Parker's matchbox build.
- DeepMind's [OpenSpiel](https://github.com/google-deepmind/open_spiel), a broad research framework covering many games and algorithms. `mnk-bench` is deliberately narrow: one game family, focused on how and where each method breaks down as the game scales.

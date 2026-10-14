# Fixing a game network's blind spots with a few oracle labels: PRISM-style targeted selection

*A study run inside mnk-bench. Code: [`src/mnkbench/submodular.py`](../../src/mnkbench/submodular.py), [`src/mnkbench/experiments/targeted.py`](../../src/mnkbench/experiments/targeted.py). Data and plots: [`docs/results/targeted/`](../results/targeted/). Reproduce with `mnk-bench experiment targeted --out results` (needs PyTorch; about 1 hour on 4 cores).*

<!--ABSTRACT-->

## 1. Motivation

The rest of mnk-bench asks *where each method breaks* as the board grows. Exact search breaks first (minimax is refused from (5,5,4)), but an exact or very deep search is still the best *teacher* available: it can tell a learned agent the right move in any position it is shown. The catch is cost. Each label is a solver call, so in practice only a small budget of positions can be labelled.

That raises a practical question. Suppose you notice your network has a specific blind spot, such as missing immediate wins, and you can collect a handful of positions that show it. **Which positions should you spend the labelling budget on?** This is the *targeted learning* problem that PRISM (Kothawade, Kaushal, Ramakrishnan, Bilmes and Iyer, AAAI 2022) studies for image classifiers. A model underperforms on a rare class, a few examples of that class form a *query set*, and a submodular mutual-information function between candidate subsets and the query set chooses which unlabelled examples to label. This document transfers that setting to games.

## 2. Background: submodular information measures

A set function *f* over a ground set *V* is submodular if marginal gains diminish: *f(A ∪ {j}) − f(A) ≥ f(B ∪ {j}) − f(B)* for *A ⊆ B*. Greedy maximisation of a monotone submodular *f* under a cardinality budget is within (1 − 1/e) of optimal. All functions below use a non-negative similarity kernel *S* (here: cosine similarity of network embeddings).

PRISM builds information measures from a submodular *f*. Given a query set *Q* (what we want) and a private set *P* (what we want to avoid), it defines:

- *submodular mutual information*: *I_f(A; Q) = f(A) + f(Q) − f(A ∪ Q)*;
- *conditional gain*: *f(A | P) = f(A ∪ P) − f(P)*;
- *conditional mutual information*: *I_f(A; Q | P)*.

Instantiated with standard functions, they give the selection objectives compared here (formulas as in the authors' SUBMODLIB implementation):

| name | objective | behaviour |
|---|---|---|
| FL (facility location) | Σᵢ∈V maxⱼ∈A Sᵢⱼ | representation/diversity, ignores Q |
| LogDet | log det(S_A + λI) | diversity (DPP-like), ignores Q |
| **GCMI** | 2λ Σⱼ∈A Σ_q∈Q S_jq | pure query relevance (modular) |
| **FLMI** | Σᵢ∈V min(maxⱼ∈A Sᵢⱼ, η max_q∈Q Sᵢ_q) | query coverage + diversity; saturates |
| **FLVMI** | Σ_q∈Q maxⱼ∈A S_qj + η Σⱼ∈A max_q S_jq | relevance + query coverage, no saturation |
| **COM** | η Σⱼ∈A ψ(Σ_q S_jq) + Σ_q ψ(Σⱼ∈A S_jq), ψ = √ | between GCMI and FLMI |
| **LogDetMI** | log det(S_A) − log det(S_A − η² S_AQ S_Q⁻¹ S_AQᵀ) | relevance + diversity |
| **FLCMI** | Σᵢ max(min(maxⱼ∈A Sᵢⱼ, η max_q Sᵢ_q) − ν max_p Sᵢ_p, 0) | like Q, unlike P |

*η* trades query relevance against diversity, and *ν* sets how strongly *P* is avoided. The implementation (`mnkbench.submodular`) provides each function's direct definition, vectorised marginal gains, plain greedy and Minoux's lazy greedy (exact for submodular functions, and 25–70× faster here). Tests check incremental gains against the definitions, check diminishing returns and monotonicity on random instances, and check greedy against brute force for the (1 − 1/e) bound.

## 3. Setup

**Game.** Mini Connect Four: the (5,5,4) game with gravity. It is small enough that the exact solver labels a position in about 20 ms, and tactical enough to produce clear failure modes.

**The network with blind spots.** An AlphaZero-style policy/value network (the project's 39k-parameter residual CNN) trained by self-play for only 4 iterations × 30 games (about 20 s). Without search, its top policy move keeps the game-theoretic outcome in about 60% of the positions where the choice matters ("decision positions").

**Target weaknesses (the "rare classes").** Positions are grouped by tactical motif, detected exactly from the rules:

| motif | definition | share of positions | base network's blunder rate there |
|---|---|---:|---:|
| `win` | the side to move can win immediately | ~10% | ~35% |
| `block` | the opponent threatens to win next move; it must be blocked | ~10% | ~24% |
| `poison` | some legal move lets the opponent win at once (in gravity games: playing directly below the opponent's threat cell) | ~16% | ~30% |

Selection methods never see these motif labels for the pool. They only see network embeddings and the query examples. The motif labels are used solely for evaluation and for the oracle reference.

**Data.** For each seed, positions are sampled from games between a mix of agents (random, rule-based, MCTS-100 and alpha-beta depth 2, with 15% random moves), at most three per game, de-duplicated up to symmetry. They are split into an unlabelled **pool** (3,000), a **development set** (1,000) from which the query set *Q* (20 positions of the target motif) and the private set *P* (20 motif-free positions) are drawn, and a **test set** (2,000).

**Oracle labels.** For a selected position the exact solver returns every *outcome-preserving* move (a win stays a win, a draw stays a draw) and the solved value.

**Selection.** Embeddings are the residual tower's output (flattened, 800-d, non-negative ReLU features), with cosine similarity as the kernel. Methods:

- baselines: random; uncertainty (max policy entropy; min top-2 margin); diversity (FL, LogDet);
- PRISM mutual information with *Q*: GCMI, FLMI, FLVMI, COM, LogDetMI (η = 1);
- PRISM conditional mutual information with *Q* and *P*: FLCMI (η = ν = 1);
- **oracle** reference: random positions *of the target motif* (it uses the hidden labels, so it bounds what perfect targeting could achieve with random choice inside the class).

Greedy selection is nested, so one greedy run at the largest budget gives every smaller budget as a prefix.

**Fine-tuning.** Each method's *k* selected positions, plus the 20 labelled query positions (the same for every method), fine-tune a copy of the network. The training budget is identical for every method: 300 Adam steps, batch 64, learning rate 1e-3, mirror augmentation. The policy target is uniform over the outcome-preserving moves and the value target is the solved outcome. A distillation term on unlabelled pool positions keeps the policy close to the original elsewhere.

**Metrics.** These are raw-policy accuracy (does the top move preserve the outcome?) on held-out *decision positions of the target motif* (**target accuracy**) and on *all* held-out decision positions (**overall accuracy**), plus the share of selected positions that belong to the motif (**hit rate**, PRISM's "how many rare-class examples were selected"). For *k* = 200 the fine-tuned networks also play 20 games each against perfect play and against depth-2 alpha-beta.

**Protocol.** 5 seeds (fresh data, queries and training seeds; the base network is fixed) × 3 motifs × 12 methods × budgets k ∈ {25, 50, 100, 200, 400}. The ablations at k = 100 on the `poison` motif cover η ∈ {¼, ½, 1, 2, 4}, |Q| ∈ {5, 10, 20, 50, 100}, and raw-board features instead of network features. Means are reported with 95% confidence intervals over seeds, and paired comparisons against random use the same seed and data.

<!--RESULTS-->

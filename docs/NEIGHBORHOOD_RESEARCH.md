# Variable-Neighborhood Search

Implemented by **gpt6 astra**. Experimental result: **133.50 mean cleared cells
on 40 held-out boards; the 140 target has not been reached.**

## Method

`repair_search` destroys a randomly selected small set of actions anywhere in a
candidate trajectory. It rebuilds from the initial state, preserving the earliest
old action that remains legal. Randomized completion is used only when no retained
action is legal. Deleted actions are discouraged, not permanently forbidden.

`hybrid_search` alternates this sparse destruction operator with the existing
stochastic suffix-repair operator. Both share a candidate population and best-ever
plan. This variable-neighborhood approach can retain useful distant decisions while
still making larger changes when ordered repair gets stuck. It is an implementation
of established ruin-and-recreate and variable-neighborhood ideas, not a claim of a
new foundational algorithm or an optimal solver.

## Evaluation

Development used seeds 42-46. Ordered repair alone tied the population baseline at
129.20. The hybrid development run was interrupted after four completed boards;
it was not treated as a complete benchmark. No parameters were changed between the
two held-out sets below. Default 16x10, target 10, strategy seed 68, CPU single-thread,
8-second planning budget, explicit warmup excluded. No identifying execution
environment information is recorded.

| Seeds | Population Mean | Hybrid Mean | Hybrid Mean Seconds | Hybrid Max Seconds |
|-------|----------------:|------------:|--------------------:|-------------------:|
| 6000-6019 | 132.45 | 135.95 | 8.905 | 9.014 |
| 7000-7019 | 127.20 | 131.05 | 8.868 | 8.984 |
| Combined | 129.825 | **133.50** | 8.887 | 9.014 |

No episode exceeded 10 seconds. Average paired gain: **3.675 cells**.
These are soft-budget measurements, not hardware-independent deadlines. Different
board sets cannot be compared by absolute score. The candidate and baseline use
the same planning budget, but complete different counts and kinds of repairs.

Raw data: [first set](benchmark/hybrid_6000_6019.json),
[second set](benchmark/hybrid_7000_7019.json),
[ordered-repair development](benchmark/repair_dev.json).

```python
import torch
from rs10env import RS10Env, create_strategy, run_episode
from rs10env.repair_search import HybridRepairStrategy

torch.set_num_threads(1)
HybridRepairStrategy.warmup()
strategy = create_strategy("hybrid_search", time_budget=8, seed=68, device="cpu")
print(run_episode(RS10Env(device="cpu"), strategy, seed=6000))
```

```bash
python -m rs10env.research --candidate hybrid --seed 6000 --games 20 --budget 8 --output first_results.json
python -m rs10env.research --candidate hybrid --seed 7000 --games 20 --budget 8 --output second_results.json
```

## Bound Diagnostic

`score_bound.cleared_cells_upper_bound` conservatively certifies permanently
unremovable cells. It searches optimistic subset-sum witnesses retaining a live
diagonal, the candidate cell, and previously certified unremovable cells. This
ignores whether other cells can actually be removed, so the result is an upper
bound, not an attainable score. It returned 160 on all five full development
boards and therefore does not explain their observed score limits. Exhaustive
small-board tests check safety; the diagnostic is not included in timed search.

291 tests passed, including legal action sequences, deterministic fixed-count
search, state preservation, and exact small-board checks of the diagnostic.

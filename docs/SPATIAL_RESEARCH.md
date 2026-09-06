# Spatial And Exact Search

Implemented by **gpt6 astra**. Neither experiment establishes progress to a
140-cell mean. Execution details are intentionally omitted.

## Geometric Neighborhoods

For a stranded cell `(r,c)` and rectangle `[r1,r2] x [c1,c2]`, use the
Chebyshev distance `max(0, r1-r, r-r2, c1-c, c-c2)`. Select a residual cell
from a candidate's terminal board and discourage old actions within a randomly
chosen radius of 1-3. Rebuild the trajectory while preserving still-legal old
actions, mixed with the previous neighborhoods. No independence of regions is
assumed: every executed rectangle is checked against the evolving board.

Development seeds 42-46: spatial mean 132.60 versus hybrid 130.00. Parameters
were then fixed for held-out seeds 8000-8019:

| Strategy | Mean Cleared | Mean Episode Seconds | Maximum Seconds |
|----------|-------------:|---------------------:|----------------:|
| hybrid_search | 129.85 | 8.853 | 9.025 |
| spatial_search | 129.90 | 8.856 | 9.043 |

Both used 8-second planning budgets; explicit warmup was excluded. No episode
exceeded 10 seconds. A 0.05-cell mean difference is not evidence of a stronger
strategy. Spatial search remains an experimental class, not a recommended default.
Raw results: [development](benchmark/spatial_dev.json),
[held-out](benchmark/spatial_8000_8019.json).

## Exact Endgame Mathematics

The initial values never change except for deletion. A state is therefore fully
specified by its live-cell occupancy bitset. For a supplied rectangle, the solver
checks the live diagonal and exact sum, then deletes its occupied bits. Distinct
actions leading to the same occupancy are merged. A memoized recurrence maximizes
additional cleared cells over all legal successors, with live-cell count as an
optimistic upper bound. Only fully solved states are cached as exact.

`solve_endgame` returns a legal continuation, cleared count, expanded nodes and
`proven_optimal`. A node cap is not a time cap; exhaustion never claims optimality.
Proofs apply only to the supplied starting state, action set and remaining horizon,
not to the original game's earlier choices.

On the five development hybrid trajectories, fixed prefixes followed by the final
8 actions were solved to optimality in 61-151 nodes, taking 0.057-0.152 seconds.
**All five optimal scores matched the existing trajectory.**

Moving the cut back to the final 16 actions still produced no improvement. Three
states were solved exactly; two exhausted 20,000 nodes and remained unproven.
Elapsed times were 2.69-14.93 seconds. These are **offline diagnostic costs**, not
compliant online episode timings. The exact solver is not integrated into timed
strategy execution.

Results: [8-action suffix](benchmark/exact_suffix8_dev.json),
[16-action suffix](benchmark/exact_suffix16_dev.json).

```bash
python -m rs10env.research --candidate spatial --baseline hybrid --seed 8000 --games 20 --budget 8 --output spatial_results.json
python -m rs10env.endgame_experiment docs/benchmark/spatial_dev.json --suffix 8 --nodes 1000 --output endgame_results.json
```

325 tests passed, including tiny-board exhaustive comparisons, input preservation,
legal trajectories and budget-exhaustion behavior. The findings suggest examining
earlier dependency chains, not assuming remaining errors lie in the last few moves.

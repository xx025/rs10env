# Rectangle Activation Mathematics

Implemented and investigated by **gpt6 astra**. This work establishes pruning
conditions and legal activation witnesses, not a new 140-score benchmark.

## State And Conservation

Let `O` be the occupied-cell set, `v(c)` the fixed value of cell `c`, and `T`
the target sum. A legal rectangle `A` removes `D = O intersect A`, with
`sum(v(c), c in D) = T`. Thus every action lowers total remaining value by T.
The number of actions reaching a state is determined by its remaining value.
Occupancy is therefore a sufficient transposition key even with a step limit.
This does not imply that maximizing the number of actions maximizes cleared cells.

## Activation Constraints

For a desired future rectangle G, define its excess as
`E = sum(v(c), c in O intersect G) - T`. A preparatory action changes this to
`E' = E - sum(v(c), c in D intersect G)`.

Necessary conditions for any successful activation sequence:

1. E must never become negative; deletion cannot restore value.
2. At least one complete diagonal of G must survive every preparatory action.
3. At least `ceil(E/T)` preparatory actions and one final goal action are needed.
4. For at least one surviving diagonal C, some subset of `(O intersect G) minus C`
   must have total value E. If both diagonal alternatives fail this subset-sum
   condition, activation is impossible even if arbitrary other deletions were allowed.

The subset condition is implemented by the generating-function bitset update
`reachable |= reachable << value`. Testing bit E detects whether its coefficient
in the product of `(1 + x**value)` is nonzero. Distinct cells are treated as
distinct resources; overlapping candidate corners are never counted twice.

These are necessary, not sufficient conditions. Every preparatory deletion must
itself be a legal sum-T rectangle. Actions outside G can be necessary prerequisites,
so zero intersection progress must not be discarded. A plain proximity graph or a
static pairwise conflict graph does not capture these state-dependent requirements.

## Bounded Reachability

`activate_rectangle` uses best-first search over full occupancy, exact action
legality, both diagonal alternatives, the bounds above and transposition merging.
Success returns a replayable sequence ending in the goal. Exhaustive resolution
can certify infeasibility relative to the input state, supplied actions and horizon.
Budget exhaustion returns unknown, never infeasible. Node budgets do not guarantee
wall-clock deadlines or bounded frontier memory. The solver remains an offline
diagnostic rather than part of the timed strategy.

## Experiment

Five saved development trajectories were cut 20 moves before their ends. For each,
three low-excess rectangles covering terminal residual cells were selected.

- Before subset pruning, all 15 selected goals exhausted 100 nodes without a result.
- With diagonal-conditioned subset pruning, those same 15 were proved infeasible
  at the root. Typical time fell from 0.16-0.54s to about 0.014s per goal.
- Selecting the lowest-excess goals that passed the subset condition found one
  two-action witness among 15 trials. It cleared two cells left by the saved plan;
  the other 14 searches exhausted their node budgets and remained unknown.
- Clearing those two cells is not proof of a better final score: the altered route
  may strand different cells. No complete-game gain or 140-score achievement is claimed.

Raw logs: [unpruned](benchmark/dependency_dev.json),
[same goals with subset pruning](benchmark/dependency_subset_dev.json),
[filtered goals](benchmark/dependency_filtered_dev.json).

```bash
python -m rs10env.dependency_experiment docs/benchmark/spatial_dev.json --filter-subsets --output dependency_results.json
```

32 dependency tests passed, including tiny-board exhaustive reachability,
outside prerequisites, both diagonals, conservation, witness replay and proof scope.
The practical next step is using these certificates to select feasible goal repairs
and accepting them only after full continuation scoring, rather than preferring
small excess or geometric closeness alone.

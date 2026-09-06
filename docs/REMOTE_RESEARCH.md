# Remote Adaptive Search Research

Implemented by: **gpt6 astra**. This experiment did **not** achieve the target
of 140 mean cleared cells within 10 seconds per episode.

## Measurement

Tests used one CPU thread with explicit compilation warmup. Identifying machine
details, software environment metadata and deployment paths are not published.

## Algorithm

`AdaptiveSearchStrategy` implements level-1 or level-2 NRPA-style policy
adaptation, using the same compiled legal-action generator as population search.
The policy has four game-phase buckets, indexed by the fraction of cells removed.
At each state, legal actions are sampled from the policy softmax. Replaying the
best trajectory updates logits by `learning_rate * (indicator - probability)`.
All probabilities for one adaptation use the pre-update policy.

Level 2 retains an outer policy. Inner searches start from copies of it; their
best trajectories guide outer updates. Periodic geometry-prior restarts and a
separate best-ever plan prevent a bad learning run from discarding a good result.
This is online per-board learning, not a trained neural model or a new foundational
algorithm. Phase buckets provide only coarse state conditioning.

## Development

Seeds 42-46, strategy seed 68, 8-second planning budgets:

| Variant | Mean Cleared |
|---------|-------------:|
| Population baseline | 129.20 |
| Level 1, 64 updates/restart | 123.40 |
| Level 1, 1024 updates/restart | 125.00 |
| Level 2, 64 inner / 16 outer | 127.20 |
| Level 2, 32 inner / 64 outer | 128.80 |

With exactly 10,000 sampled trajectories on these development boards, population
search averaged 124.00 cells (1.47s) and the 32/64 adaptive variant averaged 126.80
(3.11s). This is **not equal simulator work**: adaptation additionally replays
trajectories, and population repairs often preserve prefixes. It suggests better
use of sampled trajectories, not proof of equal-compute superiority.

## Held-Out Results

Parameters fixed at level 2, 32 inner / 64 outer, learning rate 1.0.
Seeds 5000-5029 were not used for tuning. Default 16x10 board, target 10,
strategy seed 68 reset each episode; sequential execution on the same remote host.

| Strategy | Mean Cleared | Mean Seconds | Maximum Seconds | Episodes Over 10s |
|----------|-------------:|-------------:|----------------:|------------------:|
| population_search | 125.53 | 8.735 | 9.040 | 0/30 |
| adaptive_search | 127.70 | 8.751 | 9.036 | 0/30 |

Adaptive search won 22, tied 2, and lost 6. The paired improvement was 2.17 cells,
with approximate 95% CI [0.87, 3.46]. **The 140 target was missed by 12.30 cells.**
Do not compare the absolute mean directly with earlier 129.98 results: those used
different boards and a different machine. There is no claim that this experimental
strategy is a broadly superior replacement for population search.

Raw logs include actual initial boards, returned action IDs, runtime, rollouts,
and warmup costs. Earlier development log configs predate the explicit
`levels` and `outer_steps` CLI options: filenames `adaptive64` and `adaptive1024`
mean level 1; `nested64` means level 2 with 16 outer steps. All development runs
are retained, including weaker variants, under `docs/benchmark/remote_*.json`.

## Reproduce

```bash
python -m rs10env.research --seed 5000 --games 30 --budget 8 --steps 32 --levels 2 --outer-steps 64 --output new_results.json
```

```python
from rs10env import RS10Env, create_strategy, run_episode
from rs10env.adaptive_search import AdaptiveSearchStrategy

AdaptiveSearchStrategy.warmup()
strategy = create_strategy("adaptive_search", time_budget=8, adaptation_steps=32,
                           levels=2, outer_steps=64, seed=68, device="cpu")
print(run_episode(RS10Env(device="cpu"), strategy, seed=5000))
```

Warmup is explicit and excluded from episode timing; cold startup can exceed the
10-second limit. The deadline is soft and checked every four rollouts. Supplying
`max_rollouts` to adaptive search disables its deadline for deterministic
fixed-work experiments; population search stops at either its rollout cap or
deadline. Give population search sufficient time when comparing fixed counts.

All 203 tests passed on the remote host after adding the algorithm tests.
Next research should investigate richer state-dependent value guidance or
offline-trained search guidance; additional nesting alone has not delivered 140.

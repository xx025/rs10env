# Policy Learning Experiments

Implemented by **gpt6 astra**. Two completed training experiments did not produce
a competitive policy. Neither achieves the 140-cell target. No deployment details
are included in artifacts or this report.

## Method

A small board CNN produces cell embeddings. Valid rectangles are scored using
rectangle pooling, corner and global embeddings, and normalized geometry. Illegal
actions are masked out. A global value head predicts future normalized clearance.

First, imitate HybridRepair trajectories with action cross-entropy and value loss.
Then optimize sampled episodes using REINFORCE with a detached learned baseline
and entropy regularization. Returns count cleared cells divided by board area,
excluding the environment's terminal bonus. This is a basic actor-critic pilot,
not PPO or a mature search-guided reinforcement-learning system.

## Completed Results

| Experiment | Teacher Games | Imitation Epochs | RL Games | Held-Out Games | Imitation Mean | Post-RL Mean |
|------------|--------------:|----------------:|---------:|---------------:|---------------:|-------------:|
| Pilot | 32 | 3 | 32 | 10 | 100.60 | 100.90 |
| Larger pilot | 256 | 5 | 256 | 30 | 106.83 | 102.47 |

Teacher planning budget was 0.2 seconds per board, not the earlier 8-second search
budget. The larger training set's teacher mean was 128.13, not an independent
test-set comparison. Its supervised loss fell from 2.790 to 2.680, but post-RL
held-out clearance deteriorated by 4.37 cells. Loss reduction alone is not evidence
of a strong game policy. The two experiments use different held-out boards and
therefore their absolute means are not a controlled scaling comparison.

Raw metrics: [pilot](benchmark/learning_pilot.json),
[larger pilot](benchmark/learning_scale.json). The frozen supervised and post-RL
weights were retained separately on the execution endpoint; neither is promoted
as a recommended strategy. No long-running training job was left running by these
commands. Model-only inference latency was not benchmarked against the 10-second
target because playing strength failed first.

## Reproduce

```bash
python -m rs10env.learning --output learning_pilot --teacher-games 32 --teacher-budget .2 --epochs 3 --rl-games 32 --eval-games 10 --seed 100000
python -m rs10env.learning --output learning_scale --teacher-games 256 --teacher-budget .2 --epochs 5 --rl-games 256 --eval-games 30 --seed 200000
```

Each output directory must be new. It receives `supervised.pt`, `reinforcement.pt`
and `metrics.json`. Checkpoints contain model weights and configuration, not host
identifiers. The script currently starts a fresh training run rather than resuming
optimizer state. Five learning tests passed on the execution endpoint.

## Interpretation

This rules out recommending the current small CNN and short REINFORCE schedule;
it does not rule out reinforcement learning. Larger training requires stronger
search targets, more stable updates, validation-based checkpoint selection, and
likely search-guided value use rather than replacing search with greedy inference.
Training on one selected action also penalizes equally good commuting choices;
future targets should account for equivalent actions or return-ranked alternatives.

# RS10Env

**[English](README.md)** | [简体中文](README.zh-CN.md)

Gymnasium-compatible RS10 board game environment and heuristic strategies (PyTorch).

## Latest Strategy

[Spatial and exact-search experiments](docs/SPATIAL_RESEARCH.md) found no meaningful
held-out gain from distance-based neighborhood selection. Exact suffix diagnostics
showed five fixed-prefix 8-move endgames were already optimal; this does not prove
the complete games optimal. The 140-cell target remains unmet.

Further experiment: [variable-neighborhood repair](docs/NEIGHBORHOOD_RESEARCH.md)
averaged **133.50** on 40 new paired boards versus 129.825 for population search,
with no episode over 10 seconds after warmup. **The 140 target remains unmet.**

Experimental follow-up: [remote NRPA-style algorithm research](docs/REMOTE_RESEARCH.md)
on the user's remote machine found 127.70 versus 125.53 for population search on
30 new paired boards (about 8.75s/game). **The 140-cell target was not reached.**
`adaptive_search` is available through the factory but is not a default recommendation.

**Latest strategy: `population_search`. Implemented by: gpt6 astra.**

Optional Numba-compiled population repair reaches **129.98 mean cleared cells**
on 60 held-out boards, with **8.20 seconds/game**, maximum **8.63 seconds**,
and **0/60 episodes over 10 seconds** on the test machine after warmup.
This is a measured result, not a hard real-time guarantee on other hardware.

```bash
pip install -e '.[search]'
```

```python
import torch
from rs10env import RS10Env, create_strategy, run_episode
from rs10env.fast_search import PopulationSearchStrategy

torch.set_num_threads(1)
PopulationSearchStrategy.warmup()  # Explicit JIT cost, outside the episode timer.
env = RS10Env(device="cpu")
strategy = create_strategy("population_search", time_budget=8.0, seed=68, device="cpu")
print(run_episode(env, strategy, seed=3000))
```

| Held-Out Seeds | MaxFutureMoves | TrajectorySearch | PopulationSearch | Mean Seconds | Max Seconds |
|----------------|---------------:|-----------------:|-----------------:|-------------:|------------:|
| 3000–3029 | 117.77 | 125.00 | **131.10** | 8.216 | 8.627 |
| 4000–4029 | 116.27 | 123.37 | **128.87** | 8.190 | 8.264 |
| Combined | 117.02 | 124.18 | **129.98** | 8.203 | 8.627 |

Compared with MaxFutureMoves: 60 wins, +12.97 cells on average. Compared with
TrajectorySearch: 52 wins / 2 ties / 6 losses, +5.80 cells. Default 16x10,
target 10, CPU single-threaded, strategy seed 68 reset each game. These are
different runtime budgets, not equal-time superiority claims.

The kernel enumerates sum-target row intervals within column bands, scores
only legal actions, and maintains 12 candidate trajectories. Small temporary
losses are allowed in the population while preserving a separate best-ever
plan. Planning stops at an 8-second soft deadline checked every 16 rollouts;
execution and one final batch can add time. First compilation measured 4.11s
in development; cached warmups in the benchmarks took about 0.25s. Neither
is included in the episode figures. Wall-clock runs are not bitwise reproducible;
use a fixed `max_rollouts` and a nonbinding `time_budget` for deterministic tests.
The implementation uses CPU even when the environment is on CUDA.

```bash
python -m rs10env.benchmark --population --games 30 --seed 3000
python -m rs10env.benchmark --population --games 30 --seed 4000
```

Raw results: [3000–3029](docs/benchmark/population_search_3000_3029.json),
[4000–4029](docs/benchmark/population_search_4000_4029.json).
The strategy appears in the app when the optional `search` dependency is installed.

## Previous Strategy

**`trajectory_search`. Implemented by: gpt6 astra.**

Unlike independent `multi_start` rollouts, this strategy repeatedly mutates
prefixes of its best action sequence and repairs the suffix, favoring old
moves that remain legal. It retains the incumbent unless a complete replacement
clears at least as many cells. This is heuristic trajectory optimization,
not a new fundamental algorithm or a proof of optimality.

```python
strategy = create_strategy("trajectory_search", num_rollouts=128,
                           iterations=24, batch_size=64, seed=68, device="cpu")
```

Held-out seeds 2000–2029, CPU single-threaded, default board rules:

| Strategy | Mean Cleared | Seconds/Game |
|----------|-------------:|-------------:|
| max_future_moves | 109.27 | 0.974 |
| multi_start (128) | 111.33 | 1.343 |
| multi_start (512) | 113.30 | 6.157 |
| trajectory_search | **118.10** | 4.542 |

Versus `max_future_moves`: +8.83 cells (+8.08%), 26 wins / 2 ties / 2 losses,
approximate paired 95% CI [6.94, 10.72] cells, but 4.66x the runtime.
Versus 512 independent rollouts: +4.80 cells with 26% less mean runtime;
this is a measured cost comparison, not an enforced equal-time experiment.
Only 30 held-out boards were tested, with no GPU or optimality evaluation.

Run `python -m rs10env.benchmark --games 30 --seed 2000` to reproduce.
[Per-board results](docs/benchmark/trajectory_search_2000_2029.json) and
[Chinese documentation](README.zh-CN.md) include further details.

**PyPI:** [rs10env](https://pypi.org/project/rs10env/) · **GitHub:** [xx025/rs10env](https://github.com/xx025/rs10env)

## Install

By default the project uses **CPU-only PyTorch** (smaller install; no GPU/CUDA needed if you are not training):

```bash
# With uv (recommended; installs torch from PyTorch CPU index)
uv add rs10env

# Or from source: clone repo, then sync in project root
git clone https://github.com/xx025/rs10env.git
cd rs10env
uv sync
```

For **GPU/CUDA**, install the matching torch first, then rs10env, e.g.  
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`, then `uv sync`.

## Environment

RS10Env is a turn-based env: each step selects a rectangle whose cells sum to a target (default 10) and clears it. Board: `H×W` (default 16×10), cells 0–9. Full spec (observation/action space, mask, reward): **[API reference](docs/API.md)**.

## API usage

See **[docs/API.md](docs/API.md)** for environment API and code examples (single board, multi-game comparison, low-level env + strategy).

## Streamlit app

Two modes: **single board (multiple strategies)** and **multi-game comparison** (with progress). Best strategy is highlighted.

```bash
uv add rs10env[app]
rs10env-app
# or from repo root
uv run streamlit run app.py
```

## Strategies

- `random` — uniform over valid actions  
- `greedy` — clear as many cells as possible  
- `center_bias` — prefer rectangles closer to board center  
- `large_rect` / `small_rect` — prefer larger / smaller area  
- `center_small_rect` — center + small area  
- `epsilon_greedy` — ε-greedy  
- `max_future_moves` — choose action that maximizes valid moves on the next state  

## Benchmark (strategy comparison)

Batch runs on a fixed set of boards (16×10, target_sum=10). Below: 100k games per strategy. “Best count” / “Share” = times that strategy cleared the most cells in a game.

| Strategy | Tests | Avg steps | Avg removed | Time/game (s) | Best count | Share (%) |
|----------|-------|-----------|--------------|---------------|-----------|-----------|
| random | 100,000 | 40.72 | 96.82 | 0.32 | 1,109 | 1.11 |
| greedy | 100,000 | 36.38 | 92.11 | 0.31 | 244 | 0.24 |
| center_bias | 100,000 | 43.15 | 101.15 | 0.31 | 5,185 | 5.19 |
| large_rect | 100,000 | 36.33 | 91.57 | 0.31 | 212 | 0.21 |
| small_rect | 100,000 | 45.77 | 103.44 | 0.39 | 7,316 | 7.32 |
| center_small_rect | 100,000 | 46.31 | 104.65 | 0.52 | 10,643 | 10.64 |
| max_future_moves | 100,000 | 50.07 | 113.54 | 8.12 | 80,793 | 80.79 |

`max_future_moves` clears the most on average and wins most often, at higher per-game cost; `center_small_rect` and `small_rect` offer a good trade-off.

Cells removed by strategy (boxplot):

![removed_boxplot](docs/benchmark/removed_boxplot.png)

Best-strategy share (who clears the most in each game):

![best_strategy_analysis](docs/benchmark/best_strategy_analysis.png)

Cumulative average cells removed over games:

![cumulative_avg_removed](docs/benchmark/cumulative_avg_removed.png)

## Dependencies

- Python >= 3.10  
- PyTorch >= 2.0  
- Gymnasium >= 1.0  
- NumPy >= 1.24  

## Related projects

This repo focuses on **simulation environment and strategies** (Gymnasium + heuristics). Other open-source projects that implement automation or assist tools for similar number-sum-elimination mechanics (various platforms):

| Project | Platform | Description |
|---------|----------|-------------|
| [nusery (longlifedahan)](https://github.com/longlifedahan/longlifedahan.github.io/blob/master/nusery.html) | Web | HTML/JS frontend, [playable demo](https://longlifedahan.github.io/nusery.html) |
| [Opening_Nursery_For_Mac](https://github.com/guzhoudong521/Opening_Nursery_For_Mac) | macOS | Python, pyautogui + OpenCV + Tesseract |
| [nursery-bot](https://github.com/rikkayoru/nursery-bot) | Windows | Python bot, Tesseract OCR |
| [KaiJuTuoErSuo](https://github.com/hncboy/KaiJuTuoErSuo) | Android | Java + ADB, OpenCV, OCR, DFS for elimination path |
| [tuoersuo](https://gitee.com/Nidhoog/tuoersuo) | — | Automation script (Gitee) |

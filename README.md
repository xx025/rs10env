# RS10Env

**[English](README.md)** | [简体中文](README.zh-CN.md)

Gymnasium-compatible RS10 board game environment and heuristic strategies (PyTorch).

## Install

By default the project uses **CPU-only PyTorch** (smaller install; no GPU/CUDA needed if you are not training):

```bash
# With uv (recommended; installs torch from PyTorch CPU index)
uv add rs10env

# Or from source
uv sync
```

For **GPU/CUDA**, install the matching torch first, then rs10env, e.g.  
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`, then `uv sync`.

## Environment (RS10Env)

RS10Env is a **Gymnasium**-compatible turn-based environment: on a fixed-size board, each step selects a rectangle whose cells sum to a target (default 10) and clears it, until no valid move remains.

### Board and parameters

- **Board**: `H × W` grid (default 16×10). Each cell is an integer **0–9** (0 = empty, 1–9 = value).
- **Parameters**: `H`, `W`, `target_sum` (default 10), `device` (`"cpu"` / `"cuda"`).

### Observation space

- **Type**: `gymnasium.spaces.Box`
- **Shape**: `(10, H, W)`, `dtype=np.float32`, values in `[0, 1]`.
- **Meaning**: One-hot encoding of the board; channel `k` indicates whether the cell value is `k` (0 = empty, 1–9 = digits).

### Action space

- **Type**: `gymnasium.spaces.Discrete(N)` with `N` = number of valid rectangles.
- **Meaning**: Each action is one axis-aligned rectangle (all cells in `[r1..r2] × [c1..c2]`), with area ≥ 2.

### Valid actions (mask)

An action is **valid** only if:

1. The sum of cell values in that rectangle equals `target_sum`.
2. **Diagonal rule**: at least one diagonal pair of the rectangle’s corners has both cells non-zero.

The env provides `info["action_mask"]` (boolean, shape `(N,)`) for the current step.

### Transition and reward

- **Transition**: Applying action `a` sets all cells in `rect_masks[a]` to 0.
- **Reward (step)**: (number of cells cleared this step) / (H×W).
- **Termination bonus**: If the episode ends with no valid moves, add (total cells cleared this episode) / (H×W) to the final reward.
- **terminated**: no valid action; **truncated**: step count ≥ `max_steps = (H*W)//2`.

### API summary

- **`reset(seed=None, options=None, **kwargs)`**  
  Use `board=...` (numpy or torch, shape `(H,W)`) to set the board, or a random 1–9 board is generated.  
  Returns `(obs, info)` with `info["step"]`, `info["action_mask"]`.
- **`step(action, **kwargs)`**  
  Returns `(obs, reward, terminated, truncated, info)` with `info["step"]`, `info["total_zeros"]`, `info["action_mask"]`.

## API usage

### Single board, multiple strategies

```python
import numpy as np
from rs10env import run_strategies_on_board, STRATEGY_NAMES

board = np.random.randint(1, 10, size=(16, 10), dtype=np.int32)
results = run_strategies_on_board(
    board=board,
    strategy_names=["random", "greedy", "max_future_moves"],
    device="cpu",
    H=16,
    W=10,
    target_sum=10,
)
for r in results:
    mark = " ★ best" if r["is_best"] else ""
    print(r["strategy_name"], r["total_reward"], r["steps"], r["total_cleared"], mark)
```

### Multi-game comparison

```python
from rs10env import run_strategies

def on_progress(current, total, message):
    print(f"[{current}/{total}] {message}")

summary = run_strategies(
    strategy_names=["random", "greedy", "center_bias"],
    num_games=20,
    base_seed=42,
    device="cpu",
    progress_callback=on_progress,
)
for s in summary:
    print(s["strategy_name"], s["avg_cleared"], s["avg_time_sec"], "★ best" if s["is_best"] else "")
```

### Low-level: env + strategy step-by-step

```python
from rs10env import RS10Env, create_strategy

env = RS10Env(device="cpu", H=16, W=10, target_sum=10)
obs, info = env.reset()
mask = info["action_mask"]
strategy = create_strategy("greedy")
action = strategy.get_action(env, mask)
obs, reward, terminated, truncated, info = env.step(action.item())
```

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

Batch runs on a fixed set of boards (16×10, target_sum=10). Below: ~100k games per strategy (50k for `max_future_moves`).

**Summary (avg steps, avg cells removed, time per game):**

| Strategy | Tests | Avg steps | Avg removed | Time/game (s) |
|----------|-------|-----------|--------------|---------------|
| random | 100,000 | 40.72 | 96.82 | 0.32 |
| greedy | 100,000 | 36.38 | 92.11 | 0.31 |
| center_bias | 100,000 | 43.15 | 101.15 | 0.31 |
| large_rect | 100,000 | 36.33 | 91.57 | 0.31 |
| small_rect | 100,000 | 45.77 | 103.44 | 0.39 |
| center_small_rect | 100,000 | 46.31 | 104.65 | 0.52 |
| max_future_moves | 50,000 | 50.09 | 113.58 | 8.14 |

**Best-strategy share** (per game, the strategy that cleared the most cells):

| Strategy | Best count | Share (%) |
|----------|------------|-----------|
| max_future_moves | 40,525 | 40.53 |
| center_small_rect | 26,171 | 26.17 |
| small_rect | 20,507 | 20.51 |
| center_bias | 13,766 | 13.77 |
| random | 4,341 | 4.34 |
| greedy | 1,121 | 1.12 |
| large_rect | 1,023 | 1.02 |

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

## Publishing to PyPI

The repo uses **PyPI Trusted Publishing (OIDC)**; no API token is stored. To trigger a publish:

1. **From a release**: On GitHub, go to **Releases → Create a new release**, choose a tag (e.g. `v0.1.0`), publish. The workflow runs on `release: published`.
2. **Manual run**: Go to **Actions → Publish to PyPI**, click **Run workflow**, then **Run workflow**. Builds and uploads to PyPI using OIDC.

## Related projects

This repo focuses on **simulation environment and strategies** (Gymnasium + heuristics). Other open-source projects that implement automation or assist tools for similar number-sum-elimination mechanics (various platforms):

| Project | Platform | Description |
|---------|----------|-------------|
| [Opening_Nursery_For_Mac](https://github.com/guzhoudong521/Opening_Nursery_For_Mac) | macOS | Python, pyautogui + OpenCV + Tesseract |
| [nursery-bot](https://github.com/rikkayoru/nursery-bot) | Windows | Python bot, Tesseract OCR |
| [KaiJuTuoErSuo](https://github.com/hncboy/KaiJuTuoErSuo) | Android | Java + ADB, OpenCV, OCR, DFS for elimination path |
| [tuoersuo](https://gitee.com/Nidhoog/tuoersuo) | — | Automation script (Gitee) |

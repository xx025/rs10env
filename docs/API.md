# API reference

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

---

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

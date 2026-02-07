# RS10Env

[English](README.md) | **简体中文**

基于 Gymnasium 的 RS10 棋盘环境与启发式策略（PyTorch）。

## 安装

默认使用 **仅 CPU 的 PyTorch**（体积更小；若不训练可不装 GPU/CUDA）：

```bash
# 使用 uv（推荐；从 PyTorch CPU 源安装 torch）
uv add rs10env

# 或从源码安装
uv sync
```

需要 **GPU/CUDA** 时，先安装对应版本的 torch，再安装 rs10env，例如：  
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`，然后 `uv sync`。

## 环境（RS10Env）

RS10Env 是 **Gymnasium** 兼容的回合制环境：在固定大小棋盘上，每步选择一个单元格和为目标值（默认 10）的矩形并清除，直到没有合法动作为止。

### 棋盘与参数

- **棋盘**：`H × W` 网格（默认 16×10）。每格为 **0–9** 的整数（0=空，1–9=数值）。
- **参数**：`H`、`W`、`target_sum`（默认 10）、`device`（`"cpu"` / `"cuda"`）。

### 观测空间

- **类型**：`gymnasium.spaces.Box`
- **形状**：`(10, H, W)`，`dtype=np.float32`，取值 `[0, 1]`。
- **含义**：棋盘 one-hot 编码；通道 `k` 表示该格是否为数值 `k`（0=空，1–9=数字）。

### 动作空间

- **类型**：`gymnasium.spaces.Discrete(N)`，`N` 为合法矩形数量。
- **含义**：每个动作为一个轴对齐矩形（`[r1..r2] × [c1..c2]` 内所有格），面积 ≥ 2。

### 合法动作（mask）

动作 **合法** 当且仅当：

1. 该矩形内格子和等于 `target_sum`。
2. **对角线规则**：矩形至少有一对对角格都非零。

环境在每步的 `info["action_mask"]` 中提供布尔 mask，形状 `(N,)`。

### 转移与奖励

- **转移**：执行动作 `a` 将 `rect_masks[a]` 对应格置为 0。
- **单步奖励**：(本步清除格数) / (H×W)。
- **结束奖励**：若因无合法动作结束，在最终奖励上再加 (本局累计清除格数) / (H×W)。
- **terminated**：无合法动作；**truncated**：步数 ≥ `max_steps = (H*W)//2`。

### API 概要

- **`reset(seed=None, options=None, **kwargs)`**  
  可用 `board=...`（numpy 或 torch，形状 `(H,W)`）指定棋盘，否则随机生成 1–9 棋盘。  
  返回 `(obs, info)`，含 `info["step"]`、`info["action_mask"]`。
- **`step(action, **kwargs)`**  
  返回 `(obs, reward, terminated, truncated, info)`，含 `info["step"]`、`info["total_zeros"]`、`info["action_mask"]`。

## API 使用

### 单棋盘多策略

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

### 多局对比

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

### 底层：环境 + 策略逐步

```python
from rs10env import RS10Env, create_strategy

env = RS10Env(device="cpu", H=16, W=10, target_sum=10)
obs, info = env.reset()
mask = info["action_mask"]
strategy = create_strategy("greedy")
action = strategy.get_action(env, mask)
obs, reward, terminated, truncated, info = env.step(action.item())
```

## Streamlit 应用

两种模式：**单棋盘多策略** 与 **多局对比**（带进度）。会标出最佳策略。

```bash
uv add rs10env[app]
rs10env-app
# 或在仓库根目录
uv run streamlit run app.py
```

## 策略

- `random` — 在合法动作上均匀随机  
- `greedy` — 尽量多清格  
- `center_bias` — 偏向靠近中心的矩形  
- `large_rect` / `small_rect` — 偏向面积大 / 小  
- `center_small_rect` — 中心 + 小面积  
- `epsilon_greedy` — ε-贪心  
- `max_future_moves` — 选使下一步合法动作数最多的动作  

## 基准（策略对比）

在固定棋盘集上批量对局（16×10，target_sum=10）。约 10 万局/策略（`max_future_moves` 为 5 万局）。

**汇总（平均步数、平均清除格数、单局耗时）：**

| 策略 | 测试局数 | 平均步数 | 平均清除格数 | 单局耗时 (s) |
|------|----------|----------|--------------|--------------|
| random | 100,000 | 40.72 | 96.82 | 0.32 |
| greedy | 100,000 | 36.38 | 92.11 | 0.31 |
| center_bias | 100,000 | 43.15 | 101.15 | 0.31 |
| large_rect | 100,000 | 36.33 | 91.57 | 0.31 |
| small_rect | 100,000 | 45.77 | 103.44 | 0.39 |
| center_small_rect | 100,000 | 46.31 | 104.65 | 0.52 |
| max_future_moves | 50,000 | 50.09 | 113.58 | 8.14 |

**最佳策略占比**（该局清除格数最多的策略）：

| 策略 | 最佳次数 | 占比 (%) |
|------|----------|----------|
| max_future_moves | 40,525 | 40.53 |
| center_small_rect | 26,171 | 26.17 |
| small_rect | 20,507 | 20.51 |
| center_bias | 13,766 | 13.77 |
| random | 4,341 | 4.34 |
| greedy | 1,121 | 1.12 |
| large_rect | 1,023 | 1.02 |

`max_future_moves` 平均清除最多、胜出局数最多，但单局耗时较高；`center_small_rect`、`small_rect` 在效果与耗时之间较均衡。

各策略清除格数分布（箱线图）：

![removed_boxplot](docs/benchmark/removed_boxplot.png)

最佳策略占比（每局清除最多者）：

![best_strategy_analysis](docs/benchmark/best_strategy_analysis.png)

累计平均清除格数随对局数变化：

![cumulative_avg_removed](docs/benchmark/cumulative_avg_removed.png)

## 依赖

- Python >= 3.10  
- PyTorch >= 2.0  
- Gymnasium >= 1.0  
- NumPy >= 1.24  

## 发布到 PyPI

本仓库使用 **PyPI Trusted Publishing (OIDC)**，无需保存 API Token。触发发布方式：

1. **通过 Release**：在 GitHub 进入 **Releases → Create a new release**，选择或新建标签（如 `v0.1.0`），发布。工作流在 `release: published` 时自动运行。
2. **手动运行**：进入 **Actions → Publish to PyPI**，点击 **Run workflow** → **Run workflow**。会先构建再通过 OIDC 上传到 PyPI。

## 相关项目

本仓库侧重 **仿真环境与策略**（Gymnasium + 启发式）。以下为同样针对「数和消除」类玩法的其他开源实现（自动化/辅助，多平台），仅供参考：

| 项目 | 平台 | 说明 |
|------|------|------|
| [Opening_Nursery_For_Mac](https://github.com/guzhoudong521/Opening_Nursery_For_Mac) | macOS | Python + pyautogui + OpenCV + Tesseract |
| [nursery-bot](https://github.com/rikkayoru/nursery-bot) | Windows | Python bot，Tesseract OCR |
| [KaiJuTuoErSuo](https://github.com/hncboy/KaiJuTuoErSuo) | 安卓 | Java + ADB，OpenCV、OCR，DFS 消除路径 |
| [tuoersuo](https://gitee.com/Nidhoog/tuoersuo) | — | 辅助脚本（Gitee） |

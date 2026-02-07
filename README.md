# RS10Env

RS10 棋盘游戏环境（Gymnasium 兼容）与启发式策略，纯 PyTorch 实现。

## 安装

默认使用 **CPU 版 PyTorch**（体积小，不训练时无需 GPU/CUDA）：

```bash
# 使用 uv（推荐，会从 PyTorch CPU 源安装 torch）
uv add rs10env

# 或从源码
uv sync
```

若需 **GPU/CUDA**，先安装对应版本 torch 再装 rs10env，例如：
`uv pip install torch --index-url https://download.pytorch.org/whl/cu124`，再 `uv sync`。

## 环境定义（RS10Env）

RS10Env 是 **Gymnasium** 兼容的回合制环境：在固定大小的棋盘上，每一步选择一个「和为 10」的矩形并消除，直到无合法动作为止。

### 棋盘与参数

- **棋盘**：`H × W` 网格（默认 16×10），每格一个 **0–9** 的整数。
  - **0** 表示空格（已消除），**1–9** 表示未消除的数字。
- **可配置参数**：`H`（高）、`W`（宽）、`target_sum`（目标和，默认 10）、`device`（`"cpu"` / `"cuda"`）。

### 状态空间（观测）

- **类型**：`gymnasium.spaces.Box`
- **形状**：`(10, H, W)`，`dtype=np.float32`，取值 `[0, 1]`。
- **含义**：对棋盘做 **one-hot**：第 `k` 个通道表示「该格数字是否为 `k`」（0 对应空格，1–9 对应数字 1–9）。即 `obs[c, i, j] = 1` 表示格子 `(i, j)` 的值为 `c`。

### 动作空间

- **类型**：`gymnasium.spaces.Discrete(N)`，`N` 为所有合法矩形的数量。
- **含义**：每个动作对应棋盘上的一个 **矩形区域**（由左上、右下行列确定）。矩形满足：
  - 面积 ≥ 2（至少 2 格）；
  - 由所有满足 `r1 ≤ r ≤ r2` 且 `c1 ≤ c ≤ c2` 的格子 `(r, c)` 组成。
- 动作编号与内部 `all_rects` 的顺序一致，可通过环境的 `all_rects`、`rect_masks` 等属性与具体矩形对应。

### 有效动作（合法动作掩码）

并非所有动作在当前状态下都合法。**合法**需同时满足：

1. **矩形内数字之和等于 `target_sum`**（默认 10）。
2. **对角线约束**：矩形四个角中，至少有一对「对角线」上的两个角格均为非零（即不能选四个角里有两个是 0 的矩形，保证可消除性）。

环境在 `info["action_mask"]` 中提供当前步的合法动作布尔掩码，形状 `(N,)`，便于 Maskable PPO 等算法使用。

### 状态转移

执行动作 `a` 时：

1. 取该动作对应的矩形掩码 `rect_masks[a]`。
2. 将该矩形内所有格子置为 **0**（同时更新内部 2D/3D 棋盘表示）。
3. 若当前已无合法动作，本局 **终止**（terminated）；若步数达到 `max_steps = (H*W)//2`，则 **截断**（truncated）。

### 奖励

- **每步奖励**：本步消除的格子数（即该矩形内原先非零的格数）除以棋盘总面积 `H*W`，即  
  `reward_step = 本步消除格数 / (H*W)`。
- **终止加成**：若因「无合法动作」而终止，额外加上  
  `bonus = 本局累计消除格数 / (H*W)`，即  
  `reward_total = reward_step + bonus`。  
  若因步数截断而结束，则无此加成。

### 终止与截断

- **terminated**：当前无任何合法动作（`action_mask` 全为 False）。
- **truncated**：步数达到 `max_steps = (H*W)//2`（每局步数上界）。

### 接口摘要

- **`reset(seed=None, options=None, **kwargs)`**  
  - 若传入 `board=...`（numpy 或 torch，形状 `(H,W)`），则用该棋盘初始化；否则随机生成 1–9 的棋盘。  
  - 返回 `(obs, info)`，`info` 含 `step`, `action_mask`。
- **`step(action, **kwargs)`**  
  - 返回 `(obs, reward, terminated, truncated, info)`，`info` 含 `step`, `total_zeros`, `action_mask`。

以上即 RS10Env 的完整环境定义，与代码实现一致。

## API 调用示例

### 单棋盘多策略：同一棋盘上跑多条策略，拿结果并比最优

```python
import numpy as np
from rs10env import run_strategies_on_board, STRATEGY_NAMES

# 棋盘：16×10，每格 0–9（0 表示空）
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
    mark = " ★ 最优" if r["is_best"] else ""
    print(f"{r['strategy_name']}: 奖励={r['total_reward']:.4f}, 步数={r['steps']}, 消除={r['total_cleared']}{mark}")
```

### 多局对比：每条策略跑多局，按平均表现比最优

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
    mark = " ★ 最优" if s["is_best"] else ""
    print(f"{s['strategy_name']}: 平均奖励={s['avg_reward']:.4f}, 平均消除={s['avg_cleared']:.2f}, 平均时间={s['avg_time_sec']:.4f}s{mark}")
```

### 底层：环境 + 策略单步控制

```python
from rs10env import RS10Env, create_strategy

env = RS10Env(device="cpu", H=16, W=10, target_sum=10)
obs, info = env.reset()
mask = info["action_mask"]

strategy = create_strategy("greedy")
action = strategy.get_action(env, mask)
obs, reward, terminated, truncated, info = env.step(action.item())
```

## Streamlit 应用（app）

两种模式（页内单选切换）：

- **单棋盘多策略**：输入或随机生成一个棋盘，多条策略在同一棋盘上各跑一局，展示总奖励、步数、消除格数，最优高亮。
- **多局对比**：设置每策略局数、种子，多条策略各跑多局（同种子保证棋盘一致），展示平均奖励、平均步数、平均消除、平均时间，最优高亮。

```bash
uv add rs10env[app]
rs10env-app
# 或从项目根
uv run streamlit run app.py
```

## 策略

- `random` - 随机
- `greedy` - 贪心（消除最多方块）
- `center_bias` - 中心偏好
- `large_rect` / `small_rect` - 大/小矩形偏好
- `center_small_rect` - 中心+小矩形
- `epsilon_greedy` - ε-贪心
- `max_future_moves` - 最大化未来可行步数

## 依赖

- Python >= 3.10
- PyTorch >= 2.0
- Gymnasium >= 1.0
- NumPy >= 1.24

## 发布到 PyPI（Trusted Publishing / OIDC）

使用 **PyPI Trusted Publishing (OIDC)**，无需在仓库中保存 API Token，由 GitHub Actions 用短期凭证发布。

### 1. 在 PyPI 添加 Trusted Publisher

1. 登录 [pypi.org](https://pypi.org)，进入你的项目 **rs10env**（若尚未创建，先随便上传一次或创建项目）。
2. 项目页左侧点击 **Publishing**，在 “Publishing” 配置页添加 **Trusted publisher**。
3. 选择 **GitHub Actions**，填写：
   - **Owner**：你的 GitHub 用户名或组织名
   - **Repository name**：`rs10env`
   - **Workflow name**：`publish.yml`
   - **Environment name**（可选）：留空即可；若填写（如 `pypi`），需在仓库 **Settings → Environments** 中创建同名 environment，并可配置审批等。
4. 保存后，该 workflow 即可向该 PyPI 项目发布。

### 2. 触发发布

- 在 GitHub 仓库创建 **Release** 并发布，或
- 在 **Actions** 页选择 **Publish to PyPI** 工作流，点击 **Run workflow**。

工作流会先构建再通过 OIDC 向 PyPI 上传，无需任何 Secret。

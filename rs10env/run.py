"""运行多策略对比：单局/多局统计；单棋盘多策略。"""
import time
from typing import List, Optional, Any, Union, Callable
import numpy as np
import torch

from rs10env.env import RS10Env
from rs10env.strategies import create_strategy, Strategy

STRATEGY_NAMES = [
    "random",
    "greedy",
    "center_bias",
    "large_rect",
    "small_rect",
    "center_small_rect",
    "epsilon_greedy",
    "max_future_moves",
]


def run_episode(
    env: RS10Env,
    strategy: Strategy,
    seed: Optional[int] = None,
) -> dict:
    """
    用指定策略跑一局，直到结束。
    返回: total_reward, steps, total_cleared (总消除格数)
    """
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    total_cleared = 0

    while True:
        mask = info["action_mask"]
        if not mask.any().item():
            break
        action = strategy.get_action(env, mask)
        if isinstance(action, torch.Tensor):
            action = action.item()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cleared = info["total_zeros"].item() if torch.is_tensor(info["total_zeros"]) else int(info["total_zeros"])
        if terminated or truncated:
            break

    return {
        "total_reward": float(total_reward),
        "steps": int(env.step_count),
        "total_cleared": int(total_cleared),
    }


def run_strategies(
    strategy_names: List[str],
    num_games: int = 10,
    device: Optional[str] = None,
    base_seed: Optional[int] = 42,
    H: int = 16,
    W: int = 10,
    target_sum: int = 10,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[dict]:
    """
    对每条策略跑 num_games 局（每局相同种子保证棋盘一致），汇总结果。
    progress_callback(current, total, message) 可选，用于显示进度。
    返回列表，每项: strategy_name, avg_reward, avg_steps, avg_cleared, avg_time_sec, is_best
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = RS10Env(device=device, H=H, W=W, target_sum=target_sum)
    strategies = {name: create_strategy(name, device=device) for name in strategy_names}

    results_per_strategy: dict[str, list[dict]] = {name: [] for name in strategy_names}
    total_steps = num_games * len(strategy_names)
    step = 0

    for game_id in range(num_games):
        seed = (base_seed + game_id) if base_seed is not None else None
        for name in strategy_names:
            t0 = time.perf_counter()
            ep = run_episode(env, strategies[name], seed=seed)
            ep["time_sec"] = time.perf_counter() - t0
            results_per_strategy[name].append(ep)
            step += 1
            if progress_callback is not None:
                progress_callback(step, total_steps, f"第 {game_id + 1}/{num_games} 局 · {name}")

    # 汇总：按平均 total_cleared 判定最优
    summary = []
    for name in strategy_names:
        episodes = results_per_strategy[name]
        avg_reward = sum(float(e["total_reward"]) for e in episodes) / len(episodes)
        avg_steps = sum(int(e["steps"]) for e in episodes) / len(episodes)
        avg_cleared = sum(int(e["total_cleared"]) for e in episodes) / len(episodes)
        avg_time_sec = sum(e["time_sec"] for e in episodes) / len(episodes)
        summary.append({
            "strategy_name": name,
            "avg_reward": round(avg_reward, 4),
            "avg_steps": round(avg_steps, 2),
            "avg_cleared": round(avg_cleared, 2),
            "avg_time_sec": round(avg_time_sec, 4),
        })

    best_avg_cleared = max(s["avg_cleared"] for s in summary)
    for s in summary:
        s["is_best"] = s["avg_cleared"] == best_avg_cleared

    return summary


def run_strategies_on_board(
    board: Union[np.ndarray, list],
    strategy_names: List[str],
    device: Optional[str] = None,
    H: int = 16,
    W: int = 10,
    target_sum: int = 10,
) -> List[dict]:
    """
    在同一棋盘上分别用多条策略各跑一局，返回每条策略的结果。
    board: 二维数组 (H, W)，每格 0–9（0 表示空）。
    返回列表，每项: strategy_name, total_reward, steps, total_cleared, is_best
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = RS10Env(device=device, H=H, W=W, target_sum=target_sum)
    strategies = {name: create_strategy(name, device=device) for name in strategy_names}

    if isinstance(board, list):
        board = np.asarray(board, dtype=np.int32)
    else:
        board = np.asarray(board, dtype=np.int32)

    if board.shape != (H, W):
        raise ValueError(f"棋盘形状须为 (H,W)=({H},{W})，当前为 {board.shape}")

    results = []
    for name in strategy_names:
        env.reset(board=board)
        ep = run_episode(env, strategies[name], seed=None)
        results.append({
            "strategy_name": name,
            "total_reward": round(float(ep["total_reward"]), 4),
            "steps": ep["steps"],
            "total_cleared": ep["total_cleared"],
        })

    best_cleared = max(r["total_cleared"] for r in results)
    for r in results:
        r["is_best"] = r["total_cleared"] == best_cleared

    return results

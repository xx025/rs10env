"""run_episode / run_strategies / run_strategies_on_board 测试。"""
import pytest

from rs10env import run_episode, run_strategies, run_strategies_on_board, STRATEGY_NAMES, create_strategy, RS10Env


def test_run_episode_returns_dict(env):
    strategy = create_strategy("random", device="cpu")
    result = run_episode(env, strategy, seed=5)
    assert set(result.keys()) == {"total_reward", "steps", "total_cleared"}
    assert result["steps"] >= 0
    assert result["total_cleared"] >= 0


def test_run_strategies_on_board():
    import numpy as np
    board = np.random.randint(1, 10, size=(16, 10), dtype=np.int32)
    results = run_strategies_on_board(
        board=board,
        strategy_names=["random", "greedy"],
        device="cpu",
        H=16,
        W=10,
        target_sum=10,
    )
    assert len(results) == 2
    for r in results:
        assert "strategy_name" in r
        assert "total_reward" in r
        assert "steps" in r
        assert "total_cleared" in r
        assert "is_best" in r
    assert sum(1 for r in results if r["is_best"]) >= 1


def test_run_strategies_summary():
    summary = run_strategies(
        strategy_names=["random", "greedy"],
        num_games=2,
        base_seed=10,
        device="cpu",
        H=16,
        W=10,
        target_sum=10,
    )
    assert len(summary) == 2
    for s in summary:
        assert "strategy_name" in s
        assert "avg_reward" in s
        assert "avg_steps" in s
        assert "avg_cleared" in s
        assert "avg_time_sec" in s
        assert "is_best" in s
    assert sum(1 for s in summary if s["is_best"]) >= 1


def test_run_strategies_with_progress_callback():
    calls = []
    run_strategies(
        strategy_names=["random"],
        num_games=2,
        base_seed=11,
        device="cpu",
        progress_callback=lambda c, t, m: calls.append((c, t, m)),
    )
    assert len(calls) == 2
    assert calls[0][1] == 2
    assert calls[-1][0] == 2


def test_run_strategies_on_board_wrong_shape():
    import numpy as np
    board = np.zeros((8, 8), dtype=np.int32)  # wrong shape for H=16, W=10
    with pytest.raises(ValueError, match="棋盘形状"):
        run_strategies_on_board(board=board, strategy_names=["random"], H=16, W=10)

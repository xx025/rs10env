"""run_episode / run_strategies / run_strategies_on_board 测试。"""
import pytest
import numpy as np

from rs10env import run_episode, run_strategies, run_strategies_on_board, STRATEGY_NAMES, create_strategy, RS10Env


def test_run_episode_returns_dict(env):
    strategy = create_strategy("random", device="cpu")
    result = run_episode(env, strategy, seed=5)
    assert set(result.keys()) == {"total_reward", "steps", "total_cleared"}
    assert result["steps"] >= 0
    assert result["total_cleared"] >= 0


def test_run_episode_with_board():
    env = RS10Env(device="cpu", H=1, W=2)
    strategy = create_strategy("greedy", device="cpu")
    board = np.array([[5, 5]], dtype=np.int32)

    result = run_episode(env, strategy, board=board)

    assert result == {"total_reward": 2.0, "steps": 1, "total_cleared": 2}
    np.testing.assert_array_equal(board, [[5, 5]])


@pytest.mark.parametrize("as_list", [False, True])
@pytest.mark.parametrize("values, reward, steps, cleared", [
    ([[5, 5]], 2.0, 1, 2),
    ([[9, 9]], 0.0, 0, 0),
])
def test_run_strategies_on_board_preserves_starting_board(
    monkeypatch, as_list, values, reward, steps, cleared
):
    board = values if as_list else np.array(values, dtype=np.int32)
    starting_boards = []
    reset = RS10Env.reset

    def record_reset(self, *args, **kwargs):
        result = reset(self, *args, **kwargs)
        starting_boards.append(self.board_2d.cpu().numpy().copy())
        return result

    monkeypatch.setattr(RS10Env, "reset", record_reset)
    results = run_strategies_on_board(
        board, ["random", "greedy"], device="cpu", H=1, W=2
    )

    # One constructor reset, followed by exactly one reset per strategy.
    assert len(starting_boards) == 3
    for starting_board in starting_boards[1:]:
        np.testing.assert_array_equal(starting_board, values)
    np.testing.assert_array_equal(board, values)
    assert results == [
        {
            "strategy_name": name,
            "total_reward": reward,
            "steps": steps,
            "total_cleared": cleared,
            "is_best": True,
        }
        for name in ["random", "greedy"]
    ]


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

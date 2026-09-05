"""策略与 create_strategy 测试。"""
import pytest
import torch
from unittest.mock import patch

from rs10env import create_strategy, STRATEGY_NAMES, RS10Env


def test_strategy_names_non_empty():
    assert len(STRATEGY_NAMES) >= 1
    assert "random" in STRATEGY_NAMES
    assert "greedy" in STRATEGY_NAMES


def test_create_strategy_random(env):
    s = create_strategy("random", device="cpu")
    assert s.name == "Random"
    mask = env.get_valid_actions_mask()
    if mask.any():
        action = s.get_action(env, mask)
        assert mask[action].item() or action in mask.nonzero(as_tuple=True)[0]


def test_create_strategy_greedy(env):
    s = create_strategy("greedy", device="cpu")
    assert s.name == "Greedy"
    env.reset(seed=7)
    mask = env.get_valid_actions_mask()
    if mask.any():
        action = s.get_action(env, mask)
        assert mask[action].item()


def test_create_strategy_unknown_raises():
    with pytest.raises(ValueError, match="Unknown strategy"):
        create_strategy("unknown_xyz", device="cpu")


def test_create_strategy_case_insensitive(env):
    s = create_strategy("RANDOM", device="cpu")
    assert s.name == "Random"


def test_all_strategy_names_create():
    for name in STRATEGY_NAMES:
        s = create_strategy(name, device="cpu")
        assert s is not None
        assert hasattr(s, "get_action")


@pytest.mark.parametrize("target", [8, 10])
def test_multi_start_legal_reproducible_and_non_mutating(target):
    env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
    strategies = [create_strategy("multi_start", num_rollouts=8, seed=12,
                                  device="cpu") for _ in range(2)]
    trajectories = []
    for strategy in strategies:
        _, info = env.reset(seed=42)
        actions = []
        while info["action_mask"].any():
            board = env.board_2d.clone()
            observation = env._board_3d.clone()
            step = env.step_count
            action = strategy.get_action(env, info["action_mask"])
            assert info["action_mask"][action]
            assert torch.equal(board, env.board_2d)
            assert torch.equal(observation, env._board_3d)
            assert env.step_count == step
            actions.append(action.item())
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        trajectories.append(actions)
    assert trajectories[0] == trajectories[1]


def test_multi_start_replans_after_reset():
    strategy = create_strategy("multi_start", num_rollouts=4, device="cpu")
    for h, w, target in [(4, 5, 10), (3, 6, 8), (4, 5, 10)]:
        env = RS10Env(H=h, W=w, target_sum=target, device="cpu")
        _, info = env.reset(seed=7)
        if info["action_mask"].any():
            assert info["action_mask"][strategy.get_action(env, info["action_mask"])]


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_multi_start_invalid_budget(count):
    with pytest.raises(ValueError, match="positive integer"):
        create_strategy("multi_start", num_rollouts=count, device="cpu")


def _trajectory_action_without_mutation(strategy, env, mask):
    board = env.board_2d.clone()
    onehot = env._board_3d.clone()
    counters = (env.step_count, int(env.total_zeros), env.is_done, env.max_steps)
    original_mask = mask.clone()
    action = strategy.get_action(env, mask)
    assert action.shape == torch.Size([])
    assert action.dtype == torch.int64
    assert action.device == mask.device
    assert torch.equal(env.board_2d, board)
    assert torch.equal(env._board_3d, onehot)
    assert (env.step_count, int(env.total_zeros), env.is_done, env.max_steps) == counters
    assert torch.equal(mask, original_mask)
    if mask.any():
        assert mask[action].item()
    return action


def _play_strategy_game(strategy, env, info):
    actions = []
    assert info["action_mask"].any()
    for _ in range(env.max_steps - env.step_count):
        action = _trajectory_action_without_mutation(strategy, env, info["action_mask"])
        actions.append(action.item())
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            assert env.is_done
            return actions
    pytest.fail("Game did not finish within the remaining horizon")


@pytest.mark.parametrize("h,w,target", [(4, 5, 10), (3, 6, 8)])
def test_trajectory_search_full_game_reproducible_and_non_mutating(h, w, target):
    trajectories = []
    final_boards = []
    for _ in range(2):
        env = RS10Env(H=h, W=w, target_sum=target, device="cpu")
        _, info = env.reset(seed=42)
        strategy = create_strategy("trajectory_search", num_rollouts=8,
                                   iterations=3, batch_size=8, seed=12, device="cpu")
        assert strategy.name == "TrajectorySearch"
        trajectories.append(_play_strategy_game(strategy, env, info))
        final_boards.append(env.board_2d.clone())
    assert trajectories[0] == trajectories[1]
    assert torch.equal(final_boards[0], final_boards[1])


@pytest.mark.parametrize("seed", [0, 7, 42])
@pytest.mark.parametrize("target", [8, 10])
def test_trajectory_search_preserves_multi_start_incumbent(seed, target):
    results = {}
    for name in ("multi_start", "trajectory_search"):
        env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
        _, info = env.reset(seed=seed)
        options = {"iterations": 3, "batch_size": 8} if name == "trajectory_search" else {}
        strategy = create_strategy(name, num_rollouts=8, seed=seed,
                                   device="cpu", **options)
        _play_strategy_game(strategy, env, info)
        results[name] = int(env.total_zeros)
        assert results[name] == int((env.board_2d == 0).sum())
        if name == "trajectory_search":
            scores = strategy.search_scores
            assert 2 <= len(scores) <= 4
            assert scores[0] == 20 - results["multi_start"]
            assert all(after <= before for before, after in zip(scores, scores[1:]))
            assert scores[-1] == 20 - results[name]
    assert results["trajectory_search"] >= results["multi_start"]


def test_trajectory_search_replans_after_reset():
    strategy = create_strategy("trajectory_search", num_rollouts=8,
                               iterations=3, batch_size=8, seed=12, device="cpu")
    env = RS10Env(H=4, W=5, device="cpu")
    with patch.object(strategy, "_improve_plan", wraps=strategy._improve_plan) as improve:
        _, info = env.reset(seed=42)
        action = _trajectory_action_without_mutation(strategy, env, info["action_mask"])
        assert strategy._plan
        _, _, terminated, truncated, info = env.step(action)
        assert not (terminated or truncated)
        _trajectory_action_without_mutation(strategy, env, info["action_mask"])
        assert improve.call_count == 1

        # Reset the same env while a cached continuation still exists.
        _, info = env.reset(seed=7)
        _trajectory_action_without_mutation(strategy, env, info["action_mask"])
        assert improve.call_count == 2

        env = RS10Env(H=3, W=6, target_sum=8, device="cpu")
        _, info = env.reset(seed=42)
        _play_strategy_game(strategy, env, info)
        assert improve.call_count == 3


@pytest.mark.parametrize("name", ["iterations", "batch_size"])
@pytest.mark.parametrize("value", [0, -1, 1.5, "3", None])
def test_trajectory_search_invalid_search_budget(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        create_strategy("trajectory_search", device="cpu", **{name: value})


def test_trajectory_search_respects_remaining_horizon():
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(seed=42)
    env.max_steps = 2
    action = info["action_mask"].nonzero()[0, 0]
    _, _, terminated, truncated, info = env.step(action)
    assert not (terminated or truncated)
    strategy = create_strategy("trajectory_search", num_rollouts=8,
                               iterations=3, batch_size=8, seed=12, device="cpu")
    action = _trajectory_action_without_mutation(strategy, env, info["action_mask"])
    assert strategy._plan == []
    _, _, _, truncated, _ = env.step(action)
    assert truncated
    assert env.step_count == 2
    assert strategy.search_scores[-1] == int(torch.count_nonzero(env.board_2d))


@pytest.mark.parametrize("has_move", [False, True])
def test_trajectory_search_sparse_board_and_no_moves(has_move):
    env = RS10Env(H=4, W=5, device="cpu")
    board = torch.zeros((4, 5), dtype=torch.int32)
    board[1, 1] = 4
    if has_move:
        board[2, 3] = 6
    _, info = env.reset(board=board)
    strategy = create_strategy("trajectory_search", num_rollouts=8,
                               iterations=3, batch_size=8, seed=12, device="cpu")
    assert bool(info["action_mask"].any()) == has_move
    action = _trajectory_action_without_mutation(strategy, env, info["action_mask"])
    if has_move:
        _, _, terminated, _, info = env.step(action)
        assert terminated
        assert torch.count_nonzero(env.board_2d) == 0
        assert strategy.search_scores == [0]
        action = _trajectory_action_without_mutation(strategy, env, info["action_mask"])
    assert action.item() == 0
    assert not info["action_mask"].any()


def test_future_moves_matches_reference():
    strategy = create_strategy("max_future_moves", device="cpu")
    for h, w, target in [(4, 5, 10), (3, 6, 8)]:
        env = RS10Env(H=h, W=w, target_sum=target, device="cpu")
        env.reset(seed=42)
        mask = env.get_valid_actions_mask()
        board = env.board_2d.clone()
        counts = {}
        for action in torch.where(mask)[0]:
            env.board_2d.copy_(board)
            env.board_2d[env.rect_masks[action]] = 0
            counts[action.item()] = env.get_valid_actions_mask().sum().item()
        env.board_2d.copy_(board)
        if counts:
            action = strategy.get_action(env, mask)
            assert counts[action.item()] == max(counts.values())
            assert torch.equal(env.board_2d, board)

"""策略与 create_strategy 测试。"""
import pytest
import torch

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

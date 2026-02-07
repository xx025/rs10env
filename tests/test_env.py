"""RS10Env 环境测试。"""
import numpy as np
import pytest
import torch

from rs10env import RS10Env


def test_env_creation(env):
    assert env.H == 16
    assert env.W == 10
    assert env.target_sum == 10
    assert env.board_2d is not None
    assert env.board_2d.shape == (16, 10)
    assert env.observation_space.shape == (10, 16, 10)
    assert env.action_space.n == len(env.all_rects)


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=42)
    assert obs.shape == (10, 16, 10)
    assert "step" in info
    assert "action_mask" in info
    assert info["action_mask"].shape == (len(env.all_rects),)
    assert info["step"] == 0


def test_reset_with_board(env):
    board = np.random.randint(1, 10, size=(16, 10), dtype=np.int32)
    obs, info = env.reset(board=board)
    assert obs.shape == (10, 16, 10)
    assert env.board_2d is not None
    np.testing.assert_array_equal(env.board_2d.cpu().numpy(), board)


def test_step(env):
    env.reset(seed=1)
    mask = env.get_valid_actions_mask()
    valid_indices = mask.nonzero(as_tuple=True)[0]
    if valid_indices.numel() == 0:
        pytest.skip("no valid actions for this seed")
    action = valid_indices[0].item()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (10, 16, 10)
    assert reward is not None and (isinstance(reward, (int, float)) or torch.is_tensor(reward))
    assert "action_mask" in info
    assert "total_zeros" in info
    assert env.step_count == 1


def test_run_episode_until_done(env):
    from rs10env import create_strategy, run_episode
    strategy = create_strategy("greedy", device="cpu")
    result = run_episode(env, strategy, seed=2)
    assert "total_reward" in result
    assert "steps" in result
    assert "total_cleared" in result
    assert result["steps"] >= 0
    assert result["total_cleared"] >= 0


def test_valid_actions_mask_shape(env):
    env.reset(seed=3)
    mask = env.get_valid_actions_mask()
    assert mask.shape == (len(env.all_rects),)
    assert mask.dtype == torch.bool

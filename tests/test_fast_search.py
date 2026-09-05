"""Differential and end-to-end tests for the optional compiled search."""
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from rs10env import RS10Env
from rs10env import fast_search
from rs10env.fast_search import PopulationSearchStrategy, legal_actions


@pytest.fixture(scope="module", autouse=True)
def warmup():
    PopulationSearchStrategy.warmup()


@pytest.fixture
def strategy():
    return PopulationSearchStrategy(max_rollouts=64, time_budget=60, seed=12,
                                    device="cpu")


def _lookup(env):
    lookup = np.full((env.H, env.W, env.H, env.W), -1, dtype=np.int32)
    for action, rect in enumerate(env.all_rects.tolist()):
        lookup[tuple(rect)] = action
    return lookup


def _assert_legal_actions(env, lookup, board):
    env.reset(board=board)
    original = board.copy()
    original_lookup = lookup.copy()
    actual = legal_actions(board, lookup, env.target_sum)
    expected = env.get_valid_actions_mask().nonzero(as_tuple=True)[0].numpy()
    np.testing.assert_array_equal(np.sort(actual), expected)
    np.testing.assert_array_equal(board, original)
    np.testing.assert_array_equal(lookup, original_lookup)
    assert actual.dtype == np.int32
    assert len(actual) == len(np.unique(actual))
    if len(actual):
        rects = env.all_rects[torch.from_numpy(actual).long()]
        assert torch.all((rects[:, 2] != rects[:, 0]) | (rects[:, 3] != rects[:, 1]))


@pytest.mark.parametrize("target", [8, 10])
def test_legal_actions_exhaustive_small_boards(target):
    env = RS10Env(H=2, W=2, target_sum=target, device="cpu")
    lookup = _lookup(env)
    for values in product([0, 1, target // 2, target - 1], repeat=4):
        _assert_legal_actions(env, lookup, np.array(values, dtype=np.int32).reshape(2, 2))


@pytest.mark.parametrize("target", [8, 10])
@pytest.mark.parametrize("h,w", [(h, w) for h in range(1, 6)
                                 for w in range(1, 6) if h * w > 1])
def test_legal_actions_random_sparse_boards(h, w, target):
    env = RS10Env(H=h, W=w, target_sum=target, device="cpu")
    lookup = _lookup(env)
    rng = np.random.default_rng(1000 * target + 10 * h + w)
    for sparsity in [0, 0.25, 0.5, 0.75, 0.95, 1]:
        for _ in range(8):
            board = rng.integers(1, 10, size=(h, w), dtype=np.int32)
            board[rng.random((h, w)) < sparsity] = 0
            _assert_legal_actions(env, lookup, board)


def test_legal_actions_excludes_target_valued_singletons():
    env = RS10Env(H=2, W=3, target_sum=8, device="cpu")
    board = np.array([[0, 8, 0], [0, 0, 0]], dtype=np.int32)
    _assert_legal_actions(env, _lookup(env), board)
    assert not env.get_valid_actions_mask().any()


def _action_without_mutation(strategy, env, mask):
    board = env.board_2d.clone()
    observation = env._board_3d.clone()
    original_mask = mask.clone()
    counters = (env.step_count, int(env.total_zeros), env.is_done, env.max_steps)
    action = strategy.get_action(env, mask)
    assert action.shape == torch.Size([])
    assert action.dtype == torch.int64
    assert action.device == mask.device
    assert torch.equal(board, env.board_2d)
    assert torch.equal(observation, env._board_3d)
    assert torch.equal(mask, original_mask)
    assert counters == (env.step_count, int(env.total_zeros), env.is_done, env.max_steps)
    if mask.any():
        assert mask[action].item()
    return action


@pytest.mark.parametrize("target", [8, 10])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_full_games_legal_nonmutating_reproducible_and_best_score(target, seed):
    games = []
    for _ in range(2):
        env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
        _, info = env.reset(seed=seed)
        strategy = PopulationSearchStrategy(max_rollouts=64, time_budget=60,
                                            seed=12, device="cpu")
        actions = []
        assert info["action_mask"].any()
        for step in range(env.max_steps):
            action = _action_without_mutation(strategy, env, info["action_mask"])
            if step == 0:
                scores = strategy.search_scores
                assert scores[0] == 20
                assert 16 <= strategy.rollouts <= 64
                assert strategy.rollouts % 16 == 0
                assert len(scores) == strategy.rollouts // 16 + 1
                assert all(after <= before for before, after in zip(scores, scores[1:]))
                assert strategy.rollouts == 64 or scores[-1] == 0
            else:
                assert strategy.search_scores is scores
            actions.append(action.item())
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert env.is_done
        assert strategy._plan == []
        remaining = int(torch.count_nonzero(env.board_2d))
        assert scores[-1] == remaining == 20 - int(env.total_zeros)
        games.append((actions, env.board_2d.tolist(), scores))
    assert games[0] == games[1]


@pytest.mark.parametrize("change", ["reset", "shape", "target", "horizon", "mask"])
def test_cached_plan_reused_and_invalidated(strategy, change):
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
    with patch.object(fast_search, "search_batch", wraps=fast_search.search_batch) as search:
        action = _action_without_mutation(strategy, env, info["action_mask"])
        assert strategy._plan
        calls = search.call_count
        _, _, _, _, info = env.step(action)
        action = _action_without_mutation(strategy, env, info["action_mask"])
        assert search.call_count == calls
        _, _, _, _, info = env.step(action)
        assert strategy._plan

        if change == "reset":
            _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
        elif change == "shape":
            env = RS10Env(H=5, W=4, device="cpu")
            _, info = env.reset(board=np.full((5, 4), 5, dtype=np.int32))
        elif change == "target":
            # Keep the expected board and horizon, changing only the target.
            env.target_sum = 20
            info["action_mask"] = env.get_valid_actions_mask()
        elif change == "horizon":
            env.max_steps = env.step_count + 1
        else:
            info["action_mask"] = info["action_mask"].clone()
            info["action_mask"][strategy._plan[0]] = False
        assert info["action_mask"].any()
        _action_without_mutation(strategy, env, info["action_mask"])
        assert search.call_count > calls


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_time_budget(value):
    with pytest.raises(ValueError, match="time_budget must be positive and finite"):
        PopulationSearchStrategy(time_budget=value, device="cpu")


@pytest.mark.parametrize("name", ["population_size", "max_rollouts"])
@pytest.mark.parametrize("value", [0, -1, 1.5, "3", None, True, False])
def test_invalid_integer_parameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        PopulationSearchStrategy(device="cpu", **{name: value})


def test_remaining_horizon_and_exhausted_horizon(strategy):
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
    env.max_steps = 2
    _, _, _, _, info = env.step(info["action_mask"].nonzero()[0, 0])
    action = _action_without_mutation(strategy, env, info["action_mask"])
    assert strategy._plan == []
    assert strategy.rollouts == 64
    _, _, terminated, truncated, info = env.step(action)
    assert truncated and not terminated
    assert env.step_count == 2
    assert strategy.search_scores[-1] == int(torch.count_nonzero(env.board_2d))
    with pytest.raises(ValueError, match="after the episode horizon"):
        strategy.get_action(env, info["action_mask"])


@pytest.mark.parametrize("has_move", [False, True])
def test_sparse_board_and_no_moves(strategy, has_move):
    env = RS10Env(H=4, W=5, device="cpu")
    board = np.zeros((4, 5), dtype=np.int32)
    board[1, 1] = 4
    if has_move:
        board[2, 3] = 6
    _, info = env.reset(board=board)
    assert bool(info["action_mask"].any()) == has_move
    action = _action_without_mutation(strategy, env, info["action_mask"])
    if has_move:
        _, _, terminated, _, info = env.step(action)
        assert terminated
        assert strategy.search_scores == [2, 0]
        assert int(torch.count_nonzero(env.board_2d)) == 0
        action = _action_without_mutation(strategy, env, info["action_mask"])
    assert action.item() == 0
    assert not info["action_mask"].any()

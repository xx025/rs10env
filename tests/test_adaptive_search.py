"""Fixed-work tests for the optional compiled adaptive search."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from rs10env import RS10Env, adaptive_search
from rs10env.adaptive_search import AdaptiveSearchStrategy, _adapt


@pytest.fixture(params=[1, 2], ids=["level1", "level2"])
def strategy(request):
    return AdaptiveSearchStrategy(
        levels=request.param, max_rollouts=16, adaptation_steps=4,
        outer_steps=2, time_budget=1e-9, seed=12, device="cpu",
    )


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


@pytest.mark.parametrize("target,shape", [(10, (4, 5)), (8, (3, 6))])
@pytest.mark.parametrize("levels", [1, 2])
def test_full_game_legal_nonmutating_reproducible(target, shape, levels):
    games = []
    for _ in range(2):
        env = RS10Env(H=shape[0], W=shape[1], target_sum=target, device="cpu")
        _, info = env.reset(board=np.full(shape, target // 2, dtype=np.int32))
        strategy = AdaptiveSearchStrategy(
            levels=levels, max_rollouts=16, adaptation_steps=4,
            outer_steps=2, time_budget=1e-9, seed=12, device="cpu",
        )
        actions = []
        for step in range(env.max_steps):
            assert info["action_mask"].any()
            action = _action_without_mutation(strategy, env, info["action_mask"])
            if step == 0:
                scores = strategy.search_scores
                assert strategy.rollouts == 16
                assert len(scores) == 5
                assert scores[0] == shape[0] * shape[1]
                assert all(after <= before for before, after in zip(scores, scores[1:]))
            else:
                assert strategy.search_scores is scores
            actions.append(action.item())
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert env.is_done
        assert strategy._plan == []
        assert scores[-1] == int(torch.count_nonzero(env.board_2d))
        games.append((actions, env.board_2d.tolist(), scores.copy()))
    assert games[0] == games[1]


@pytest.mark.parametrize("change", ["reset", "shape", "target", "horizon", "mask"])
def test_cache_reuse_and_invalidation(strategy, change):
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
    with patch.object(adaptive_search, "_adaptive_batch",
                      wraps=adaptive_search._adaptive_batch) as search:
        action = _action_without_mutation(strategy, env, info["action_mask"])
        assert strategy._plan
        calls = search.call_count
        assert calls == 4
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
            env.target_sum = 20
            info["action_mask"] = env.get_valid_actions_mask()
        elif change == "horizon":
            env.max_steps = env.step_count + 1
        else:
            info["action_mask"] = info["action_mask"].clone()
            info["action_mask"][strategy._plan[0]] = False
        assert info["action_mask"].any()
        _action_without_mutation(strategy, env, info["action_mask"])
        assert search.call_count == calls + 4


def test_remaining_and_exhausted_horizon(strategy):
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
    env.max_steps = 2
    _, _, _, _, info = env.step(info["action_mask"].nonzero()[0, 0])
    action = _action_without_mutation(strategy, env, info["action_mask"])
    assert strategy._plan == []
    assert strategy.rollouts == 16
    _, _, terminated, truncated, info = env.step(action)
    assert truncated and not terminated
    assert env.step_count == 2
    assert strategy.search_scores[-1] == int(torch.count_nonzero(env.board_2d))
    with pytest.raises(ValueError, match="after the episode horizon"):
        strategy.get_action(env, info["action_mask"])


@pytest.mark.parametrize("rollouts", [1, 5, 16])
def test_fixed_budget_includes_partial_batches_and_ignores_solved_board(strategy, rollouts):
    strategy.max_rollouts = rollouts
    env = RS10Env(H=1, W=2, device="cpu")
    _, info = env.reset(board=np.array([[5, 5]], dtype=np.int32))
    action = _action_without_mutation(strategy, env, info["action_mask"])
    assert strategy.rollouts == rollouts
    assert strategy.search_scores == [2] + [0] * ((rollouts + 3) // 4)
    _, _, terminated, _, info = env.step(action)
    assert terminated
    action = _action_without_mutation(strategy, env, info["action_mask"])
    assert action.item() == 0
    assert strategy._plan == []
    assert strategy._expected is None
    assert strategy._context is None


@pytest.mark.parametrize("name", ["time_budget", "learning_rate"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_invalid_positive_finite_parameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        AdaptiveSearchStrategy(device="cpu", **{name: value})


@pytest.mark.parametrize("name", ["adaptation_steps", "outer_steps", "max_rollouts"])
@pytest.mark.parametrize("value", [0, -1, 1.5, "3", True, False])
def test_invalid_integer_parameters(name, value):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        AdaptiveSearchStrategy(device="cpu", **{name: value})


@pytest.mark.parametrize("name", ["adaptation_steps", "outer_steps"])
def test_required_integer_parameters_reject_none(name):
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        AdaptiveSearchStrategy(device="cpu", **{name: None})


@pytest.mark.parametrize("value", [0, 3, -1, 1.0, "2", None, True, False])
def test_invalid_levels(value):
    with pytest.raises(ValueError, match="levels must be 1 or 2"):
        AdaptiveSearchStrategy(levels=value, device="cpu")


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize("offset", [-1000.0, 0.0, 1000.0])
def test_adapt_matches_preupdate_softmax_gradient(length, offset):
    env = RS10Env(H=4, W=5, device="cpu")
    initial = np.full((4, 5), 5, dtype=np.int32)
    env.reset(board=initial.copy())
    rects = np.ascontiguousarray(env.all_rects.numpy(), dtype=np.int32)
    lookup = np.full((env.H, env.W, env.H, env.W), -1, dtype=np.int32)
    lookup[tuple(rects.T)] = np.arange(len(rects), dtype=np.int32)
    root_mask = env.get_valid_actions_mask().numpy().copy()
    excluded = np.flatnonzero(root_mask)[-1]
    root_mask[excluded] = False
    policy = np.tile(np.linspace(-0.5, 0.5, len(rects)), (4, 1)) + offset
    original = policy.copy()
    expected_delta = np.zeros_like(policy)
    learning_rate = 0.75
    path = np.full(length + 1, -1, dtype=np.int32)
    buckets = []

    for step in range(length):
        live = int(torch.count_nonzero(env.board_2d))
        bucket = min(3, 4 * (initial.size - live) // initial.size)
        buckets.append(bucket)
        mask = env.get_valid_actions_mask().numpy().copy()
        if step == 0:
            mask &= root_mask
        actions = np.flatnonzero(mask)
        assert len(actions) > 1
        chosen = actions[0]
        path[step] = chosen
        logits = original[bucket, actions]
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        expected_delta[bucket, actions] -= learning_rate * probabilities
        expected_delta[bucket, chosen] += learning_rate
        env.step(int(chosen))

    if length > 1:
        # Repeated buckets distinguish frozen-policy updates from sequential ones.
        assert len(set(buckets)) < length
    snapshots = [array.copy() for array in (initial, rects, lookup, root_mask, path)]
    _adapt(initial, rects, lookup, 10, root_mask, policy, path, length, learning_rate)
    assert np.isfinite(policy).all()
    np.testing.assert_allclose(policy, original + expected_delta, rtol=0, atol=1e-12)
    np.testing.assert_allclose((policy - original).sum(axis=1), 0, atol=1e-10)
    for array, before in zip((initial, rects, lookup, root_mask, path), snapshots):
        np.testing.assert_array_equal(array, before)
    if length == 1:
        assert policy[0, path[0]] > original[0, path[0]]
        competitors = root_mask.copy()
        competitors[path[0]] = False
        assert np.all(policy[0, competitors] < original[0, competitors])
        np.testing.assert_array_equal(policy[0, ~root_mask], original[0, ~root_mask])
        np.testing.assert_array_equal(policy[1:], original[1:])

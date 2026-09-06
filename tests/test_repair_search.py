"""End-to-end coverage of the inherited population-search contract."""

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from rs10env import RS10Env
from rs10env.fast_search import PopulationSearchStrategy
from rs10env.repair_search import HybridRepairStrategy, RepairSearchStrategy, SpatialRepairStrategy


@pytest.fixture(scope="module", params=[RepairSearchStrategy, HybridRepairStrategy, SpatialRepairStrategy])
def strategy_class(request):
    cls = request.param
    cls.warmup()
    return cls


def _action_without_mutation(strategy, env, mask):
    board = env.board_2d.clone()
    observation = env._board_3d.clone()
    rects = env.all_rects.clone()
    original_mask = mask.clone()
    counters = (env.step_count, int(env.total_zeros), env.is_done, env.max_steps)
    action = strategy.get_action(env, mask)
    assert action.shape == torch.Size([])
    assert action.dtype == torch.int64
    assert action.device == mask.device
    assert torch.equal(env.board_2d, board)
    assert torch.equal(env._board_3d, observation)
    assert torch.equal(env.all_rects, rects)
    assert torch.equal(mask, original_mask)
    assert counters == (env.step_count, int(env.total_zeros), env.is_done, env.max_steps)
    if mask.any():
        assert mask[action].item()
    return action


@pytest.mark.parametrize("target", [8, 10])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_full_games_legal_deterministic_and_residual_scores(strategy_class, target, seed):
    games = []
    for _ in range(2):
        env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
        _, info = env.reset(seed=seed)
        strategy = strategy_class(max_rollouts=64, time_budget=60, seed=12,
                                  device="cpu")
        assert isinstance(strategy, PopulationSearchStrategy)
        assert type(strategy).get_action is PopulationSearchStrategy.get_action
        assert info["action_mask"].any()
        initial_nonzeros = int(torch.count_nonzero(env.board_2d))
        actions = []
        for step in range(env.max_steps):
            action = _action_without_mutation(strategy, env, info["action_mask"])
            if step == 0:
                scores = strategy.search_scores
                rollouts = strategy.rollouts
                assert scores[0] == initial_nonzeros
                assert 16 <= rollouts <= 64 and rollouts % 16 == 0
                assert len(scores) == rollouts // 16 + 1
                assert all(isinstance(score, int) and 0 <= score <= initial_nonzeros
                           for score in scores)
                assert all(after <= before for before, after in zip(scores, scores[1:]))
                assert rollouts == 64 or scores[-1] == 0
            else:
                assert strategy.search_scores is scores
                assert strategy.rollouts == rollouts
            actions.append(action.item())
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert env.is_done
        assert not info["action_mask"].any()
        assert strategy._plan == []
        residual = int(torch.count_nonzero(env.board_2d))
        assert scores[-1] == residual == initial_nonzeros - int(env.total_zeros)
        games.append((actions, env.board_2d.tolist(), scores, rollouts))
    assert games[0] == games[1]


@pytest.mark.parametrize("target", [8, 10])
def test_sparse_residual_counts_nonzeros_not_area(strategy_class, target):
    env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
    board = np.zeros((4, 5), dtype=np.int32)
    board[1, 1], board[2, 3] = 3, target - 3
    original = board.copy()
    _, info = env.reset(board=board)
    strategy = strategy_class(max_rollouts=64, time_budget=60, seed=12, device="cpu")
    action = _action_without_mutation(strategy, env, info["action_mask"])
    np.testing.assert_array_equal(board, original)
    assert strategy.search_scores == [2, 0]
    _, _, terminated, _, info = env.step(action)
    assert terminated
    assert int(torch.count_nonzero(env.board_2d)) == strategy.search_scores[-1]
    assert int(env.total_zeros) == 2
    assert not info["action_mask"].any()
    assert _action_without_mutation(strategy, env, info["action_mask"]).item() == 0

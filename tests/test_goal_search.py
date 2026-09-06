"""Goal-search contracts and an independent subset-feasibility oracle."""

from itertools import product

import numpy as np
import pytest
import torch

pytest.importorskip("numba")

from rs10env import RS10Env
from rs10env.fast_search import PopulationSearchStrategy
from rs10env.goal_search import GoalSearchStrategy, _subset_possible, goal_batch


@pytest.fixture(scope="module", autouse=True)
def warmup():
    GoalSearchStrategy.warmup()


@pytest.fixture
def strategy():
    return GoalSearchStrategy(max_rollouts=64, time_budget=60, seed=12,
                              device="cpu")


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


def test_warmup(strategy):
    assert GoalSearchStrategy.warmup() is None
    assert goal_batch.nopython_signatures
    assert _subset_possible.nopython_signatures
    assert strategy.name == "GoalSearch"
    assert strategy._search_batch is goal_batch
    assert strategy._plan == []


@pytest.mark.parametrize("target", [8, 10])
@pytest.mark.parametrize("seed", [0, 7, 42])
def test_full_games_legal_nonmutating_reproducible_and_final_scores(target, seed):
    games = []
    for _ in range(2):
        env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
        _, info = env.reset(seed=seed)
        strategy = GoalSearchStrategy(max_rollouts=64, time_budget=60, seed=12,
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
def test_sparse_score_and_no_move_fallback(strategy, target):
    env = RS10Env(H=4, W=5, target_sum=target, device="cpu")
    board = np.zeros((4, 5), dtype=np.int32)
    board[1, 1], board[2, 3] = 3, target - 3
    original = board.copy()
    _, info = env.reset(board=board)
    action = _action_without_mutation(strategy, env, info["action_mask"])
    np.testing.assert_array_equal(board, original)
    assert strategy.search_scores == [2, 0]
    _, _, terminated, _, info = env.step(action)
    assert terminated
    assert int(torch.count_nonzero(env.board_2d)) == strategy.search_scores[-1]
    assert int(env.total_zeros) == 2
    assert not info["action_mask"].any()
    assert _action_without_mutation(strategy, env, info["action_mask"]).item() == 0


def test_masked_root_does_not_mask_later_moves(strategy):
    env = RS10Env(H=1, W=4, device="cpu")
    _, info = env.reset(board=np.full((1, 4), 5, dtype=np.int32))
    root = env.all_rects.tolist().index([0, 0, 0, 1])
    mask = torch.zeros_like(info["action_mask"])
    mask[root] = True
    assert info["action_mask"][root]
    action = _action_without_mutation(strategy, env, mask)
    assert action.item() == root
    scores = strategy.search_scores
    assert scores[-1] == 0
    assert len(strategy._plan) == 1
    _, _, terminated, truncated, info = env.step(action)
    assert not terminated and not truncated
    action = _action_without_mutation(strategy, env, info["action_mask"])
    assert not mask[action]
    assert strategy.search_scores is scores
    _, _, terminated, _, _ = env.step(action)
    assert terminated
    assert int(torch.count_nonzero(env.board_2d)) == scores[-1]


@pytest.mark.parametrize("remaining", [1, 2])
def test_remaining_and_exhausted_horizon(strategy, remaining):
    env = RS10Env(H=4, W=5, device="cpu")
    _, info = env.reset(board=np.full((4, 5), 5, dtype=np.int32))
    env.max_steps = remaining + 1
    _, _, _, _, info = env.step(info["action_mask"].nonzero()[0, 0])
    for step in range(remaining):
        action = _action_without_mutation(strategy, env, info["action_mask"])
        assert len(strategy._plan) == remaining - step - 1
        assert strategy.rollouts == 64
        _, _, terminated, truncated, info = env.step(action)
        assert not terminated
        assert truncated == (step == remaining - 1)
    assert env.step_count == env.max_steps
    assert info["action_mask"].any()
    assert strategy.search_scores[-1] == int(torch.count_nonzero(env.board_2d))
    with pytest.raises(ValueError, match="after the episode horizon"):
        _action_without_mutation(strategy, env, info["action_mask"])


@pytest.mark.parametrize("target", [41, 45])
def test_large_target_fallback_with_legal_board(strategy, target):
    env = RS10Env(H=2, W=3, target_sum=target, device="cpu")
    board = np.array([[9, 9, 9], [9, target - 36, 0]], dtype=np.int32)
    original = board.copy()
    _, info = env.reset(board=board)
    assert info["action_mask"].any()
    assert env.max_steps - env.step_count >= 2
    action = _action_without_mutation(strategy, env, info["action_mask"])
    np.testing.assert_array_equal(board, original)
    assert strategy.search_scores == [5, 0]
    _, _, terminated, _, info = env.step(action)
    assert terminated
    assert not info["action_mask"].any()
    assert int(torch.count_nonzero(env.board_2d)) == strategy.search_scores[-1]


def _brute_subset_possible(board, rect, excess):
    r1, c1, r2, c2 = rect
    cells = [(r, c) for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)
             if board[r, c] != 0]
    for removed in product([False, True], repeat=len(cells)):
        if sum(int(board[cell]) for cell, take in zip(cells, removed) if take) != excess:
            continue
        live = {cell for cell, take in zip(cells, removed) if not take}
        if (((r1, c1) in live and (r2, c2) in live)
                or ((r1, c2) in live and (r2, c1) in live)):
            return True
    return False


@pytest.mark.parametrize("shape", [(1, 3), (3, 1), (2, 2)])
def test_subset_possible_matches_exhaustive_small_subsets(shape):
    rect = np.array([0, 0, shape[0] - 1, shape[1] - 1], dtype=np.int32)
    for values in product(range(3), repeat=shape[0] * shape[1]):
        board = np.array(values, dtype=np.int32).reshape(shape)
        original = board.copy()
        original_rect = rect.copy()
        for excess in range(sum(values) + 2):
            expected = _brute_subset_possible(board, rect, excess)
            assert _subset_possible(board, rect, excess) == expected, (values, excess)
        np.testing.assert_array_equal(board, original)
        np.testing.assert_array_equal(rect, original_rect)


@pytest.mark.parametrize("values,excess,expected", [
    ([[1, 2, 0], [0, 2, 1]], 3, False),  # Even interior cannot remove odd excess.
    ([[1, 2, 0], [0, 2, 1]], 4, True),
    ([[1, 2, 0], [0, 2, 1]], 0, True),
    ([[1, 2, 0], [0, 2, 1]], 5, False),  # Cannot spend protected corners.
    ([[0, 2, 1], [1, 2, 0]], 4, True),  # Only the other diagonal is live.
    ([[1, 2, 1], [0, 2, 0]], 0, False),
    ([[2, 5, 5], [5, 3, 0]], 5, True),  # May delete an unprotected corner.
])
def test_subset_possible_corners_parity_and_rectangle_scope(values, excess, expected):
    board = np.full((4, 5), 1, dtype=np.int32)
    board[1:3, 1:4] = values
    rect = np.array([1, 1, 2, 3], dtype=np.int32)
    original = board.copy()
    assert _brute_subset_possible(board, rect, excess) == expected
    assert _subset_possible(board, rect, excess) == expected
    np.testing.assert_array_equal(board, original)

"""Check deletion certificates against every reachable small-board state."""

from functools import lru_cache

import numpy as np
import pytest

from rs10env import RS10Env
from rs10env.score_bound import cleared_cells_upper_bound


@pytest.fixture(params=[False, True], ids=["python", "numba"])
def use_numba(request):
    if request.param:
        pytest.importorskip("numba")
    return request.param


def _assert_bound_against_exact_search(board, target, use_numba):
    env = RS10Env(H=board.shape[0], W=board.shape[1], target_sum=target,
                  device="cpu")
    reachable = {}

    @lru_cache(None)
    def visit(values):
        state = np.array(values, dtype=np.int32).reshape(board.shape)
        # Use the direct environment mask, never a search kernel or bound oracle.
        env.reset(board=state.copy())
        actions = env.get_valid_actions_mask().nonzero(as_tuple=True)[0].tolist()
        optimum = 0
        ever_removed = np.zeros(board.shape, dtype=bool)
        for action in actions:
            env.reset(board=state.copy())
            assert env.get_valid_actions_mask()[action].item()
            env.step(action)
            child = env.board_2d.numpy().copy()
            removed = (state != 0) & (child == 0)
            assert removed.any()
            future_optimum, future_removed = visit(tuple(child.ravel()))
            optimum = max(optimum, int(removed.sum()) + future_optimum)
            ever_removed |= removed | future_removed
        original = state.copy()
        upper, forced = cleared_cells_upper_bound(state, target, use_numba=use_numba)
        np.testing.assert_array_equal(state, original)
        assert isinstance(upper, int)
        assert forced.shape == state.shape and forced.dtype == np.bool_
        assert optimum <= upper <= np.count_nonzero(state)
        assert not forced[state == 0].any()
        assert upper == np.count_nonzero(state) - np.count_nonzero(forced)
        # The union includes all descendants, not just an optimal trajectory.
        assert not (forced & ever_removed).any()
        reachable[values] = (upper, forced.copy())
        return optimum, ever_removed

    root = tuple(board.ravel())
    optimum, _ = visit(root)
    upper, forced = reachable[root]
    for values in reachable:
        state = np.array(values).reshape(board.shape)
        np.testing.assert_array_equal(state[forced], board[forced])
    return optimum, upper, forced


@pytest.mark.parametrize("shape", [(2, 2), (2, 3)])
@pytest.mark.parametrize("target", [8, 10])
@pytest.mark.parametrize("seed", [0, 7, 42, 123])
def test_seeded_boards_bound_exact_optimum_and_forced_cells(shape, target, seed,
                                                         use_numba):
    rng = np.random.default_rng(seed)
    board = rng.integers(0, 10, size=shape, dtype=np.int32)
    # Guarantee a legal branch while retaining seeded optional cells.
    board[0, :2] = [3, target - 3]
    _assert_bound_against_exact_search(board, target, use_numba)


@pytest.mark.parametrize("transpose", [False, True], ids=["row", "column"])
@pytest.mark.parametrize("values,target,expected", [
    ([4, 6], 10, 2),
    ([3, 0, 5], 8, 2),
    ([0, 4, 0, 6, 0], 10, 2),
    ([2, 3, 5], 10, 3),
    ([4, 4, 9], 8, 2),
    ([0, 8, 0], 8, 0),
    ([0, 0, 0], 10, 0),
    ([1, 0, 1], 10, 0),
])
def test_sparse_degenerate_rectangles_count_mandatory_corners_once(
        values, target, expected, transpose, use_numba):
    board = np.array([values], dtype=np.int32)
    if transpose:
        board = board.T.copy()
    optimum, upper, forced = _assert_bound_against_exact_search(board, target, use_numba)
    assert optimum == upper == expected
    assert np.count_nonzero(forced) == np.count_nonzero(board) - expected


@pytest.mark.parametrize("board,target,expected", [
    ([[4, 0], [0, 6]], 10, 2),
    ([[0, 3], [5, 0]], 8, 2),
    ([[8, 0], [0, 0]], 8, 0),
    ([[4, 4, 9], [0, 0, 9]], 8, 2),
])
def test_sparse_diagonals_and_permanently_alive_cells(board, target, expected,
                                                    use_numba):
    board = np.array(board, dtype=np.int32)
    optimum, upper, _ = _assert_bound_against_exact_search(board, target, use_numba)
    assert optimum == upper == expected

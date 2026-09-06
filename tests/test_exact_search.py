"""Independent tiny-board oracle and budget checks; no strategy integration."""

from itertools import product

import pytest

from rs10env.exact_search import solve_endgame


def rectangles(height, width):
    # RS10Env: upper-triangular row pairs, then column pairs, excluding area 1.
    return [(top, left, bottom, right)
            for top in range(height) for bottom in range(top, height)
            for left in range(width) for right in range(left, width)
            if top != bottom or left != right]


def moves(board, rects, target):
    for action, (top, left, bottom, right) in enumerate(rects):
        if not ((board[top][left] and board[bottom][right])
                or (board[top][right] and board[bottom][left])):
            continue
        cells = [(r, c) for r in range(top, bottom + 1)
                 for c in range(left, right + 1)]
        if sum(board[r][c] for r, c in cells) != target:
            continue
        child = [list(row) for row in board]
        gain = sum(board[r][c] != 0 for r, c in cells)
        for r, c in cells:
            child[r][c] = 0
        yield action, child, gain


def brute(board, rects, target, steps):
    if steps == 0:
        return 0
    return max((gain + brute(child, rects, target, steps - 1)
                for _, child, gain in moves(board, rects, target)), default=0)


def replay(board, rects, target, result):
    cleared = 0
    for action in result.actions:
        legal = {a: (child, gain) for a, child, gain in moves(board, rects, target)}
        assert action in legal
        board, gain = legal[action]
        cleared += gain
    assert cleared == result.cleared


@pytest.mark.parametrize("steps", [None, 0, 1, 2])
def test_exhaustive_tiny_boards(steps):
    rects = rectangles(2, 2)
    for values in product(range(3), repeat=4):
        board = [list(values[:2]), list(values[2:])]
        result = solve_endgame(board, rects, target_sum=2, max_steps=steps)
        assert result.proven_optimal
        assert result.cleared == brute(board, rects, 2, 4 if steps is None else steps)
        assert steps is None or len(result.actions) <= steps
        replay(board, rects, 2, result)
        assert board == [list(values[:2]), list(values[2:])]


@pytest.mark.parametrize("budget", [1, 2, 3, 100])
def test_budget_returns_legal_continuation(budget):
    board = [[5, 5, 5, 5, 1]]
    rects = rectangles(1, 5)
    result = solve_endgame(board, rects, max_nodes=budget)
    assert 1 <= result.nodes <= budget
    assert result.cleared <= 4
    replay(board, rects, 10, result)
    if budget == 1:
        assert result.cleared == 2
        assert not result.proven_optimal
    if budget == 100:
        assert result.cleared == 4
        assert result.proven_optimal


def test_successor_deduplication_and_action_order():
    rects = [(0, 1, 0, 2), (0, 0, 0, 1), (0, 0, 0, 1)]
    result = solve_endgame([[5, 5, 1]], rects, max_nodes=2)
    assert result.actions == (1,)
    assert result.cleared == 2
    assert result.nodes == 2
    assert result.proven_optimal


def test_solved_occupancy_reused_across_move_orders():
    board = [[5, 5, 1, 5, 5]]
    rects = [(0, 0, 0, 1), (0, 3, 0, 4)]
    result = solve_endgame(board, rects, max_nodes=4)
    assert result.cleared == 4
    assert result.nodes == 4
    assert result.proven_optimal
    replay(board, rects, 10, result)


@pytest.mark.parametrize("board,expected", [
    ([[5, 0], [0, 5]], 2),
    ([[0, 5], [5, 0]], 2),
    ([[5, 5], [0, 0]], 0),
    ([[0, 0], [0, 0]], 0),
])
def test_live_diagonals(board, expected):
    result = solve_endgame(board, [(0, 0, 1, 1)], max_nodes=1)
    assert result.cleared == expected
    assert result.proven_optimal


def test_arbitrary_width_occupancy():
    board = [[0] * 70 + [5, 5]]
    result = solve_endgame(board, [(0, 70, 0, 71)], max_nodes=1)
    assert result.cleared == 2
    assert result.proven_optimal


def test_supplied_indices_match_environment():
    from rs10env.env import RS10Env

    env = RS10Env(H=2, W=3, device="cpu")
    assert env.all_rects.tolist() == [list(rect) for rect in rectangles(2, 3)]
    result = solve_endgame([[5, 5, 0], [0, 0, 0]], env.all_rects)
    assert tuple(env.all_rects[result.actions[0]].tolist()) == (0, 0, 0, 1)
    assert result.cleared == 2


def test_empty_board():
    result = solve_endgame([], [])
    assert result.actions == ()
    assert result.cleared == 0
    assert result.proven_optimal


@pytest.mark.parametrize("kwargs", [
    {"max_nodes": 0}, {"max_nodes": -1}, {"target_sum": 0},
    {"max_steps": -1},
])
def test_invalid_limits(kwargs):
    with pytest.raises(ValueError):
        solve_endgame([[5, 5]], [(0, 0, 0, 1)], **kwargs)


@pytest.mark.parametrize("board,rects", [
    ([[5], [5, 0]], []),
    ([[-1, 11]], []),
    ([[5, 5]], [(0, 0, 1, 1)]),
    ([[5, 5]], [(0, 0, 0, 0)]),
])
def test_invalid_geometry_or_values(board, rects):
    with pytest.raises(ValueError):
        solve_endgame(board, rects)


def test_fractional_values_rejected():
    with pytest.raises(TypeError):
        solve_endgame([[5.5, 4.5]], [(0, 0, 0, 1)])

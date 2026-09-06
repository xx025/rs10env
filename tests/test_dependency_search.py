"""Independent unpruned reachability oracle, witnesses, and proof scope."""

from itertools import product

import pytest

from rs10env.dependency_search import activate_rectangle


def rectangles(height, width):
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
        for r, c in cells:
            child[r][c] = 0
        yield action, child


def brute(board, rects, goal, target, steps):
    # No goal-corner or excess pruning: independently checks their soundness.
    if steps == 0:
        return False
    return any(action == goal or brute(child, rects, goal, target, steps - 1)
               for action, child in moves(board, rects, target))


def replay(board, rects, goal, target, result):
    if not result.success:
        assert result.actions == ()
        return
    assert not result.proven_infeasible
    assert result.actions[-1] == goal
    top, left, bottom, right = rects[goal]
    for step, action in enumerate(result.actions):
        old_total = sum(map(sum, board))
        old_goal = sum(board[r][c] for r in range(top, bottom + 1)
                       for c in range(left, right + 1))
        legal = dict(moves(board, rects, target))
        assert action in legal
        child = legal[action]
        assert old_total - sum(map(sum, child)) == target
        new_goal = sum(child[r][c] for r in range(top, bottom + 1)
                       for c in range(left, right + 1))
        a, b, c, d = rects[action]
        intersection = sum(board[r][col]
                           for r in range(max(top, a), min(bottom, c) + 1)
                           for col in range(max(left, b), min(right, d) + 1))
        assert old_goal - new_goal == intersection
        if step < len(result.actions) - 1:
            assert new_goal >= target
            assert ((child[top][left] and child[bottom][right])
                    or (child[top][right] and child[bottom][left]))
        board = child
    assert new_goal == 0


@pytest.mark.parametrize("steps", [None, 0, 1, 2, 3])
def test_tiny_unpruned_reachability(steps):
    rects = rectangles(2, 2)
    for values in product(range(3), repeat=4):
        board = [list(values[:2]), list(values[2:])]
        for goal in range(len(rects)):
            result = activate_rectangle(board, rects, goal, target_sum=2,
                                        max_steps=steps)
            expected = brute(board, rects, goal, 2, 4 if steps is None else steps)
            assert result.success == expected
            assert result.proven_infeasible == (not expected)
            assert steps is None or len(result.actions) <= steps
            replay(board, rects, goal, 2, result)
        assert board == [list(values[:2]), list(values[2:])]


@pytest.mark.parametrize("budget", [1, 2, 3, 100])
def test_outside_prerequisite_and_budget(budget):
    board = [[5, 2, 5], [5, 5, 0], [0, 8, 0]]
    rects = [(0, 0, 0, 2), (1, 0, 1, 1), (0, 1, 2, 1)]
    result = activate_rectangle(board, rects, 0, max_nodes=budget)
    assert result.nodes == min(budget, 3)
    assert not result.proven_infeasible
    assert result.success == (budget >= 3)
    if result.success:
        assert result.actions == (1, 2, 0)
    replay(board, rects, 0, 10, result)


def test_horizon_and_action_set_proof_scope():
    board = [[5, 2, 5], [5, 5, 0], [0, 8, 0]]
    rects = [(0, 0, 0, 2), (1, 0, 1, 1), (0, 1, 2, 1)]
    assert activate_rectangle(board, rects, 0, max_steps=2).proven_infeasible
    assert activate_rectangle(board, rects, 0, max_steps=3).success
    assert activate_rectangle(board, [rects[0], rects[2]], 0).proven_infeasible
    assert activate_rectangle(board, rects, 0).success


@pytest.mark.parametrize("board,rects", [
    ([[2, 5, 5], [5, 3, 0]], [(0, 0, 1, 1), (0, 1, 0, 2)]),
    ([[5, 5, 2], [0, 3, 5]], [(0, 1, 1, 2), (0, 0, 0, 1)]),
])
def test_either_diagonal_can_survive_corner_deletion(board, rects):
    result = activate_rectangle(board, rects, 0)
    assert result.actions == (1, 0)
    replay(board, rects, 0, 10, result)


@pytest.mark.parametrize("board,rects", [
    # Legal interior removal overshoots the goal's excess of one.
    ([[4, 2, 5], [0, 8, 0]], [(0, 0, 0, 2), (0, 1, 1, 1)]),
    # Exact excess removal still destroys the last live goal diagonal.
    ([[5, 5, 5, 5]], [(0, 0, 0, 2), (0, 2, 0, 3)]),
    # Already below target, or no live diagonal despite sufficient sum.
    ([[2, 2]], [(0, 0, 0, 1)]),
    ([[5, 5], [0, 0]], [(0, 0, 1, 1)]),
])
def test_irreversible_failures(board, rects):
    result = activate_rectangle(board, rects, 0)
    assert not result.success
    assert result.proven_infeasible


def test_complementary_masks_and_transpositions():
    # Two disjoint outside removals commute. Same goal occupancy is NOT
    # the same state; their union reached in either order IS the same state.
    board = [[5, 1, 5, 0, 5, 5, 0, 5, 5]]
    rects = [(0, 0, 0, 2), (0, 4, 0, 5), (0, 7, 0, 8), (0, 4, 0, 5)]
    for budget in (1, 2, 3, 4):
        result = activate_rectangle(board, rects, 0, max_nodes=budget)
        assert result.nodes == budget
        assert not result.success
        assert result.proven_infeasible == (budget == 4)


def test_bound_includes_goal_action():
    board = [[5, 5, 5, 5, 5, 5]]
    rects = [(0, 0, 0, 5), (0, 1, 0, 2), (0, 3, 0, 4)]
    result = activate_rectangle(board, rects, 0, max_steps=2, max_nodes=1)
    assert result.proven_infeasible
    assert result.nodes == 1
    result = activate_rectangle(board, rects, 0, max_steps=3)
    assert result.success
    replay(board, rects, 0, 10, result)


def test_general_bitset_and_immediate_goal():
    board = [[0] * 70 + [5, 5]]
    rects = [(0, 70, 0, 71), (0, 70, 0, 71)]
    result = activate_rectangle(board, rects, 1, max_nodes=1, max_steps=1)
    assert result.actions == (1,)
    assert result.nodes == 1
    replay(board, rects, 1, 10, result)
    assert activate_rectangle(board, rects, 1, max_steps=0).proven_infeasible


@pytest.mark.parametrize("kwargs", [
    {"max_nodes": 0}, {"max_nodes": -1}, {"target_sum": 0},
    {"target_sum": -1}, {"max_steps": -1}, {"goal_action": -1},
    {"goal_action": 1},
])
def test_invalid_limits_or_goal(kwargs):
    args = {"goal_action": 0, **kwargs}
    with pytest.raises(ValueError):
        activate_rectangle([[5, 5]], [(0, 0, 0, 1)], **args)


@pytest.mark.parametrize("board,rects", [
    ([[5], [5, 0]], [(0, 0, 1, 0)]),
    ([[-1, 11]], [(0, 0, 0, 1)]),
    ([[5, 5]], [(0, 0, 1, 1)]),
    ([[5, 5]], [(0, 0, 0, 0)]),
    ([], []),
])
def test_invalid_board_or_geometry(board, rects):
    with pytest.raises(ValueError):
        activate_rectangle(board, rects, 0)


def test_fractional_values_rejected():
    with pytest.raises(TypeError):
        activate_rectangle([[5.5, 4.5]], [(0, 0, 0, 1)], 0)

"""Offline upper bound on future legally cleared nonzero cells, without refill.

This is an optimistic geometry/subset-sum diagnostic, not a move strategy.
Only NumPy is required; Numba, when installed, accelerates targets <= 62.
Larger targets use Python's arbitrary-width integer bitsets and may be slow.
No environment is created, no seeds are generated, and no files are written.

Proof of safety:
Start with an empty set F of certified permanently alive cells. For any future
legal move containing a candidate x, its surviving cells form a subset of the
input rectangle, with sum target_sum. That subset includes x, a live diagonal,
and every member of F in the rectangle. Therefore it is one of the subsets
tested here. If none exists, x cannot be cleared. Adding all such x to F is
sound by induction. Updates are simultaneous, so there is no circular proof.
At the fixed point, every cell ever cleared belongs to the complement of F
among input nonzeros; its size bounds the number cleared by any legal sequence.

Passing the subset test does NOT establish reachability: arbitrary optional
cells are allowed to disappear independently, even when no legal sequence can
do that. Thus the bound can be loose. Forced cells remain present, not zeroed.
The certificate assumes nonnegative integer values, legal diagonal-alive,
sum-target rectangles of area >= 2, and deletion only (no refill or movement).
In particular, unchecked invalid actions accepted by env.step are not covered.
"""

import operator

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


def _bound(board, target_sum):
    height, width = board.shape
    forced = np.zeros((height, width), dtype=np.bool_)
    while True:
        removable = np.zeros((height, width), dtype=np.bool_)
        for top in range(height):
            for bottom in range(top, height):
                for left in range(width):
                    for right in range(left, width):
                        if top == bottom and left == right:
                            continue
                        for diagonal in range(2):
                            first_col = left if diagonal == 0 else right
                            last_col = right if diagonal == 0 else left
                            if (board[top, first_col] == 0
                                    or board[bottom, last_col] == 0):
                                continue
                            # Union by position: forced corners count once,
                            # including in one-row/one-column rectangles.
                            mandatory_sum = 0
                            feasible = True
                            for row in range(top, bottom + 1):
                                for col in range(left, right + 1):
                                    mandatory = (
                                        forced[row, col]
                                        or (row == top and col == first_col)
                                        or (row == bottom and col == last_col)
                                    )
                                    if mandatory:
                                        value = int(board[row, col])
                                        if value > target_sum - mandatory_sum:
                                            feasible = False
                                            break
                                        mandatory_sum += value
                                if not feasible:
                                    break
                            if not feasible:
                                continue
                            for candidate_row in range(top, bottom + 1):
                                for candidate_col in range(left, right + 1):
                                    value = int(board[candidate_row, candidate_col])
                                    if (value == 0
                                            or forced[candidate_row, candidate_col]
                                            or removable[candidate_row, candidate_col]):
                                        continue
                                    candidate_mandatory = (
                                        (candidate_row == top and candidate_col == first_col)
                                        or (candidate_row == bottom and candidate_col == last_col)
                                    )
                                    remaining = target_sum - mandatory_sum
                                    if not candidate_mandatory:
                                        if value > remaining:
                                            continue
                                        remaining -= value
                                    bits = 1
                                    goal = 1 << remaining
                                    for row in range(top, bottom + 1):
                                        if bits & goal:
                                            break
                                        for col in range(left, right + 1):
                                            if (forced[row, col]
                                                    or (row == top and col == first_col)
                                                    or (row == bottom and col == last_col)
                                                    or (row == candidate_row and col == candidate_col)):
                                                continue
                                            value = int(board[row, col])
                                            if 0 < value <= remaining:
                                                # Truncate BEFORE shifting to
                                                # avoid signed overflow in JIT.
                                                mask = (1 << (remaining - value + 1)) - 1
                                                bits |= (bits & mask) << value
                                                if bits & goal:
                                                    break
                                    if bits & goal:
                                        removable[candidate_row, candidate_col] = True
        changed = False
        upper_bound = 0
        for row in range(height):
            for col in range(width):
                if board[row, col] != 0 and not forced[row, col]:
                    if removable[row, col]:
                        upper_bound += 1
                    else:
                        forced[row, col] = True
                        changed = True
        if not changed:
            return upper_bound, forced


_compiled_bound = njit(cache=False)(_bound) if njit is not None else None


def cleared_cells_upper_bound(board, target_sum=10, *, use_numba=True):
    """Return ``(upper_bound, forced_mask)`` without modifying ``board``.

    Args:
        board: Any H-by-W nonnegative integer NumPy-compatible board, with
            values representable as int64. Zero denotes an already empty cell.
        target_sum: Positive integer rectangle sum, representable as int64.
        use_numba: Use optional Numba for target_sum <= 62. False selects the
            same algorithm with Python integer bitsets. JIT caching is disabled.

    Returns:
        An integer upper bound on ADDITIONAL nonzero cells legally cleared,
        and an H-by-W boolean mask of certified permanently unremovable input
        nonzeros. Initial zeros are never counted or marked forced. For a bound
        on final empty cells, add the input zero count. For cumulative score
        from a partially played board, add the actual already-cleared count.

    The implementation scans all rectangles, both diagonals, and candidate
    cells, with a subset DP over optional cells. At most count_nonzero(board)
    productive forcing passes occur, followed by a fixed-point pass. This is
    intended for offline diagnostics, not an inner search loop. An upper bound
    below 140 certifies that 140 additional clears are impossible; a bound at
    least 140 says nothing about whether 140 is attainable.
    """
    if isinstance(target_sum, (bool, np.bool_)):
        raise TypeError("target_sum must be a positive integer, not bool")
    target_sum = operator.index(target_sum)
    if not 1 <= target_sum <= np.iinfo(np.int64).max:
        raise ValueError("target_sum must be a positive int64 integer")
    values = np.asarray(board)
    if values.ndim != 2:
        raise ValueError("board must be two-dimensional")
    if values.dtype.kind not in "iu":
        raise TypeError("board must contain integers")
    if np.any(values < 0) or np.any(values > np.iinfo(np.int64).max):
        raise ValueError("board values must be nonnegative int64 integers")
    values = np.ascontiguousarray(values, dtype=np.int64)
    implementation = (
        _compiled_bound
        if use_numba and _compiled_bound is not None and target_sum <= 62
        else _bound
    )
    upper_bound, forced = implementation(values, target_sum)
    return int(upper_bound), forced

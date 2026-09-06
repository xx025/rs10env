"""Pure, bounded goal-rectangle reachability for deletion-only RS10 boards.

Let O be the live-cell set, v its fixed nonnegative weights, G the goal,
and T > 0 the legal action sum. A legal action A deletes D = O & A with
v(D) = T. Thus v(O') = v(O) - T and the goal excess E = v(O & G) - T
changes by exactly p = v(D & G), where 0 <= p <= T. A successful prefix
cannot overshoot E or destroy both goal diagonals: neither weights nor
corners ever return. Conversely these necessary constraints discard no
prefix that eventually executes G. Keeping the disjunction of diagonals
searches both orientations without prematurely protecting a fixed pair.

At least ceil(E / T) preparatory moves and one goal move remain. This is
only a lower bound, not a feasibility test: legal removals may be unavailable
or need zero-progress outside prerequisites. Full occupancy, not excess or
goal occupancy alone, is the transposition key. Conservation makes depth
unique: depth = (v(O_initial) - v(O)) / T, even across different move orders.
The finite deletion DAG can therefore be exhausted without losing paths,
including under a step limit. Node-budget interruption proves nothing.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
from operator import index


@dataclass(frozen=True)
class ActivationResult:
    actions: tuple[int, ...]
    success: bool
    proven_infeasible: bool
    nodes: int


def activate_rectangle(board, rects, goal_action, target_sum=10,
                       max_nodes=1000, max_steps=None):
    """Find a legal sequence ending with the supplied goal action index.

    Inputs follow ``solve_endgame``: nonnegative integer board rows and
    inclusive (top, left, bottom, right) rectangles of area >= 2. Action
    indices retain supplied order; use ``env.all_rects.tolist()``. Python
    integer bitsets have no board-size limit. Inputs are not modified; no
    environment access, logging, random state, or file I/O is used.

    ``max_steps`` limits the entire returned sequence, INCLUDING the goal.
    ``proven_infeasible`` means no such sequence exists using these rects
    within that horizon, not that the board cannot otherwise be cleared.
    Without a horizon it certifies unrestricted goal unreachability in the
    supplied action set. Sound pruning counts as exhaustive resolution.

    ``max_nodes`` is a positive cap on popped unique occupancy states,
    including the root and states resolved by bounds. A nonempty frontier
    when the budget expires returns unknown (both flags false). Failure or
    unknown returns no actions, rather than an uncertified partial plan.
    Best-first order favors smaller excess, then depth plus the admissible
    remaining-move bound. Success is a witness, NOT a shortest-path claim.
    The budget limits expansions, not wall time or frontier memory.
    """
    target_sum = index(target_sum)
    max_nodes = index(max_nodes)
    goal_action = index(goal_action)
    if target_sum <= 0 or max_nodes <= 0:
        raise ValueError("target_sum and max_nodes must be positive integers")
    if max_steps is not None:
        max_steps = index(max_steps)
        if max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
    rows = [tuple(index(value) for value in row) for row in board]
    height = len(rows)
    width = len(rows[0]) if rows else 0
    if any(len(row) != width for row in rows):
        raise ValueError("board must be rectangular")
    values = tuple(value for row in rows for value in row)
    if any(value < 0 for value in values):
        raise ValueError("board values must be nonnegative")
    initial = sum(1 << cell for cell, value in enumerate(values) if value)
    rectangles = []
    for rect in rects:
        top, left, bottom, right = map(index, rect)
        if not (0 <= top <= bottom < height and 0 <= left <= right < width):
            raise ValueError("rectangle coordinates are outside the board")
        if top == bottom and left == right:
            raise ValueError("rectangles must have area at least two")
        mask = sum(1 << (row * width + col)
                   for row in range(top, bottom + 1)
                   for col in range(left, right + 1))
        diag1 = (1 << (top * width + left)) | (1 << (bottom * width + right))
        diag2 = (1 << (top * width + right)) | (1 << (bottom * width + left))
        rectangles.append((mask & initial, diag1, diag2))
    if not 0 <= goal_action < len(rectangles):
        raise ValueError("goal_action must index a supplied rectangle")

    def weight(mask):
        total = 0
        while mask:
            bit = mask & -mask
            total += values[bit.bit_length() - 1]
            mask ^= bit
        return total

    def subset_possible(mask, total):
        if total < 0:
            return False
        reachable = 1
        cap = (1 << (total + 1)) - 1
        while mask:
            bit = mask & -mask
            reachable |= reachable << values[bit.bit_length() - 1]
            reachable &= cap
            mask ^= bit
        return bool(reachable & (1 << total))

    goal, corner1, corner2 = rectangles[goal_action]
    excess = weight(initial & goal) - target_sum
    bound = (excess + target_sum - 1) // target_sum + 1
    # A serial tie-breaker preserves deterministic supplied-action order.
    frontier = [(excess, bound, 0, initial, 0)]
    parents = {initial: None}
    serial = nodes = 0
    while frontier and nodes < max_nodes:
        excess, _, _, occupied, depth = heappop(frontier)
        nodes += 1
        if excess < 0 or (occupied & corner1 != corner1
                          and occupied & corner2 != corner2):
            continue
        # All removed goal cells must avoid at least one surviving diagonal.
        # This relaxation ignores rectangle reachability, so only failure prunes.
        if not any(occupied & corners == corners
                   and subset_possible((occupied & goal) & ~corners, excess)
                   for corners in (corner1, corner2)):
            continue
        remaining = (excess + target_sum - 1) // target_sum + 1
        if max_steps is not None and depth + remaining > max_steps:
            continue
        if excess == 0:
            actions = [goal_action]
            state = occupied
            while parents[state] is not None:
                state, action = parents[state]
                actions.append(action)
            actions.reverse()
            return ActivationResult(tuple(actions), True, False, nodes)

        for action, (mask, diag1, diag2) in enumerate(rectangles):
            if occupied & diag1 != diag1 and occupied & diag2 != diag2:
                continue
            removed = occupied & mask
            if weight(removed) != target_sum:
                continue
            successor = occupied ^ removed
            if successor in parents:
                continue
            if successor & corner1 != corner1 and successor & corner2 != corner2:
                continue
            next_excess = excess - weight(removed & goal)
            if next_excess < 0:
                continue
            estimate = depth + 1 + (next_excess + target_sum - 1) // target_sum + 1
            if max_steps is not None and estimate > max_steps:
                continue
            parents[successor] = (occupied, action)
            serial += 1
            heappush(frontier, (next_excess, estimate, serial, successor, depth + 1))

    return ActivationResult((), False, not frontier, nodes)

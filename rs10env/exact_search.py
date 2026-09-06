"""Bounded, dependency-free exact search for deletion-only small endgames."""

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class ExactSearchResult:
    actions: tuple[int, ...]
    cleared: int
    proven_optimal: bool
    nodes: int


def solve_endgame(board, rects, target_sum=10, max_nodes=10000, max_steps=None):
    """Maximize additional nonzero cells cleared by legal rectangle actions.

    ``board`` is a rectangular iterable of nonnegative integer rows. ``rects``
    contains inclusive (top, left, bottom, right) coordinates, in action order;
    pass RS10Env.all_rects (or its tolist()) to preserve environment indices.
    Optimality is relative to these supplied rectangles and ``max_steps``, an
    optional nonnegative limit on continuation actions, not elapsed game steps.

    ``max_nodes`` is a positive limit on expanded, uncached occupancy states,
    including the root. Budget exhaustion returns a legal best-found prefix
    and never claims optimality. This is a node budget, not a wall-time limit.
    Inputs are not modified; no environment, logging, or file I/O is used.
    """
    target_sum = index(target_sum)
    max_nodes = index(max_nodes)
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

    cache = {}
    nodes = 0
    exhausted = False

    def search(occupied, depth):
        nonlocal nodes, exhausted
        # Fixed positive target means occupancy determines removed sum and
        # hence depth, so even with a step limit the mask alone is a valid key.
        if occupied in cache:
            cleared, actions = cache[occupied]
            return cleared, actions, True
        if nodes >= max_nodes:
            exhausted = True
            return 0, (), False
        nodes += 1
        if not occupied or depth == max_steps:
            cache[occupied] = (0, ())
            return 0, (), True

        successors = {}
        for action, (mask, diag1, diag2) in enumerate(rectangles):
            if occupied & diag1 != diag1 and occupied & diag2 != diag2:
                continue
            removed = occupied & mask
            total = 0
            remaining = removed
            while remaining and total <= target_sum:
                bit = remaining & -remaining
                total += values[bit.bit_length() - 1]
                remaining ^= bit
            if total == target_sum:
                successor = occupied ^ removed
                successors.setdefault(successor, (action, removed.bit_count()))

        best, actions, complete = 0, (), True
        live_bound = occupied.bit_count()
        # Clearing more cells first gives useful incumbents under tight budgets.
        for successor, (action, gain) in sorted(
                successors.items(), key=lambda item: -item[1][1]):
            if gain > best:
                best, actions = gain, (action,)
            # No continuation can clear more than all currently live cells.
            if best == live_bound:
                break
            if depth + 1 == max_steps or not successor:
                extra, suffix, solved = 0, (), True
            else:
                extra, suffix, solved = search(successor, depth + 1)
            if gain + extra > best:
                best, actions = gain + extra, (action,) + suffix
            complete = complete and solved
            if exhausted:
                return best, actions, False
            if best == live_bound:
                break
        if complete:
            cache[occupied] = (best, actions)
        return best, actions, complete

    cleared, actions, complete = search(initial, 0)
    return ExactSearchResult(actions, cleared, complete and not exhausted, nodes)

"""Experimental full-game search biased toward stranded-cell rectangle goals.

Subset feasibility is only a necessary condition for activation. Goals guide
rollouts, but candidates are ranked by the full board's residual cell count.
"""

import numpy as np
from numba import njit

from rs10env.fast_search import PopulationSearchStrategy, legal_actions, search_batch
from rs10env.repair_search import hybrid_batch


@njit(cache=True)
def _subset_possible(board, rect, excess):
    """Can excess be removed while retaining either live diagonal?"""
    r1, c1, r2, c2 = rect
    for diagonal in range(2):
        ca, cb = c1, c2
        if diagonal:
            ca, cb = c2, c1
        if board[r1, ca] == 0 or board[r2, cb] == 0:
            continue
        reachable = np.int64(1)
        cap = (np.int64(1) << (excess + 1)) - np.int64(1)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if (r == r1 and c == ca) or (r == r2 and c == cb):
                    continue
                value = board[r, c]
                if 0 < value <= excess:
                    reachable = (reachable | (reachable << value)) & cap
        if reachable & (np.int64(1) << excess):
            return True
    return False


@njit(cache=True)
def goal_batch(initial, rects, lookup, target, horizon, root_mask,
               population, lengths, losses, best_path, best_length,
               best_loss, rng, count):
    first = count - max(1, count // 8)
    best_length, best_loss = hybrid_batch(
        initial, rects, lookup, target, horizon, root_mask,
        population, lengths, losses, best_path, best_length, best_loss, rng, first)
    h, w = initial.shape
    for trial in range(count - first):
        # Keep the signed-int64 subset arithmetic deliberately bounded.
        if target <= 0 or target + 20 > 60 or horizon < 2:
            best_length, best_loss = search_batch(
                initial, rects, lookup, target, horizon, root_mask,
                population, lengths, losses, best_path, best_length, best_loss, rng, 1)
            continue
        parent = rng.integers(0, len(lengths))
        parent_length = min(lengths[parent], horizon)
        cut = rng.integers(0, min(parent_length, horizon - 1) + 1)
        board = initial.copy()
        terminal = initial.copy()
        path = np.full(horizon, -1, dtype=np.int32)
        for step in range(parent_length):
            a = population[parent, step]
            r1, c1, r2, c2 = rects[a]
            terminal[r1:r2 + 1, c1:c2 + 1] = 0
            if step < cut:
                path[step] = a
                board[r1:r2 + 1, c1:c2 + 1] = 0

        sums = np.zeros((h + 1, w + 1), dtype=np.int64)
        residuals = np.zeros((h + 1, w + 1), dtype=np.int64)
        for r in range(h):
            for c in range(w):
                sums[r + 1, c + 1] = (board[r, c] + sums[r, c + 1]
                                       + sums[r + 1, c] - sums[r, c])
                residuals[r + 1, c + 1] = (
                    int(terminal[r, c] != 0) + residuals[r, c + 1]
                    + residuals[r + 1, c] - residuals[r, c])
        goal = -1
        excess = 0
        eligible = 0
        for a in range(len(rects)):
            r1, c1, r2, c2 = rects[a]
            total = (sums[r2 + 1, c2 + 1] - sums[r1, c2 + 1]
                     - sums[r2 + 1, c1] + sums[r1, c1])
            if total <= target or total > target + 20:
                continue
            if (residuals[r2 + 1, c2 + 1] - residuals[r1, c2 + 1]
                    - residuals[r2 + 1, c1] + residuals[r1, c1]) == 0:
                continue
            e = total - target
            if (e + target - 1) // target + 1 > horizon - cut:
                continue
            if not _subset_possible(board, rects[a], e):
                continue
            # Reservoir sampling gives every eligible goal equal probability.
            eligible += 1
            if rng.integers(0, eligible) == 0:
                goal, excess = a, e
        if goal < 0:
            best_length, best_loss = search_batch(
                initial, rects, lookup, target, horizon, root_mask,
                population, lengths, losses, best_path, best_length, best_loss, rng, 1)
            continue

        gr1, gc1, gr2, gc2 = rects[goal]
        bias = np.zeros(len(rects))
        strength = rng.uniform(0, 8)
        for step in range(cut, parent_length):
            bias[population[parent, step]] = strength
        size_weight = rng.uniform(0.3, 3)
        center_weight = rng.uniform(0, 4)
        progress_weight = rng.uniform(1, 4)
        pending = True
        length = cut
        for step in range(cut, horizon):
            actions = legal_actions(board, lookup, target)
            chosen = -1
            chosen_progress = 0
            if pending and excess == 0:
                for a in actions:
                    if a == goal and (step != 0 or root_mask[a]):
                        chosen = a
                        break
            # Retry unrestricted at the same state if protection blocks all moves.
            for attempt in range(2):
                if chosen >= 0:
                    break
                best_value = -np.inf
                for a in actions:
                    if step == 0 and not root_mask[a]:
                        continue
                    r1, c1, r2, c2 = rects[a]
                    progress = 0
                    if pending:
                        diag1 = (board[gr1, gc1] != 0 and board[gr2, gc2] != 0
                                 and not (r1 <= gr1 <= r2 and c1 <= gc1 <= c2)
                                 and not (r1 <= gr2 <= r2 and c1 <= gc2 <= c2))
                        diag2 = (board[gr1, gc2] != 0 and board[gr2, gc1] != 0
                                 and not (r1 <= gr1 <= r2 and c1 <= gc2 <= c2)
                                 and not (r1 <= gr2 <= r2 and c1 <= gc1 <= c2))
                        if not (diag1 or diag2):
                            continue
                        for r in range(max(r1, gr1), min(r2, gr2) + 1):
                            for c in range(max(c1, gc1), min(c2, gc2) + 1):
                                progress += board[r, c]
                        if progress > excess:
                            continue
                        remaining = excess - progress
                        if (remaining + target - 1) // target + 1 > horizon - step - 1:
                            continue
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    dist = np.hypot((r1 + r2) / 2 - h / 2,
                                    (c1 + c2) / 2 - w / 2) / max(h, w)
                    value = (progress_weight * progress + bias[a]
                             - size_weight * np.log(area) - center_weight * dist
                             - np.log(-np.log(max(rng.random(), 1e-15))))
                    if value > best_value:
                        best_value, chosen, chosen_progress = value, a, progress
                if chosen < 0:
                    if not pending:
                        break
                    pending = False
            if chosen < 0:
                break
            path[step] = chosen
            length += 1
            r1, c1, r2, c2 = rects[chosen]
            board[r1:r2 + 1, c1:c2 + 1] = 0
            if chosen == goal:
                pending = False
            elif pending:
                excess -= chosen_progress

        loss = np.count_nonzero(board)
        if loss < best_loss:
            best_loss, best_length = loss, length
            best_path[:] = path
        if loss <= losses[parent] or rng.random() < np.exp((losses[parent] - loss) / 2):
            population[parent] = path
            lengths[parent], losses[parent] = length, loss
        if rng.random() < 0.05:
            population[parent] = best_path
            lengths[parent], losses[parent] = best_length, best_loss
    return best_length, best_loss


class GoalSearchStrategy(PopulationSearchStrategy):
    """Shared-budget hybrid repair and stochastic rectangle activation trials."""

    _search_batch = staticmethod(goal_batch)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "GoalSearch"

    @staticmethod
    def warmup():
        """Explicitly compile hybrid, fallback, and goal paths before timed use."""
        initial = np.array([[2, 5, 5, 0], [3, 5, 0, 0], [4, 6, 1, 9]], dtype=np.int32)
        h, w = initial.shape
        rects = np.array([
            (r1, c1, r2, c2)
            for r1 in range(h) for c1 in range(w)
            for r2 in range(r1, h) for c2 in range(c1, w)
            if r1 != r2 or c1 != c2
        ], dtype=np.int32)
        lookup = np.full((h, w, h, w), -1, dtype=np.int32)
        for a, (r1, c1, r2, c2) in enumerate(rects):
            lookup[r1, c1, r2, c2] = a
        horizon = 4
        population = np.full((2, horizon), -1, dtype=np.int32)
        lengths = np.zeros(2, dtype=np.int32)
        loss = np.count_nonzero(initial)
        losses = np.full(2, loss, dtype=np.int32)
        best_path = np.full(horizon, -1, dtype=np.int32)
        root_mask = np.ones(len(rects), dtype=np.bool_)
        rng = np.random.default_rng(0)
        # Start with a goal-only trial so hybrid cannot clear its seed first.
        best_length, best_loss = goal_batch(
            initial, rects, lookup, 10, horizon, root_mask, population,
            lengths, losses, best_path, 0, loss, rng, 1)
        goal_batch(initial, rects, lookup, 10, horizon, root_mask, population,
                   lengths, losses, best_path, best_length, best_loss, rng, 4)

"""Order-preserving ruin-and-recreate search. Experimental, CPU only."""
import numpy as np
from numba import njit

from rs10env.fast_search import PopulationSearchStrategy, legal_actions, search_batch


@njit(cache=True)
def repair_batch(initial, rects, lookup, target, horizon, root_mask,
                 population, lengths, losses, best_path, best_length,
                 best_loss, rng, count, spatial=False):
    h, w = initial.shape
    for trial in range(count):
        parent = rng.integers(0, len(lengths))
        board = initial.copy()
        path = np.full(horizon, -1, dtype=np.int32)
        size_weight = rng.uniform(0.3, 3)
        center_weight = rng.uniform(0, 4)
        # Destroy a small set anywhere in the trajectory, not its whole suffix.
        banned = np.zeros(len(rects), dtype=np.bool_)
        if lengths[parent]:
            if spatial:
                terminal = initial.copy()
                for j in range(lengths[parent]):
                    a = population[parent, j]
                    r1, c1, r2, c2 = rects[a]
                    terminal[r1:r2 + 1, c1:c2 + 1] = 0
                remaining = np.flatnonzero(terminal)
                anchor = rng.integers(0, h * w)
                if len(remaining):
                    anchor = remaining[rng.integers(0, len(remaining))]
                ar, ac = anchor // w, anchor % w
                radius = rng.integers(1, 4)
                for j in range(lengths[parent]):
                    a = population[parent, j]
                    r1, c1, r2, c2 = rects[a]
                    # L-infinity distance from a stranded cell to an action rectangle.
                    distance = max(0, r1 - ar, ar - r2, c1 - ac, ac - c2)
                    if distance <= radius:
                        banned[a] = True
            else:
                removals = rng.integers(1, max(2, lengths[parent] // 5))
                for _ in range(removals):
                    banned[population[parent, rng.integers(0, lengths[parent])]] = True
        length = 0
        for step in range(horizon):
            chosen = -1
            # Preserve the earliest still-legal old move, even across a broken dependency.
            for j in range(lengths[parent]):
                a = population[parent, j]
                if banned[a] or (step == 0 and not root_mask[a]):
                    continue
                r1, c1, r2, c2 = rects[a]
                if not ((board[r1, c1] and board[r2, c2]) or
                        (board[r1, c2] and board[r2, c1])):
                    continue
                if board[r1:r2 + 1, c1:c2 + 1].sum() == target:
                    chosen = a
                    break
            if chosen < 0:
                actions = legal_actions(board, lookup, target)
                value_best = -np.inf
                for a in actions:
                    if step == 0 and not root_mask[a]:
                        continue
                    r1, c1, r2, c2 = rects[a]
                    area = (r2 - r1 + 1) * (c2 - c1 + 1)
                    dist = np.hypot((r1 + r2) / 2 - h / 2,
                                    (c1 + c2) / 2 - w / 2) / max(h, w)
                    value = (-size_weight * np.log(area) - center_weight * dist
                             - np.log(-np.log(max(rng.random(), 1e-15))))
                    if banned[a]:
                        value -= 20
                    if value > value_best:
                        value_best, chosen = value, a
            if chosen < 0:
                break
            path[step] = chosen
            length += 1
            r1, c1, r2, c2 = rects[chosen]
            board[r1:r2 + 1, c1:c2 + 1] = 0
        loss = np.count_nonzero(board)
        if loss < best_loss:
            best_loss, best_length = loss, length
            best_path[:] = path
        if loss <= losses[parent] or rng.random() < np.exp((losses[parent] - loss) / 1.0):
            population[parent] = path
            lengths[parent], losses[parent] = length, loss
        if rng.random() < 0.02:
            population[parent] = best_path
            lengths[parent], losses[parent] = best_length, best_loss
    return best_length, best_loss


class RepairSearchStrategy(PopulationSearchStrategy):
    """Sparse trajectory destruction followed by ordered legal-action repair."""

    _search_batch = staticmethod(repair_batch)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "RepairSearch"

    @staticmethod
    def warmup():
        repair_batch(np.array([[5, 5]], dtype=np.int32),
                     np.array([[0, 0, 0, 1]], dtype=np.int32),
                     np.zeros((1, 2, 1, 2), dtype=np.int32), 10, 1,
                     np.ones(1, dtype=np.bool_), np.full((1, 1), -1, dtype=np.int32),
                     np.zeros(1, dtype=np.int32), np.full(1, 2, dtype=np.int32),
                     np.full(1, -1, dtype=np.int32), 0, 2, np.random.default_rng(0), 2)


@njit(cache=True)
def hybrid_batch(initial, rects, lookup, target, horizon, root_mask,
                 population, lengths, losses, best_path, best_length,
                 best_loss, rng, count):
    # The two neighborhoods share candidates and a best-ever solution.
    first = count // 2
    best_length, best_loss = repair_batch(
        initial, rects, lookup, target, horizon, root_mask,
        population, lengths, losses, best_path, best_length, best_loss, rng, first)
    return search_batch(initial, rects, lookup, target, horizon, root_mask,
                        population, lengths, losses, best_path, best_length,
                        best_loss, rng, count - first)


class HybridRepairStrategy(RepairSearchStrategy):
    """Variable neighborhoods: sparse destruction and stochastic suffix repair."""

    _search_batch = staticmethod(hybrid_batch)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "HybridRepair"

    @staticmethod
    def warmup():
        hybrid_batch(np.array([[5, 5]], dtype=np.int32),
                     np.array([[0, 0, 0, 1]], dtype=np.int32),
                     np.zeros((1, 2, 1, 2), dtype=np.int32), 10, 1,
                     np.ones(1, dtype=np.bool_), np.full((1, 1), -1, dtype=np.int32),
                     np.zeros(1, dtype=np.int32), np.full(1, 2, dtype=np.int32),
                     np.full(1, -1, dtype=np.int32), 0, 2, np.random.default_rng(0), 2)


@njit(cache=True)
def spatial_batch(initial, rects, lookup, target, horizon, root_mask,
                  population, lengths, losses, best_path, best_length,
                  best_loss, rng, count):
    first = count // 2
    best_length, best_loss = repair_batch(
        initial, rects, lookup, target, horizon, root_mask, population, lengths,
        losses, best_path, best_length, best_loss, rng, first, True)
    return hybrid_batch(initial, rects, lookup, target, horizon, root_mask,
                        population, lengths, losses, best_path, best_length,
                        best_loss, rng, count - first)


class SpatialRepairStrategy(HybridRepairStrategy):
    """Rebuild neighborhoods of stranded cells using rectangle distance."""

    _search_batch = staticmethod(spatial_batch)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "SpatialRepair"

    @staticmethod
    def warmup():
        spatial_batch(np.array([[5, 5]], dtype=np.int32),
                      np.array([[0, 0, 0, 1]], dtype=np.int32),
                      np.zeros((1, 2, 1, 2), dtype=np.int32), 10, 1,
                      np.ones(1, dtype=np.bool_), np.full((1, 1), -1, dtype=np.int32),
                      np.zeros(1, dtype=np.int32), np.full(1, 2, dtype=np.int32),
                      np.full(1, -1, dtype=np.int32), 0, 2, np.random.default_rng(0), 4)

"""Optional compiled population search (install rs10env[search])."""
import time

import numpy as np
import torch
from numba import njit

from rs10env.strategies import Strategy


@njit(cache=True)
def legal_actions(board, lookup, target):
    h, w = board.shape
    actions = np.empty(lookup.size, dtype=np.int32)
    count = 0
    for c1 in range(w):
        strip = np.zeros(h, dtype=np.int32)
        for c2 in range(c1, w):
            strip += board[:, c2]
            for r1 in range(h):
                if strip[r1] == 0:
                    continue
                total = 0
                for r2 in range(r1, h):
                    total += strip[r2]
                    if total > target:
                        break
                    if total != target or strip[r2] == 0:
                        continue
                    if ((board[r1, c1] != 0 and board[r2, c2] != 0)
                            or (board[r1, c2] != 0 and board[r2, c1] != 0)):
                        action = lookup[r1, c1, r2, c2]
                        if action >= 0:
                            actions[count] = action
                            count += 1
    return actions[:count]


@njit(cache=True)
def search_batch(initial, rects, lookup, target, horizon, root_mask,
                 population, lengths, losses, best_path, best_length,
                 best_loss, rng, count):
    h, w = initial.shape
    for trial in range(count):
        parent = rng.integers(0, len(lengths))
        board = initial.copy()
        path = np.full(horizon, -1, dtype=np.int32)
        cut = 0
        if lengths[parent] > 0 and rng.random() > 0.05:
            cut = rng.integers(0, lengths[parent])
        for step in range(cut):
            a = population[parent, step]
            path[step] = a
            r1, c1, r2, c2 = rects[a]
            board[r1:r2 + 1, c1:c2 + 1] = 0
        bias = np.zeros(len(rects))
        strength = rng.uniform(0, 14)
        for step in range(cut, lengths[parent]):
            bias[population[parent, step]] = strength
        size_weight = rng.uniform(0.3, 3)
        center_weight = rng.uniform(0, 4)
        length = cut
        for step in range(cut, horizon):
            actions = legal_actions(board, lookup, target)
            best_value = -np.inf
            chosen = -1
            for a in actions:
                if step == 0 and not root_mask[a]:
                    continue
                r1, c1, r2, c2 = rects[a]
                area = (r2 - r1 + 1) * (c2 - c1 + 1)
                dist = np.sqrt(((r1 + r2) / 2 - h / 2) ** 2
                               + ((c1 + c2) / 2 - w / 2) ** 2) / max(h, w)
                value = (bias[a] - size_weight * np.log(area) - center_weight * dist
                         - np.log(-np.log(max(rng.random(), 1e-15))))
                if step == cut and cut < lengths[parent] and a == population[parent, cut]:
                    value -= 30
                if value > best_value:
                    chosen = a
                    best_value = value
            if chosen < 0:
                break
            path[step] = chosen
            length += 1
            r1, c1, r2, c2 = rects[chosen]
            board[r1:r2 + 1, c1:c2 + 1] = 0
        loss = np.count_nonzero(board)
        if loss < best_loss:
            best_loss = loss
            best_path[:] = path
            best_length = length
        # A separate best-ever plan allows the population to cross small valleys.
        if loss <= losses[parent] or rng.random() < np.exp((losses[parent] - loss) / 2):
            population[parent] = path
            lengths[parent] = length
            losses[parent] = loss
        if rng.random() < 0.05:
            population[parent] = best_path
            lengths[parent] = best_length
            losses[parent] = best_loss
    return best_length, best_loss


class PopulationSearchStrategy(Strategy):
    """Time-bounded population repair; JIT warmup is explicit, not hidden.

    time_budget covers planning, not environment execution or compilation.
    max_rollouts provides deterministic experiments independently of CPU speed.
    """

    _search_batch = staticmethod(search_batch)

    def __init__(self, time_budget=8.0, population_size=12, max_rollouts=1000000,
                 seed=None, device=None):
        super().__init__("PopulationSearch", seed, device)
        if not np.isfinite(time_budget) or time_budget <= 0:
            raise ValueError("time_budget must be positive and finite")
        for name, value in [("population_size", population_size), ("max_rollouts", max_rollouts)]:
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.time_budget = time_budget
        self.population_size = population_size
        self.max_rollouts = max_rollouts
        self._rng = np.random.default_rng(self._seed)
        self._plan = []
        self._expected = None
        self._context = None

    @staticmethod
    def warmup():
        """Compile before latency-sensitive use; first compilation can exceed 10s."""
        search_batch(np.array([[5, 5]], dtype=np.int32),
                     np.array([[0, 0, 0, 1]], dtype=np.int32),
                     np.zeros((1, 2, 1, 2), dtype=np.int32), 10, 1,
                     np.ones(1, dtype=np.bool_), np.full((1, 1), -1, dtype=np.int32),
                     np.zeros(1, dtype=np.int32), np.full(1, 2, dtype=np.int32),
                     np.full(1, -1, dtype=np.int32), 0, 2, np.random.default_rng(0), 1)

    @torch.no_grad()
    def get_action(self, env, valid_actions_mask):
        if not valid_actions_mask.any():
            return self._get_fallback_action(valid_actions_mask, env)
        start = time.perf_counter()
        board = env.board_2d.cpu().numpy()
        horizon = env.max_steps - env.step_count
        context = (env.H, env.W, env.target_sum, horizon)
        if (not self._plan or context != self._context
                or not np.array_equal(board, self._expected)
                or not valid_actions_mask[self._plan[0]]):
            if horizon < 1:
                raise ValueError("Cannot search after the episode horizon")
            rects = env.all_rects.cpu().numpy()
            lookup = np.full((env.H, env.W, env.H, env.W), -1, dtype=np.int32)
            for a, (r1, c1, r2, c2) in enumerate(rects):
                lookup[r1, c1, r2, c2] = a
            population = np.full((self.population_size, horizon), -1, dtype=np.int32)
            lengths = np.zeros(self.population_size, dtype=np.int32)
            losses = np.full(self.population_size, np.count_nonzero(board), dtype=np.int32)
            best_path = np.full(horizon, -1, dtype=np.int32)
            best_length, best_loss = 0, np.count_nonzero(board)
            root_mask = valid_actions_mask.cpu().numpy()
            self.rollouts = 0
            self.search_scores = [int(best_loss)]
            while self.rollouts < self.max_rollouts:
                if self.rollouts and time.perf_counter() - start >= self.time_budget:
                    break
                count = min(16, self.max_rollouts - self.rollouts)
                best_length, best_loss = self._search_batch(
                    board, rects, lookup, env.target_sum, horizon, root_mask,
                    population, lengths, losses, best_path, best_length, best_loss,
                    self._rng, count)
                self.rollouts += count
                self.search_scores.append(int(best_loss))
                if best_loss == 0:
                    break
            self._plan = best_path[:best_length].tolist()
            self.planning_seconds = time.perf_counter() - start
        action = self._plan.pop(0)
        self._expected = board.copy()
        r1, c1, r2, c2 = env.all_rects[action].tolist()
        self._expected[r1:r2 + 1, c1:c2 + 1] = 0
        self._context = (env.H, env.W, env.target_sum, horizon - 1)
        return torch.tensor(action, dtype=torch.int64, device=valid_actions_mask.device)

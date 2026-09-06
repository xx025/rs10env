"""Experimental CPU depth-1/2 NRPA-style search (requires rs10env[search]).

Depth 2 retains an outer policy across inner searches. Call warmup() before
timing. A fixed max_rollouts overrides the wall-clock budget and early stopping.
"""

import time

import numpy as np
import torch
from numba import njit

from rs10env.fast_search import legal_actions
from rs10env.strategies import Strategy


@njit(cache=True)
def _adapt(initial, rects, lookup, target, root_mask, policy, path, length,
           learning_rate):
    # All probabilities use the pre-update policy, as in NRPA adaptation.
    # Replaying states also ensures negatives include only legal actions.
    updated = policy.copy()
    board = initial.copy()
    initial_live = np.count_nonzero(initial)
    live = initial_live
    for step in range(length):
        bucket = min(3, 4 * (initial_live - live) // max(1, initial_live))
        actions = legal_actions(board, lookup, target)
        maximum = -np.inf
        for a in actions:
            if step != 0 or root_mask[a]:
                maximum = max(maximum, policy[bucket, a])
        total = 0.0
        for a in actions:
            if step != 0 or root_mask[a]:
                total += np.exp(policy[bucket, a] - maximum)
        for a in actions:
            if step != 0 or root_mask[a]:
                updated[bucket, a] -= (learning_rate
                                      * np.exp(policy[bucket, a] - maximum) / total)
        chosen = path[step]
        updated[bucket, chosen] += learning_rate
        r1, c1, r2, c2 = rects[chosen]
        live -= np.count_nonzero(board[r1:r2 + 1, c1:c2 + 1])
        board[r1:r2 + 1, c1:c2 + 1] = 0
    policy[:] = updated


@njit(cache=True)
def _adaptive_batch(initial, rects, lookup, target, horizon, root_mask,
                    geometry, policy, local_path, best_path, stats, rng,
                    first_trial, count, adaptation_steps, learning_rate,
                    levels, outer_steps, outer_policy, outer_path, outer_stats):
    initial_live = np.count_nonzero(initial)
    for trial in range(first_trial, first_trial + count):
        if trial % adaptation_steps == 0:
            if levels == 1 or (trial // adaptation_steps) % outer_steps == 0:
                size_weight = rng.uniform(0.3, 3.0)
                center_weight = rng.uniform(0.0, 4.0)
                for bucket in range(4):
                    for a in range(len(rects)):
                        policy[bucket, a] = (-size_weight * geometry[a, 0]
                                             - center_weight * geometry[a, 1])
                if levels == 2:
                    outer_policy[:] = policy
                    outer_stats[0] = 0
                    outer_stats[1] = initial_live + 1
            if levels == 2:
                policy = outer_policy.copy()
            stats[0] = 0
            stats[1] = initial_live + 1

        board = initial.copy()
        path = np.full(horizon, -1, dtype=np.int32)
        length = 0
        live = initial_live
        for step in range(horizon):
            bucket = min(3, 4 * (initial_live - live) // max(1, initial_live))
            actions = legal_actions(board, lookup, target)
            chosen = -1
            best_value = -np.inf
            for a in actions:
                if step == 0 and not root_mask[a]:
                    continue
                # Gumbel-max samples the legal-action softmax without a sum.
                value = policy[bucket, a] - np.log(-np.log(max(rng.random(), 1e-15)))
                if value > best_value:
                    chosen = a
                    best_value = value
            if chosen < 0:
                break
            path[step] = chosen
            length += 1
            r1, c1, r2, c2 = rects[chosen]
            live -= np.count_nonzero(board[r1:r2 + 1, c1:c2 + 1])
            board[r1:r2 + 1, c1:c2 + 1] = 0

        if live < stats[3]:
            best_path[:] = path
            stats[2] = length
            stats[3] = live
        if live < stats[1] or (live == stats[1] and rng.integers(0, 2) == 0):
            local_path[:] = path
            stats[0] = length
            stats[1] = live

        _adapt(initial, rects, lookup, target, root_mask, policy, local_path,
               stats[0], learning_rate)
        if levels == 2 and (trial + 1) % adaptation_steps == 0:
            if (stats[1] < outer_stats[1]
                    or (stats[1] == outer_stats[1] and rng.integers(0, 2) == 0)):
                outer_path[:] = local_path
                outer_stats[:] = stats[:2]
            _adapt(initial, rects, lookup, target, root_mask, outer_policy,
                   outer_path, outer_stats[0], learning_rate)
    return policy


class AdaptiveSearchStrategy(Strategy):
    """Depth-1/2 best-trajectory softmax adaptation with geometry restarts.

    time_budget defaults to 8 seconds per new plan, excluding execution and
    explicit warmup. The deadline is soft: at least one four-rollout batch is
    run, and a batch (including any due outer updates) cannot be interrupted.
    The deadline is checked between batches, not just between inner runs.
    max_rollouts, when supplied, ignores
    the deadline and solved-board early stopping, running exactly that many
    rollouts (including a partial final batch/inner run) for fixed-work experiments.
    Randomness is local to the strategy; reproducibility requires the same
    sequence of input states.

    device controls the Strategy interface only; all search is CPU Numba.
    levels=1 preserves independent restarts after adaptation_steps rollout/update
    pairs. levels=2 starts each such inner run from a copy of the outer policy,
    then adapts the outer policy toward the best inner solution in the current
    cycle. After outer_steps completed inner runs, the cycle restarts. Incomplete
    inner runs still contribute to the returned best-ever path, but do not update
    the outer policy. Both levels use the same four live-cell phase buckets.
    search_scores records best remaining live-cell counts (lower is better).
    """

    def __init__(self, time_budget=8.0, seed=None, device=None,
                 max_rollouts=None, adaptation_steps=64, learning_rate=1.0,
                 levels=2, outer_steps=16):
        super().__init__("AdaptiveSearch", seed, device)
        if not np.isfinite(time_budget) or time_budget <= 0:
            raise ValueError("time_budget must be positive and finite")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        for name, value in [("adaptation_steps", adaptation_steps),
                            ("outer_steps", outer_steps),
                            ("max_rollouts", max_rollouts)]:
            if name == "max_rollouts" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(levels, int) or isinstance(levels, bool) or levels not in (1, 2):
            raise ValueError("levels must be 1 or 2")
        self.time_budget = float(time_budget)
        self.max_rollouts = max_rollouts
        self.adaptation_steps = adaptation_steps
        self.learning_rate = float(learning_rate)
        self.levels = levels
        self.outer_steps = outer_steps
        self._rng = np.random.default_rng(self._seed)
        self._plan = []
        self._expected = None
        self._context = None
        self.rollouts = 0
        self.search_scores = []
        self.planning_seconds = 0.0

    @staticmethod
    def warmup():
        """Compile both depths' rollout, adaptation and restart paths with a local RNG.

        Run explicitly before latency-sensitive use; cold compilation may take
        more than 10 seconds. Uses the same contiguous dtypes as get_action.
        """
        for levels in (1, 2):
            _adaptive_batch(
                np.array([[5, 5]], dtype=np.int32),
                np.array([[0, 0, 0, 1]], dtype=np.int32),
                np.zeros((1, 2, 1, 2), dtype=np.int32), 10, 1,
                np.ones(1, dtype=np.bool_), np.zeros((1, 2), dtype=np.float64),
                np.zeros((4, 1), dtype=np.float64),
                np.full(1, -1, dtype=np.int32), np.full(1, -1, dtype=np.int32),
                np.array([0, 3, 0, 3], dtype=np.int64),
                np.random.default_rng(0), 0, 5, 2, 1.0, levels, 2,
                np.zeros((4, 1), dtype=np.float64),
                np.full(1, -1, dtype=np.int32),
                np.array([0, 3], dtype=np.int64))

    @torch.no_grad()
    def get_action(self, env, valid_actions_mask):
        if not valid_actions_mask.any():
            self._plan = []
            self._expected = None
            self._context = None
            return self._get_fallback_action(valid_actions_mask, env)
        start = time.perf_counter()
        board = np.ascontiguousarray(env.board_2d.cpu().numpy(), dtype=np.int32)
        horizon = int(env.max_steps - env.step_count)
        context = (env.H, env.W, env.target_sum, horizon)
        if (not self._plan or context != self._context
                or not np.array_equal(board, self._expected)
                or not valid_actions_mask[self._plan[0]]):
            if horizon < 1:
                raise ValueError("Cannot search after the episode horizon")
            rects = np.ascontiguousarray(env.all_rects.cpu().numpy(), dtype=np.int32)
            lookup = np.full((env.H, env.W, env.H, env.W), -1, dtype=np.int32)
            lookup[tuple(rects.T)] = np.arange(len(rects), dtype=np.int32)
            r1, c1, r2, c2 = rects.T
            geometry = np.empty((len(rects), 2), dtype=np.float64)
            geometry[:, 0] = np.log((r2 - r1 + 1) * (c2 - c1 + 1))
            geometry[:, 1] = np.sqrt(
                ((r1 + r2) / 2 - env.H / 2) ** 2
                + ((c1 + c2) / 2 - env.W / 2) ** 2) / max(env.H, env.W)
            policy = np.empty((4, len(rects)), dtype=np.float64)
            local_path = np.full(horizon, -1, dtype=np.int32)
            best_path = np.full(horizon, -1, dtype=np.int32)
            live = np.count_nonzero(board)
            # Local length/loss and best-ever length/loss, shared across batches.
            stats = np.array([0, live + 1, 0, live + 1], dtype=np.int64)
            outer_policy = np.empty_like(policy)
            outer_path = np.full(horizon, -1, dtype=np.int32)
            outer_stats = np.array([0, live + 1], dtype=np.int64)
            root_mask = np.ascontiguousarray(valid_actions_mask.cpu().numpy(),
                                             dtype=np.bool_)
            self.rollouts = 0
            self.search_scores = [int(live)]
            while True:
                if self.max_rollouts is not None:
                    count = min(4, self.max_rollouts - self.rollouts)
                    if count == 0:
                        break
                else:
                    if self.rollouts and time.perf_counter() - start >= self.time_budget:
                        break
                    count = 4
                policy = _adaptive_batch(
                    board, rects, lookup, int(env.target_sum), horizon, root_mask,
                    geometry, policy, local_path, best_path, stats, self._rng,
                    self.rollouts, count, self.adaptation_steps, self.learning_rate,
                    self.levels, self.outer_steps, outer_policy, outer_path, outer_stats)
                self.rollouts += count
                self.search_scores.append(int(stats[3]))
                if stats[3] == 0 and self.max_rollouts is None:
                    break
            self._plan = best_path[:stats[2]].tolist()
            self.planning_seconds = time.perf_counter() - start
            if not self._plan:
                raise RuntimeError("No rollout action matched the supplied valid-action mask")
        action = self._plan.pop(0)
        self._expected = board.copy()
        r1, c1, r2, c2 = env.all_rects[action].tolist()
        self._expected[r1:r2 + 1, c1:c2 + 1] = 0
        self._context = (env.H, env.W, env.target_sum, horizon - 1)
        return torch.tensor(action, dtype=torch.int64, device=valid_actions_mask.device)

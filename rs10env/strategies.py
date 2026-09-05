"""RS10 heuristic strategies and NumPy rollout search."""
import torch
import numpy as np
from typing import Tuple, List, Optional

from rs10env.env import RS10Env


class Strategy:
    """策略基类"""

    DEFAULT_SEED: int = 68

    def __init__(self, name: str, seed: Optional[int] = None, device: Optional[str] = None):
        self.name = name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._seed = seed if seed is not None else self.DEFAULT_SEED
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self._seed)

    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _get_fallback_action(self, valid_actions_mask: torch.Tensor, env: RS10Env) -> torch.Tensor:
        """无有效动作时的回退：从有效动作中随机选（一般不应出现）。"""
        valid_indices = torch.where(valid_actions_mask)[0]
        if valid_indices.numel() == 0:
            return torch.tensor(0, device=valid_actions_mask.device, dtype=torch.int64)
        idx = torch.randint(0, valid_indices.numel(), (1,), device=valid_indices.device, generator=self._generator)
        return valid_indices[idx].squeeze()


class RandomStrategy(Strategy):
    """随机策略：从有效动作中均匀随机选择"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("Random", seed, device)

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        valid_indices = torch.where(valid_actions_mask)[0]
        idx = torch.randint(0, valid_indices.numel(), (1,), device=valid_indices.device, generator=self._generator)
        return valid_indices[idx].squeeze()


class GreedyStrategy(Strategy):
    """贪心策略：优先选择消除方块最多的矩形"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("Greedy", seed, device)

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        rect_counts = (env.rect_masks * (env.board_2d > 0)).sum(dim=(1, 2))
        scores = rect_counts.float().masked_fill(~valid_actions_mask, float("-inf"))
        best_action = scores.argmax()
        if scores[best_action] == float("-inf"):
            return self._get_fallback_action(valid_actions_mask, env)
        return best_action


class CenterBiasStrategy(Strategy):
    """中心偏好策略：距离棋盘中心越近越好"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("CenterBias", seed, device)
        self._distances = None

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        device = valid_actions_mask.device
        if self._distances is None:
            r1, c1, r2, c2 = env.all_rects[:, 0], env.all_rects[:, 1], env.all_rects[:, 2], env.all_rects[:, 3]
            r1, r2, c1, c2 = r1.to(device), r2.to(device), c1.to(device), c2.to(device)
            rect_centers_h = (r1 + r2).float() / 2.0
            rect_centers_w = (c1 + c2).float() / 2.0
            center_h = torch.tensor(env.H / 2.0, dtype=torch.float32, device=device)
            center_w = torch.tensor(env.W / 2.0, dtype=torch.float32, device=device)
            self._distances = torch.sqrt((rect_centers_h - center_h) ** 2 + (rect_centers_w - center_w) ** 2)
        distances = self._distances.masked_fill(~valid_actions_mask, float("inf"))
        return distances.argmin()


class LargeRectStrategy(Strategy):
    """大矩形策略：偏好面积较大的矩形"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("LargeRect", seed, device)

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        scores = env.rect_masks_area.float().masked_fill(~valid_actions_mask, float("-inf"))
        max_score = scores.max()
        if max_score == float("-inf"):
            return self._get_fallback_action(valid_actions_mask, env)
        best_mask = (scores == max_score) & valid_actions_mask
        best_indices = torch.where(best_mask)[0]
        idx = torch.randint(0, best_indices.numel(), (1,), device=best_indices.device, generator=self._generator)
        return best_indices[idx].squeeze()


class SmallRectStrategy(Strategy):
    """小矩形策略：偏好面积较小的矩形"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("SmallRect", seed, device)

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        scores = env.rect_masks_area.float().masked_fill(~valid_actions_mask, float("inf"))
        min_score = scores.min()
        if min_score == float("inf"):
            return self._get_fallback_action(valid_actions_mask, env)
        best_mask = (scores == min_score) & valid_actions_mask
        best_indices = torch.where(best_mask)[0]
        idx = torch.randint(0, best_indices.numel(), (1,), device=best_indices.device, generator=self._generator)
        return best_indices[idx].squeeze()


class CenterSmallRectStrategy(Strategy):
    """中心+小矩形策略：靠近中心且面积较小"""

    def __init__(self, center_weight: float = 0.5, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("CenterSmallRect", seed, device)
        self.center_weight = center_weight
        self._distances = None

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        device = valid_actions_mask.device
        if self._distances is None:
            r1, c1, r2, c2 = env.all_rects[:, 0], env.all_rects[:, 1], env.all_rects[:, 2], env.all_rects[:, 3]
            r1, r2, c1, c2 = r1.to(device), r2.to(device), c1.to(device), c2.to(device)
            rect_centers_h = (r1 + r2).float() / 2.0
            rect_centers_w = (c1 + c2).float() / 2.0
            center_h = torch.tensor(env.H / 2.0, dtype=torch.float32, device=device)
            center_w = torch.tensor(env.W / 2.0, dtype=torch.float32, device=device)
            self._distances = torch.sqrt((rect_centers_h - center_h) ** 2 + (rect_centers_w - center_w) ** 2)
        distances = self._distances.masked_fill(~valid_actions_mask, float("inf"))
        areas = env.rect_masks_area.float().masked_fill(~valid_actions_mask, float("inf"))
        valid_distances = distances[valid_actions_mask]
        valid_areas = areas[valid_actions_mask]
        if valid_distances.numel() > 0:
            min_dist = valid_distances.min()
            max_dist = valid_distances.max()
            normalized_dist = (distances - min_dist) / (max_dist - min_dist) if max_dist > min_dist else torch.zeros_like(distances)
        else:
            normalized_dist = distances
        if valid_areas.numel() > 0:
            min_area = valid_areas.min()
            max_area = valid_areas.max()
            normalized_area = (areas - min_area) / (max_area - min_area) if max_area > min_area else torch.zeros_like(areas)
        else:
            normalized_area = areas
        scores = -(self.center_weight * normalized_dist + (1 - self.center_weight) * normalized_area)
        scores = scores.masked_fill(~valid_actions_mask, float("-inf"))
        return scores.argmax()


class HybridStrategy(Strategy):
    """混合策略：多个策略的加权组合"""

    def __init__(self, strategies: List[Tuple[Strategy, float]], seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("Hybrid", seed, device)
        self.strategies = strategies

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor([w for _, w in self.strategies], dtype=torch.float32, device=valid_actions_mask.device)
        weights = weights / weights.sum()
        idx = torch.multinomial(weights, 1, generator=self._generator).item()
        return self.strategies[idx][0].get_action(env, valid_actions_mask)


class EpsilonGreedyStrategy(Strategy):
    """Epsilon-贪心：以 1-ε 选贪心，ε 随机探索"""

    def __init__(self, epsilon: float = 0.1, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__(f"EpsilonGreedy(eps={epsilon})", seed, device)
        self.epsilon = epsilon
        self.greedy = GreedyStrategy(seed, device)
        self.random = RandomStrategy(seed, device)

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        if torch.rand(1, device=valid_actions_mask.device, generator=self._generator).item() < self.epsilon:
            return self.random.get_action(env, valid_actions_mask)
        return self.greedy.get_action(env, valid_actions_mask)


class MaxFutureMovesStrategy(Strategy):
    """最大化未来可行步数：选执行后可行步数最多的动作"""

    def __init__(self, seed: Optional[int] = None, device: Optional[str] = None):
        super().__init__("MaxFutureMoves", seed, device)
        self._sim_env = RS10Env(device=device)
        self._sim_env_device = device

    def _ensure_sim_env_device(self, env: RS10Env) -> None:
        if (self._sim_env.device != env.device or self._sim_env.H != env.H
                or self._sim_env.W != env.W or self._sim_env.target_sum != env.target_sum):
            self._sim_env = RS10Env(device=env.device, H=env.H, W=env.W, target_sum=env.target_sum)
            self._sim_env_device = env.device

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        self._ensure_sim_env_device(env)
        valid_indices = torch.where(valid_actions_mask)[0]
        if valid_indices.numel() == 1:
            return valid_indices[0]
        self._sim_env.reset(board=env.board_2d.clone())
        num_valid = valid_indices.shape[0]
        future_moves_counts = torch.zeros(num_valid, dtype=torch.int32, device=valid_indices.device)
        for i in range(num_valid):
            action = valid_indices[i].item()
            self._sim_env.board_2d.copy_(env.board_2d)
            self._sim_env.board_2d[env.rect_masks[action]] = 0
            new_valid_mask = self._sim_env.get_valid_actions_mask_prefix()
            future_moves_counts[i] = new_valid_mask.sum().to(torch.int32)
        best_idx = future_moves_counts.argmax()
        best_action = valid_indices[best_idx]
        max_count = future_moves_counts[best_idx]
        best_mask = future_moves_counts == max_count
        best_candidates = valid_indices[best_mask]
        if best_candidates.numel() > 1:
            idx = torch.randint(0, best_candidates.numel(), (1,), device=best_candidates.device, generator=self._generator)
            return best_candidates[idx].squeeze()
        return best_action


class MultiStartStrategy(Strategy):
    """Search complete randomized rollouts and execute the best legal plan.

    Planning uses batched NumPy prefix sums on CPU, including for CUDA envs.
    The remaining plan is reused only while the observed state matches it.
    """

    def __init__(self, num_rollouts: int = 128, seed: Optional[int] = None,
                 device: Optional[str] = None):
        super().__init__("MultiStart", seed, device)
        if not isinstance(num_rollouts, int) or num_rollouts < 1:
            raise ValueError("num_rollouts must be a positive integer")
        self.num_rollouts = num_rollouts
        self._rng = np.random.default_rng(self._seed)
        self._plan = []
        self._expected = None
        self._context = None

    @torch.no_grad()
    def get_action(self, env: RS10Env, valid_actions_mask: torch.Tensor) -> torch.Tensor:
        if not valid_actions_mask.any():
            return self._get_fallback_action(valid_actions_mask, env)
        board = env.board_2d.cpu().numpy()
        remaining = env.max_steps - env.step_count
        context = (env.H, env.W, env.target_sum, remaining)
        if (not self._plan or self._context != context
                or not np.array_equal(board, self._expected)
                or not valid_actions_mask[self._plan[0]]):
            r1, c1, r2, c2 = env.all_rects.cpu().numpy().T
            area = (r2 - r1 + 1) * (c2 - c1 + 1)
            distance = np.hypot((r1 + r2) / 2 - env.H / 2,
                                (c1 + c2) / 2 - env.W / 2)
            distance /= max(env.H, env.W)
            n = self.num_rollouts
            boards = np.broadcast_to(board, (n, env.H, env.W)).copy()
            paths = [[] for _ in range(n)]
            # Diverse small-rectangle/central policies avoid identical rollouts.
            size_weight = self._rng.uniform(0.5, 3.0, (n, 1))
            center_weight = self._rng.uniform(0.0, 4.0, (n, 1))
            prior = -size_weight * np.log(area) - center_weight * distance
            rows = np.arange(n)
            for depth in range(max(1, remaining)):
                prefix = np.pad(boards, ((0, 0), (1, 0), (1, 0)))
                prefix = prefix.cumsum(axis=1).cumsum(axis=2)
                sums = (prefix[:, r2 + 1, c2 + 1] - prefix[:, r1, c2 + 1]
                        - prefix[:, r2 + 1, c1] + prefix[:, r1, c1])
                diagonal = (((boards[:, r1, c1] != 0) & (boards[:, r2, c2] != 0))
                            | ((boards[:, r1, c2] != 0) & (boards[:, r2, c1] != 0)))
                valid = (sums == env.target_sum) & diagonal
                if depth == 0:
                    valid &= valid_actions_mask.cpu().numpy()
                active = valid.any(axis=1)
                if not active.any():
                    break
                scores = prior + self._rng.gumbel(size=valid.shape)
                scores[~valid] = -np.inf
                actions = scores.argmax(axis=1)
                for i in rows[active]:
                    a = int(actions[i])
                    paths[i].append(a)
                    boards[i, r1[a]:r2[a] + 1, c1[a]:c2[a] + 1] = 0
            best = np.count_nonzero(boards, axis=(1, 2)).argmin()
            self._plan = paths[best]
        action = self._plan.pop(0)
        self._expected = board.copy()
        r1, c1, r2, c2 = env.all_rects[action].tolist()
        self._expected[r1:r2 + 1, c1:c2 + 1] = 0
        self._context = (env.H, env.W, env.target_sum, remaining - 1)
        return torch.tensor(action, device=valid_actions_mask.device, dtype=torch.int64)


def create_strategy(strategy_name: str, **kwargs) -> Strategy:
    """根据名称创建策略实例。"""
    strategies = {
        "random": RandomStrategy,
        "greedy": GreedyStrategy,
        "center_bias": CenterBiasStrategy,
        "large_rect": LargeRectStrategy,
        "small_rect": SmallRectStrategy,
        "center_small_rect": CenterSmallRectStrategy,
        "epsilon_greedy": EpsilonGreedyStrategy,
        "max_future_moves": MaxFutureMovesStrategy,
        "multi_start": MultiStartStrategy,
    }
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(**kwargs)

"""RS10 启发式策略（纯 PyTorch）。"""
import torch
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
        if self._sim_env.device != env.device:
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
            self._sim_env._board_3d.copy_(env._board_3d)
            self._sim_env.step(action)
            new_valid_mask = self._sim_env.get_valid_actions_mask()
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
    }
    strategy_class = strategies.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return strategy_class(**kwargs)

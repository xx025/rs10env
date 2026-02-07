"""RS10 游戏环境：Gymnasium 兼容，纯 PyTorch 实现。"""
from gymnasium import Env, spaces
import torch
import numpy as np


class RS10Env(Env):
    """RS10 棋盘环境：在 H×W 棋盘上选择和为 target_sum 的矩形消除。"""

    def __init__(self, *args, **kwargs):
        super().__init__()

        self.device = kwargs.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.H = kwargs.get("H", 16)
        self.W = kwargs.get("W", 10)

        self.target_sum = kwargs.get("target_sum", 10)
        self.strict_action_check = False
        self.board_area = float(self.H * self.W)
        self.board_2d = None
        self.total_zeros = torch.zeros((), device=self.device, dtype=torch.int32)
        self._board_3d = None

        self.all_rects = (
            self._generate_all_rects().to(self.device).to(torch.int32)
        )
        self.rect_masks = (
            self._generate_rect_masks().to(torch.bool).to(self.device)
        )
        self.rect_masks_area = self.rect_masks.sum(dim=(1, 2))
        self.prefix_buf = torch.empty(
            (self.H + 1, self.W + 1), device=self.device, dtype=torch.int32
        )

        self.max_steps = (self.H * self.W) // 2

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(10, self.H, self.W), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(self.all_rects))

        self.step_count = 0
        self.is_done = False
        self.reset()

    @torch.no_grad()
    def reset(self, seed=None, options=None, **kwargs):
        super().reset(seed=seed)
        if "board" in kwargs:
            board = kwargs["board"]
            if isinstance(board, np.ndarray):
                self.board_2d = torch.from_numpy(board).to(torch.int32).to(self.device)
            elif isinstance(board, torch.Tensor):
                self.board_2d = board.to(torch.int32).to(self.device)
            else:
                raise TypeError(f"Unsupported board type: {type(board)}")
        else:
            if seed is not None:
                torch.manual_seed(seed)
            self.board_2d = torch.randint(
                1, 10, (self.H, self.W), dtype=torch.int32, device=self.device
            )
        self._board_3d = torch.nn.functional.one_hot(
            self.board_2d.to(torch.int64), num_classes=10
        ).to(self.device)
        self.step_count = 0
        self.is_done = False
        self.total_zeros = 0
        valid_actions_mask = self.get_valid_actions_mask()
        info = {"step": 0, "action_mask": valid_actions_mask}
        obs = self._board_3d.permute(2, 0, 1)
        return obs, info

    @torch.no_grad()
    def _generate_all_rects(self):
        r1, r2 = torch.triu_indices(self.H, self.H)
        c1, c2 = torch.triu_indices(self.W, self.W)
        r_pairs = torch.stack([r1, r2], dim=1)
        c_pairs = torch.stack([c1, c2], dim=1)
        r_rep = r_pairs.repeat_interleave(c_pairs.size(0), dim=0)
        c_rep = c_pairs.repeat(r_pairs.size(0), 1)
        rects = torch.cat([r_rep, c_rep], dim=1)
        heights = rects[:, 1] - rects[:, 0] + 1
        widths = rects[:, 3] - rects[:, 2] + 1
        areas = heights * widths
        valid_rects = rects[areas >= 2]
        return valid_rects[:, [0, 2, 1, 3]]

    @torch.no_grad()
    def _generate_rect_masks(self):
        N = self.all_rects.shape[0]
        r1, c1, r2, c2 = self.all_rects.unbind(1)
        grid_r = torch.arange(self.H, device=self.device).view(1, self.H, 1)
        grid_c = torch.arange(self.W, device=self.device).view(1, 1, self.W)
        mask = (
            (grid_r >= r1.view(N, 1, 1))
            & (grid_r <= r2.view(N, 1, 1))
            & (grid_c >= c1.view(N, 1, 1))
            & (grid_c <= c2.view(N, 1, 1))
        )
        return mask

    @torch.no_grad()
    def get_valid_actions_mask(self):
        rect_sums = (self.rect_masks * self.board_2d).sum(dim=(1, 2))
        mask_sum = rect_sums == self.target_sum
        r1, c1, r2, c2 = self.all_rects.T
        diag1_ok = (self.board_2d[r1, c1] != 0) & (self.board_2d[r2, c2] != 0)
        diag2_ok = (self.board_2d[r1, c2] != 0) & (self.board_2d[r2, c1] != 0)
        diag_ok = diag1_ok | diag2_ok
        return mask_sum & diag_ok

    @torch.no_grad()
    def get_valid_actions_mask_prefix(self):
        prefix = self.prefix_buf
        prefix.zero_()
        prefix[1:, 1:] = self.board_2d.to(torch.int32)
        prefix.cumsum_(dim=0)
        prefix.cumsum_(dim=1)
        r1, c1, r2, c2 = self.all_rects.T
        rect_sums = (
            prefix[r2 + 1, c2 + 1]
            - prefix[r1, c2 + 1]
            - prefix[r2 + 1, c1]
            + prefix[r1, c1]
        )
        return rect_sums == self.target_sum

    @torch.no_grad()
    def step(self, action, **kwargs):
        terminated = False
        truncated = False
        reward = 0.0
        self.step_count += 1

        valid_actions_mask = kwargs.get("valid_actions_mask", None)
        if valid_actions_mask is None:
            valid_actions_mask = self.get_valid_actions_mask()

        removed = torch.zeros((), device=self.device, dtype=torch.int32)
        if self.strict_action_check:
            pass
        else:
            mask = self.rect_masks[action]
            removed = torch.count_nonzero((self.board_2d > 0) & mask)
            reward = removed.to(torch.float32) / self.board_area
            self.board_2d[mask] = 0
            self._board_3d[mask] = 0
            self._board_3d[mask, 0] = 1

        valid_actions_mask = self.get_valid_actions_mask()
        terminated = (~valid_actions_mask.any()).to(torch.bool).item()

        self.total_zeros += removed
        if terminated:
            bonus = self.total_zeros.to(torch.float32) / self.board_area
            reward += bonus
            self.is_done = True

        if self.step_count >= self.max_steps:
            truncated = True
            self.is_done = True

        info = {
            "step": self.step_count,
            "total_zeros": self.total_zeros,
            "action_mask": valid_actions_mask,
        }
        obs = self._board_3d.permute(2, 0, 1)
        return obs, reward, terminated, truncated, info

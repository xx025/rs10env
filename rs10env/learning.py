"""Small CPU imitation + REINFORCE pilot; no environment reward bonus is used."""

import argparse
import copy
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .env import RS10Env


class PolicyNet(nn.Module):
    """Score legal inclusive (top, left, bottom, right) rectangles on one board."""

    def __init__(self, channels=16, hidden=64, target_sum=10):
        super().__init__()
        self.config = dict(channels=channels, hidden=hidden, target_sum=target_sum)
        self.encoder = nn.Sequential(
            nn.Conv2d(10, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(channels * 6 + 5, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.value = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, board, rects, valid_mask):
        """Return (legal logits, original action indices, scalar value).

        Empty masks return empty logits/indices, never a fallback action.
        Prefix sums pool features without constructing rectangle-sized masks.
        """
        h, w = board.shape
        encoded = F.one_hot(board.long(), 10).permute(2, 0, 1).float()
        features = self.encoder(encoded.unsqueeze(0))[0]
        global_mean = features.mean(dim=(1, 2))
        value = self.value(global_mean).squeeze(-1)
        indices = valid_mask.bool().nonzero(as_tuple=True)[0]
        r1, c1, r2, c2 = rects[indices].long().unbind(1)
        height, width = r2 - r1 + 1, c2 - c1 + 1
        area = (height * width).float()
        # The occupancy channel shares the differentiable prefix-pooling path.
        fields = torch.cat([features, (board != 0).float().unsqueeze(0)])
        prefix = F.pad(fields.cumsum(1).cumsum(2), (1, 0, 1, 0))
        sums = (prefix[:, r2 + 1, c2 + 1] - prefix[:, r1, c2 + 1]
                - prefix[:, r2 + 1, c1] + prefix[:, r1, c1]).T
        geometry = torch.stack([
            width / w, height / h, sums[:, -1] / area,
            sums[:, -1] / (h * w),
            torch.full_like(area, self.config['target_sum'] / (9 * h * w)),
        ], dim=1)
        context = torch.cat([
            sums[:, :-1] / area[:, None], features[:, r1, c1].T,
            features[:, r1, c2].T, features[:, r2, c1].T,
            features[:, r2, c2].T, global_mean.expand(indices.numel(), -1),
            geometry,
        ], dim=1)
        return self.scorer(context).squeeze(-1), indices, value


def play(env, policy=None, *, seed, teacher=None, greedy=True):
    """Collect detached states/actions and area-normalized return-to-go targets."""
    if (policy is None) == (teacher is None):
        raise ValueError('Supply exactly one policy or teacher')
    # Environment resets seed the global torch RNG; isolate board generation.
    with torch.random.fork_rng(devices=[]):
        env.reset(seed=seed)
    trajectory, rewards = [], []
    initial = int(torch.count_nonzero(env.board_2d))
    for _ in range(env.max_steps):
        mask = env.get_valid_actions_mask_prefix()
        if not mask.any():
            break
        board = env.board_2d.clone()
        with torch.no_grad():
            if teacher is not None:
                action = int(teacher.get_action(env, mask))
            else:
                logits, indices, _ = policy(board, env.all_rects, mask)
                local = logits.argmax() if greedy else torch.distributions.Categorical(logits=logits).sample()
                action = int(indices[local])
        if not 0 <= action < mask.numel() or not bool(mask[action]):
            raise ValueError('Refusing to execute an invalid action')
        before = int(torch.count_nonzero(board))
        _, _, terminated, truncated, _ = env.step(action)
        rewards.append((before - int(torch.count_nonzero(env.board_2d))) / env.board_area)
        trajectory.append(dict(board=board, mask=mask.clone(), action=action))
        if terminated or truncated:
            break
    remaining = int(torch.count_nonzero(env.board_2d))
    total = 0.0
    for transition, reward in zip(reversed(trajectory), reversed(rewards)):
        total += reward
        transition['return'] = total
    return trajectory, dict(seed=seed, steps=len(trajectory), cleared=initial - remaining,
                            remaining=remaining, cleared_fraction=(initial - remaining) / env.board_area)


def generate(env, games, budget, seed):
    """Generate HybridRepair demonstrations, with a fresh teacher for each board."""
    from .repair_search import HybridRepairStrategy

    HybridRepairStrategy.warmup()
    samples, metrics = [], []
    for index in range(games):
        game_seed = seed + index
        teacher = HybridRepairStrategy(time_budget=budget, seed=game_seed, device='cpu')
        episode, result = play(env, seed=game_seed, teacher=teacher)
        samples.extend(episode)
        metrics.append(result)
    return samples, metrics


def train_episode(policy, rects, episode, optimizer, *, imitation=False, entropy=0.01):
    """One update, using CE or undiscounted REINFORCE with a detached value baseline.

    Returns are normalized by board area, not standardized within an episode;
    this preserves meaningful targets even for single-action episodes.
    """
    if not episode:
        return 0.0
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for sample in episode:
        logits, indices, value = policy(sample['board'], rects, sample['mask'])
        local = (indices == sample['action']).nonzero(as_tuple=True)[0]
        if local.numel() != 1:
            raise ValueError('Training action must be legal')
        distribution = torch.distributions.Categorical(logits=logits)
        target = value.new_tensor(sample['return'])
        if imitation:
            actor = F.cross_entropy(logits.unsqueeze(0), local)
        else:
            actor = -distribution.log_prob(local[0]) * (target - value.detach())
            actor = actor - entropy * distribution.entropy()
        loss = (actor + 0.5 * F.mse_loss(value, target)) / len(episode)
        # Accumulate gradients, not whole-episode CNN computation graphs.
        loss.backward()
        total += float(loss.detach())
    nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
    optimizer.step()
    return total


def evaluate(env, policy, seeds):
    policy.eval()
    results = [play(env, policy, seed=seed)[1] for seed in seeds]
    return dict(games=results, mean_cleared_fraction=sum(
        game['cleared_fraction'] for game in results) / len(results))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--teacher-games', type=int, default=32)
    parser.add_argument('--teacher-budget', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--rl-games', type=int, default=32)
    parser.add_argument('--eval-games', type=int, default=10)
    parser.add_argument('--seed', type=int, default=100000)
    args = parser.parse_args(argv)
    if args.teacher_games < 1 or args.epochs < 1 or args.eval_games < 1 or args.rl_games < 0:
        parser.error('teacher-games, epochs, eval-games must be positive; rl-games nonnegative')
    if not math.isfinite(args.teacher_budget) or args.teacher_budget <= 0:
        parser.error('teacher-budget must be positive and finite')
    if args.seed < 0 or args.seed + max(args.teacher_games, args.rl_games, args.eval_games) + 1_000_000 >= 2**32:
        parser.error('seed and derived seeds must fit unsigned 32-bit integers')
    if args.teacher_games > 500_000 or args.rl_games > 500_000:
        parser.error('training seed ranges must not overlap')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = RS10Env(H=16, W=10, device='cpu')
    policy = PolicyNet()
    config = {key: value for key, value in vars(args).items() if key != 'output'}
    config.update(H=16, W=10, learning_rate=0.001, entropy=0.01,
                  value_weight=0.5, return_normalization='cleared_cells / board_area',
                  rl_seed_offset=500_000, eval_seed_offset=1_000_000,
                  teacher_timing_note='Wall-clock search may vary despite fixed seeds')
    samples, teacher_metrics = generate(env, args.teacher_games, args.teacher_budget, args.seed)
    if not samples:
        raise ValueError('Teacher generated no legal moves')
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    supervised_losses = []
    for _ in range(args.epochs):
        losses = []
        for index in torch.randperm(len(samples)).tolist():
            losses.append(train_episode(policy, env.all_rects, [samples[index]], optimizer, imitation=True))
        supervised_losses.append(sum(losses) / len(losses))
    torch.save(dict(model_state=policy.state_dict(), model_config=policy.config,
                    config=config), args.output / 'supervised.pt')
    baseline = copy.deepcopy(policy).eval().requires_grad_(False)
    seeds = range(args.seed + 1_000_000, args.seed + 1_000_000 + args.eval_games)
    before = evaluate(env, baseline, seeds)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config['learning_rate'])
    rl_metrics, rl_losses = [], []
    for index in range(args.rl_games):
        episode, result = play(env, policy, seed=args.seed + 500_000 + index, greedy=False)
        rl_losses.append(train_episode(policy, env.all_rects, episode, optimizer, entropy=config['entropy']))
        rl_metrics.append(result)
    torch.save(dict(model_state=policy.state_dict(), model_config=policy.config,
                    config=config), args.output / 'reinforcement.pt')
    after = evaluate(env, policy, seeds)
    metrics = dict(config=config, teacher=teacher_metrics, supervised_losses=supervised_losses,
                   rl_losses=rl_losses, rl_games=rl_metrics, supervised=before, reinforcement=after,
                   paired_mean_change=after['mean_cleared_fraction'] - before['mean_cleared_fraction'])
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: metrics[key] for key in ['paired_mean_change']}))


if __name__ == '__main__':
    main()

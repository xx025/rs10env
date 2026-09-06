"""Bounded background training with atomic checkpoints and validation selection."""
import argparse
import json
import signal
import time
from pathlib import Path

import torch

from rs10env import RS10Env
from rs10env.learning import PolicyNet, play, train_episode, evaluate
from rs10env.repair_search import HybridRepairStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hours', type=float, default=8)
    parser.add_argument('--teacher-games', type=int, default=10000)
    parser.add_argument('--rl-games', type=int, default=50000)
    parser.add_argument('--teacher-budget', type=float, default=0.5)
    parser.add_argument('--validate-every', type=int, default=250)
    parser.add_argument('--validation-games', type=int, default=50)
    parser.add_argument('--seed', type=int, default=3000000)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (not 0 < args.hours <= 24 or min(args.teacher_games, args.rl_games,
            args.validate_every, args.validation_games) < 1
            or not 0 < args.teacher_budget <= 8):
        parser.error('Invalid budgets; hours must be in (0, 24]')
    if max(args.teacher_games, args.rl_games, args.validation_games) >= 500000:
        parser.error('Seed ranges must remain disjoint')
    config = {k: v for k, v in vars(args).items() if k not in ('output', 'resume', 'hours')}
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy = PolicyNet()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0001)
    index, best = 0, -1.0
    if args.resume:
        checkpoint = torch.load(args.output / 'latest.pt', weights_only=True)
        if checkpoint['config'] != config:
            parser.error('Resume configuration differs from checkpoint')
        policy.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        index, best = checkpoint['index'], checkpoint['best_validation']
        torch.set_rng_state(checkpoint['rng_state'])
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    start = time.monotonic()
    deadline = start + args.hours * 3600
    env = RS10Env(device='cpu')
    HybridRepairStrategy.warmup()

    def save(name):
        temporary = args.output / (name + '.tmp')
        torch.save(dict(model_state=policy.state_dict(), model_config=policy.config,
                        optimizer_state=optimizer.state_dict(), index=index,
                        best_validation=best, rng_state=torch.get_rng_state(),
                        config=config), temporary)
        temporary.replace(args.output / name)

    def log(record):
        record['elapsed_seconds'] = time.monotonic() - start
        with (args.output / 'progress.jsonl').open('a') as output:
            output.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record, allow_nan=False), flush=True)

    save('latest.pt')
    log({'event': 'started', 'index': index, 'hours': args.hours})
    total = args.teacher_games + args.rl_games
    try:
        while index < total and not stop and time.monotonic() < deadline:
            imitation = index < args.teacher_games
            if imitation:
                teacher = HybridRepairStrategy(time_budget=args.teacher_budget,
                                               seed=args.seed + index, device='cpu')
                episode, result = play(env, seed=args.seed + index, teacher=teacher)
            else:
                episode, result = play(env, policy, seed=args.seed + 500000 + index
                                       - args.teacher_games, greedy=False)
            loss = train_episode(policy, env.all_rects, episode, optimizer,
                                 imitation=imitation, entropy=0.005)
            index += 1
            if index % 25 == 0:
                save('latest.pt')
                log({'event': 'training', 'index': index, 'phase': 'imitation' if imitation else 'rl',
                     'loss': loss, 'cleared': result['cleared']})
            if index % args.validate_every == 0 or index == args.teacher_games or index == total:
                scores = evaluate(env, policy, range(args.seed + 1000000,
                                  args.seed + 1000000 + args.validation_games))
                score = scores['mean_cleared_fraction'] * 160
                if score > best:
                    best = score
                    save('best.pt')
                save('latest.pt')
                log({'event': 'validation', 'index': index, 'mean_cleared': score,
                     'best_validation': best})
    finally:
        save('latest.pt')
        log({'event': 'stopped', 'index': index, 'completed': index == total,
             'best_validation': best})


if __name__ == '__main__':
    main()

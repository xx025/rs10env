"""Paired benchmark: python -m rs10env.benchmark --games 100 --seed 1000."""
import argparse
import json
import time

import numpy as np
import torch

from rs10env import RS10Env, create_strategy, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=128)
    args = parser.parse_args()
    if args.games < 2:
        parser.error("--games must be at least 2")
    torch.set_num_threads(1)
    env = RS10Env(device="cpu")
    names = ["center_small_rect", "max_future_moves", "multi_start"]
    scores = {name: [] for name in names}
    times = {name: [] for name in names}
    for seed in range(args.seed, args.seed + args.games):
        record = {"seed": seed}
        for name in names:
            kwargs = {"num_rollouts": args.rollouts} if name == "multi_start" else {}
            strategy = create_strategy(name, seed=68, device="cpu", **kwargs)
            start = time.perf_counter()
            result = run_episode(env, strategy, seed=seed)
            elapsed = time.perf_counter() - start
            scores[name].append(result["total_cleared"])
            times[name].append(elapsed)
            record[name] = {"cleared": result["total_cleared"], "seconds": elapsed}
        print(json.dumps(record), flush=True)
    delta = np.array(scores["multi_start"]) - np.array(scores["max_future_moves"])
    margin = 1.96 * delta.std(ddof=1) / np.sqrt(args.games)
    print(json.dumps({
        "summary": {name: {"avg_cleared": np.mean(scores[name]),
                           "avg_seconds": np.mean(times[name])} for name in names},
        "paired_difference": float(delta.mean()),
        "approx_95pct_ci": [delta.mean() - margin, delta.mean() + margin],
        "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()),
                             int((delta < 0).sum())],
        "games": args.games, "base_seed": args.seed, "rollouts": args.rollouts,
    }), flush=True)


if __name__ == "__main__":
    main()

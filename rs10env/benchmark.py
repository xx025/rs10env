"""Paired benchmark: python -m rs10env.benchmark --games 100 --seed 1000."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rs10env import RS10Env, create_strategy, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, help="Write complete results to a new JSON file")
    args = parser.parse_args()
    if args.games < 2:
        parser.error("--games must be at least 2")
    torch.set_num_threads(1)
    env = RS10Env(device="cpu")
    names = ["max_future_moves", "multi_start", "multi_start_512", "trajectory_search"]
    records = []
    scores = {name: [] for name in names}
    times = {name: [] for name in names}
    for seed in range(args.seed, args.seed + args.games):
        record = {"seed": seed}
        for name in names:
            kwargs = {}
            factory_name = name
            if name in ("multi_start", "trajectory_search"):
                kwargs["num_rollouts"] = args.rollouts
            if name == "multi_start_512":
                factory_name = "multi_start"
                kwargs["num_rollouts"] = 512
            if name == "trajectory_search":
                kwargs.update(iterations=args.iterations, batch_size=args.batch_size)
            strategy = create_strategy(factory_name, seed=68, device="cpu", **kwargs)
            start = time.perf_counter()
            result = run_episode(env, strategy, seed=seed)
            elapsed = time.perf_counter() - start
            scores[name].append(result["total_cleared"])
            times[name].append(elapsed)
            record[name] = {"cleared": result["total_cleared"], "seconds": elapsed}
        print(json.dumps(record), flush=True)
        records.append(record)
    comparisons = {}
    for baseline in names[:-1]:
        delta = np.array(scores["trajectory_search"]) - np.array(scores[baseline])
        margin = 1.96 * delta.std(ddof=1) / np.sqrt(args.games)
        comparisons[baseline] = {
            "paired_difference": float(delta.mean()),
            "approx_95pct_ci": [delta.mean() - margin, delta.mean() + margin],
            "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()),
                                 int((delta < 0).sum())],
        }
    summary = {
        "summary": {name: {"avg_cleared": np.mean(scores[name]),
                           "avg_seconds": np.mean(times[name])} for name in names},
        "trajectory_search_vs": comparisons,
        "games": args.games, "base_seed": args.seed, "rollouts": args.rollouts,
        "iterations": args.iterations, "batch_size": args.batch_size,
        "strategy_seed": 68, "torch_threads": 1,
    }
    print(json.dumps(summary), flush=True)
    if args.output:
        with args.output.open("x") as output:
            json.dump({"episodes": records, **summary}, output, indent=2)
            output.write("\n")


if __name__ == "__main__":
    main()

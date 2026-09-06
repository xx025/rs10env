"""Remote algorithm comparison with explicit warmup and paired boards."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import numba
import torch

from rs10env import RS10Env
from rs10env.fast_search import PopulationSearchStrategy
from rs10env.adaptive_search import AdaptiveSearchStrategy
from rs10env.repair_search import RepairSearchStrategy, HybridRepairStrategy, SpatialRepairStrategy
from rs10env.goal_search import GoalSearchStrategy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--budget", type=float, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--levels", type=int, choices=[1, 2], default=2)
    parser.add_argument("--outer-steps", type=int, default=16)
    parser.add_argument("--rollouts", type=int)
    parser.add_argument("--candidate", choices=["adaptive", "repair", "hybrid", "spatial", "goal"], default="adaptive")
    parser.add_argument("--baseline", choices=["population", "hybrid"], default="population")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.output.exists() or not args.output.parent.is_dir() or args.games < 1
            or args.budget <= 0 or not np.isfinite(args.budget)
            or (args.rollouts is not None and args.rollouts < 1)):
        parser.error("Need positive games and a new output file in an existing directory")
    torch.set_num_threads(1)
    warmups = {}
    candidate = {"repair": RepairSearchStrategy, "adaptive": AdaptiveSearchStrategy,
                 "hybrid": HybridRepairStrategy, "spatial": SpatialRepairStrategy,
                 "goal": GoalSearchStrategy}[args.candidate]
    candidate_name = args.candidate + "_search"
    baseline = HybridRepairStrategy if args.baseline == "hybrid" else PopulationSearchStrategy
    baseline_name = args.baseline + "_search"
    names = (baseline_name, candidate_name)
    if baseline_name == candidate_name:
        parser.error("Candidate and baseline must differ")
    for cls in (baseline, candidate):
        start = time.perf_counter()
        cls.warmup()
        warmups[cls.__name__] = time.perf_counter() - start
    env = RS10Env(device="cpu")
    records = []
    for seed in range(args.seed, args.seed + args.games):
        record = {"seed": seed}
        for name, cls in [(baseline_name, baseline),
                          (candidate_name, candidate)]:
            kwargs = {"time_budget": args.budget, "seed": 68, "device": "cpu"}
            if args.rollouts:
                kwargs["max_rollouts"] = args.rollouts
            if cls is AdaptiveSearchStrategy:
                kwargs["adaptation_steps"] = args.steps
                kwargs["levels"] = args.levels
                kwargs["outer_steps"] = args.outer_steps
            strategy = cls(**kwargs)
            start = time.perf_counter()
            _, info = env.reset(seed=seed)
            board = env.board_2d.tolist()
            actions = []
            while info["action_mask"].any():
                action = strategy.get_action(env, info["action_mask"])
                if not info["action_mask"][action]:
                    raise RuntimeError("Illegal action from search")
                actions.append(int(action))
                _, _, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
            record[name] = {"cleared": int(env.total_zeros),
                            "seconds": time.perf_counter() - start,
                            "rollouts": strategy.rollouts, "actions": actions}
        record["board"] = board
        records.append(record)
        print(json.dumps({"seed": seed, **{name: {k: v for k, v in record[name].items()
                          if k != "actions"} for name in names}}), flush=True)
    summary = {}
    for name in names:
        times = [r[name]["seconds"] for r in records]
        summary[name] = {"avg_cleared": np.mean([r[name]["cleared"] for r in records]),
                         "avg_seconds": np.mean(times), "max_seconds": max(times),
                         "over_10_seconds": sum(t > 10 for t in times)}
    delta = np.array([r[candidate_name]["cleared"] - r[baseline_name]["cleared"]
                      for r in records])
    comparison = {"mean_difference": float(delta.mean()),
                  "wins_ties_losses": [int((delta > 0).sum()), int((delta == 0).sum()),
                                       int((delta < 0).sum())]}
    if len(delta) > 1:
        margin = 1.96 * delta.std(ddof=1) / np.sqrt(len(delta))
        comparison["approx_95pct_ci"] = [delta.mean() - margin, delta.mean() + margin]
    result = {"summary": summary, "comparison": comparison,
              "episodes": records, "warmups": warmups,
              "config": {k: v for k, v in vars(args).items() if k != "output"}}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

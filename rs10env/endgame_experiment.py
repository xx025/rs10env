"""Check exact last-k-move improvements on saved legal trajectories."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rs10env import RS10Env
from rs10env.exact_search import solve_endgame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suffix", type=int, default=8)
    parser.add_argument("--nodes", type=int, default=1000)
    parser.add_argument("--games", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    if min(args.suffix, args.nodes, args.games) < 1:
        parser.error("Budgets must be positive")
    data = json.loads(args.input.read_text())
    torch.set_num_threads(1)
    records = []
    for episode in data["episodes"][:args.games]:
        original = np.array(episode["board"], dtype=np.int32)
        env = RS10Env(H=original.shape[0], W=original.shape[1], device="cpu")
        actions = episode["hybrid_search"]["actions"]
        _, info = env.reset(board=original.copy())
        cut = max(0, len(actions) - args.suffix)
        for a in actions[:cut]:
            assert info["action_mask"][a]
            _, _, _, _, info = env.step(a)
        before = int(env.total_zeros)
        start = time.perf_counter()
        result = solve_endgame(env.board_2d.tolist(), env.all_rects.tolist(),
                               max_nodes=args.nodes, max_steps=env.max_steps - env.step_count)
        seconds = time.perf_counter() - start
        for a in result.actions:
            assert info["action_mask"][a]
            _, _, _, _, info = env.step(a)
        assert int(env.total_zeros) == before + result.cleared
        record = {"seed": episode["seed"], "baseline": episode["hybrid_search"]["cleared"],
                  "exact_result": before + result.cleared, "proven_optimal": result.proven_optimal,
                  "nodes": result.nodes, "seconds": seconds}
        records.append(record)
        print(json.dumps(record), flush=True)
    with args.output.open("x") as output:
        json.dump({"suffix": args.suffix, "node_budget": args.nodes, "episodes": records}, output, indent=2)


if __name__ == "__main__":
    main()

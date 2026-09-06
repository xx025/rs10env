"""Measure reachability of rectangles covering cells stranded by a saved plan."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rs10env import RS10Env
from rs10env.dependency_search import activate_rectangle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--goals", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--suffix", type=int, default=20)
    parser.add_argument("--filter-subsets", action="store_true")
    args = parser.parse_args()
    if min(args.games, args.goals, args.nodes, args.suffix) < 1:
        parser.error("Budgets must be positive")
    if args.output.exists() or not args.output.parent.is_dir():
        parser.error("Output must be a new file in an existing directory")
    data = json.loads(args.input.read_text())
    torch.set_num_threads(1)
    records = []
    for episode in data["episodes"][:args.games]:
        initial = np.array(episode["board"], dtype=np.int32)
        env = RS10Env(H=initial.shape[0], W=initial.shape[1], device="cpu")
        rects = env.all_rects.numpy()
        actions = episode["hybrid_search"]["actions"]
        terminal = initial.copy()
        for a in actions:
            r1, c1, r2, c2 = rects[a]
            terminal[r1:r2 + 1, c1:c2 + 1] = 0
        cut = max(0, len(actions) - args.suffix)
        state = initial.copy()
        for a in actions[:cut]:
            r1, c1, r2, c2 = rects[a]
            state[r1:r2 + 1, c1:c2 + 1] = 0
        candidates = []
        for a, (r1, c1, r2, c2) in enumerate(rects):
            if not np.any(terminal[r1:r2 + 1, c1:c2 + 1]):
                continue
            if not ((state[r1, c1] and state[r2, c2]) or
                    (state[r1, c2] and state[r2, c1])):
                continue
            excess = int(state[r1:r2 + 1, c1:c2 + 1].sum()) - 10
            if excess > 0:
                if args.filter_subsets:
                    feasible = False
                    for corners in ({(r1, c1), (r2, c2)}, {(r1, c2), (r2, c1)}):
                        if not all(state[r, c] for r, c in corners):
                            continue
                        reachable = 1
                        cap = (1 << (excess + 1)) - 1
                        for r in range(r1, r2 + 1):
                            for c in range(c1, c2 + 1):
                                if (r, c) not in corners:
                                    reachable |= reachable << int(state[r, c])
                                    reachable &= cap
                        feasible |= bool(reachable & (1 << excess))
                    if not feasible:
                        continue
                candidates.append((excess, a))
        for excess, a in sorted(candidates)[:args.goals]:
            start = time.perf_counter()
            result = activate_rectangle(state.tolist(), rects.tolist(), a,
                                        max_nodes=args.nodes, max_steps=env.max_steps - cut)
            seconds = time.perf_counter() - start
            record = {"seed": episode["seed"], "cut": cut, "goal": int(a),
                      "excess": excess, "success": result.success,
                      "proven_infeasible": result.proven_infeasible,
                      "nodes": result.nodes, "seconds": seconds,
                      "actions": list(result.actions)}
            if result.success:
                _, info = env.reset(board=state.copy())
                for action in result.actions:
                    assert info["action_mask"][action]
                    _, _, _, _, info = env.step(action)
                record["stranded_cells_rescued"] = int(np.count_nonzero(
                    (terminal > 0) & (env.board_2d.numpy() == 0)))
                record["partial_total_cleared"] = int(np.count_nonzero(initial)
                                                       - np.count_nonzero(env.board_2d.numpy()))
            records.append(record)
            print(json.dumps(record), flush=True)
    with args.output.open("x") as output:
        json.dump({"node_budget": args.nodes, "suffix": args.suffix,
                   "filter_subsets": args.filter_subsets,
                   "goals_per_board": args.goals, "records": records}, output, indent=2)


if __name__ == "__main__":
    main()

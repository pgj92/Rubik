"""
Solve cubes by searching with the learned cost-to-go heuristic J(s).

This is the inference half of DeepCubeA. Training (train/davi.py) gives us
J(s) ≈ moves-to-solve; here we use it to guide Batch Weighted A* (BWAS):

    f(s) = g(s) + lambda * J(s)

where g(s) is the path length so far and lambda weights the heuristic. The
"batch" part: we pop the best N nodes from the open set and evaluate J on all
their children in a single network forward pass — the GPU likes big batches,
so this is far faster than expanding one node at a time.

Why search rescues us
---------------------
A greedy policy (always step to the lowest-J child) is brittle: one bad
heuristic estimate derails it. A* keeps a frontier and can recover, so even an
imperfect J solves cubes a greedy policy can't. This is exactly the gap we saw
in Phase 1 — PPO learned near-optimal *policies* but couldn't discover them at
depth. Search supplies the discovery.

Usage:
    python -m eval.solve_search --checkpoint runs/<run>/value_net.pt
    python -m eval.solve_search --checkpoint runs/<run>/value_net.pt \
        --max-depth 26 --episodes 100 --lam 2.0 --batch-expand 1000 --max-nodes 1000000
"""
import argparse
import heapq
import itertools

import numpy as np
import torch

from cube.cube import N_MOVES, N_STICKERS, get_perms, solved_state, scramble
from train.davi import ValueNet

PERMS = get_perms()
SOLVED = solved_state()
SOLVED_BYTES = SOLVED.tobytes()


def load_value_net(checkpoint_path: str, device: torch.device) -> ValueNet:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net = ValueNet(ckpt.get("hidden_size", 1024), ckpt.get("n_layers", 4)).to(device)
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net


@torch.no_grad()
def heuristic(states: np.ndarray, net: ValueNet, device) -> np.ndarray:
    """J(s) for a (B, 54) batch of states -> (B,) numpy array, clamped >= 0."""
    t = torch.as_tensor(states, device=device)
    j = net(t).clamp(min=0.0)
    return j.cpu().numpy()


@torch.no_grad()
def bwas_solve(start: np.ndarray, net: ValueNet, device, lam: float,
               batch_expand: int, max_nodes: int):
    """
    Batch Weighted A*. Returns (solution_moves or None, nodes_expanded).

    f = g + lam * J. We expand the `batch_expand` lowest-f open nodes at once,
    score all their children with one net call, and push improvements.
    """
    start_bytes = start.tobytes()
    if start_bytes == SOLVED_BYTES:
        return [], 0

    best_g = {start_bytes: 0}
    came_from: dict[bytes, tuple[bytes, int]] = {}        # child -> (parent, move)
    counter = itertools.count()                           # tie-breaker for heap

    h0 = float(heuristic(start[None, :], net, device)[0])
    open_heap = [(lam * h0, next(counter), 0, start_bytes, start)]

    nodes_expanded = 0
    while open_heap and nodes_expanded < max_nodes:
        # Pop a batch of the lowest-f nodes.
        batch = []
        while open_heap and len(batch) < batch_expand:
            f, _, g, sb, s = heapq.heappop(open_heap)
            if g > best_g.get(sb, g):                     # stale heap entry
                continue
            batch.append((g, sb, s))
        if not batch:
            break

        parents = np.stack([s for (_, _, s) in batch], axis=0)        # (b, 54)
        children = parents[:, PERMS]                                  # (b, 12, 54)
        nodes_expanded += len(batch)

        # Goal check before spending a net call.
        for bi, (g, sb, s) in enumerate(batch):
            for m in range(N_MOVES):
                if children[bi, m].tobytes() == SOLVED_BYTES:
                    return _reconstruct(came_from, sb, m), nodes_expanded

        flat = children.reshape(-1, N_STICKERS)
        h = heuristic(flat, net, device).reshape(len(batch), N_MOVES)

        for bi, (g, sb, s) in enumerate(batch):
            ng = g + 1
            for m in range(N_MOVES):
                cb = children[bi, m].tobytes()
                if ng < best_g.get(cb, np.inf):
                    best_g[cb] = ng
                    came_from[cb] = (sb, m)
                    f = ng + lam * float(h[bi, m])
                    heapq.heappush(open_heap, (f, next(counter), ng, cb, children[bi, m]))

    return None, nodes_expanded


def _reconstruct(came_from: dict, last_parent_bytes: bytes, last_move: int) -> list[int]:
    """Walk parent pointers back to start; return the move list start->goal."""
    moves = [last_move]
    cur = last_parent_bytes
    while cur in came_from:
        parent, m = came_from[cur]
        moves.append(m)
        cur = parent
    return moves[::-1]


def evaluate_depth(net, depth, n_episodes, lam, batch_expand, max_nodes, device, seed):
    rng = np.random.default_rng(seed)
    n_solved = 0
    sol_lengths, nodes_list = [], []
    for _ in range(n_episodes):
        state, _ = scramble(depth, rng)
        sol, nodes = bwas_solve(state, net, device, lam, batch_expand, max_nodes)
        nodes_list.append(nodes)
        if sol is not None:
            n_solved += 1
            sol_lengths.append(len(sol))
    return {
        "success_rate": n_solved / n_episodes,
        "mean_len": float(np.mean(sol_lengths)) if sol_lengths else None,
        "mean_nodes": float(np.mean(nodes_list)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--lam", type=float, default=2.0,
                        help="heuristic weight; higher = greedier/faster, less optimal")
    parser.add_argument("--batch-expand", type=int, default=1000,
                        help="open nodes expanded per batch (bigger = better GPU use)")
    parser.add_argument("--max-nodes", type=int, default=1_000_000,
                        help="give up after expanding this many nodes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = load_value_net(args.checkpoint, device)

    print(f"checkpoint: {args.checkpoint}  (BWAS lambda={args.lam}, "
          f"batch={args.batch_expand}, max_nodes={args.max_nodes}, "
          f"{args.episodes} episodes/depth)\n")
    print(f"{'depth':>6} {'solve%':>8} {'mean_len':>10} {'mean_nodes':>12}")
    print("-" * 40)
    for depth in range(1, args.max_depth + 1):
        r = evaluate_depth(net, depth, args.episodes, args.lam, args.batch_expand,
                           args.max_nodes, device, seed=args.seed + 1000 * depth)
        len_str = f"{r['mean_len']:.1f}" if r["mean_len"] is not None else "n/a"
        print(f"{depth:>6} {r['success_rate'] * 100:>7.1f}% {len_str:>10} {r['mean_nodes']:>12.0f}")


if __name__ == "__main__":
    main()

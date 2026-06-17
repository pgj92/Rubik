"""
DAVI: Deep Approximate Value Iteration for the Rubik's cube.

This is Phase 2 — the approach that actually scales to full scrambles, where
model-free PPO plateaus around depth ~7 (see results/). It's the training half
of the DeepCubeA recipe (Agostinelli et al., 2019):

    Learn a cost-to-go function J(s) ≈ "minimum number of moves to solve s",
    then (in eval/solve_search.py) use J as a heuristic to guide search.

Why this sidesteps PPO's exploration wall
------------------------------------------
PPO needs to *stumble* onto the solved state to get any signal, and the chance
of that decays like 12^-depth. DAVI never explores: it *generates* its own
training states by scrambling from solved, and learns J by value iteration —
bootstrapping each state's value from its neighbours:

    J(s) = 0                                  if s is solved
    J(s) = min over moves a of [ 1 + J(s') ]   otherwise,  s' = a applied to s

The "1" is the cost of one move. We regress a network toward the right-hand
side, using a slowly-updated *target* copy of the network for the J(s') term
(standard trick to keep the bootstrap target stable). Every state — solvable in
1 move or 26 — produces a usable target every step, so there is no all-or-
nothing episode lottery. That is the whole point.

Usage:
    python -m train.davi --total-updates 20000 --batch-size 10000
"""
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cube.cube import (
    N_MOVES,
    N_STICKERS,
    N_COLORS,
    get_perms,
    solved_state,
)


@dataclass
class Args:
    exp_name: str = "davi"
    seed: int = 1
    cuda: bool = True

    # --- DAVI core ---
    total_updates: int = 20_000
    """number of gradient updates"""
    batch_size: int = 10_000
    """fresh scrambled states generated per update"""
    max_scramble_depth: int = 30
    """states are random walks of length ~Uniform[1, this] from solved.
    God's number in QTM is 26, so 30 comfortably covers all of state space."""
    target_update_interval: int = 50
    """copy online weights into the target network every N updates"""

    # --- optimization ---
    learning_rate: float = 1e-3
    hidden_size: int = 1024
    n_layers: int = 4
    """number of hidden Linear+ReLU layers"""

    save_model: bool = True


def _onehot(x: torch.Tensor) -> torch.Tensor:
    """(batch, 54) int color indices -> (batch, 324) float one-hot."""
    return F.one_hot(x.long(), num_classes=N_COLORS).reshape(x.shape[0], -1).float()


OBS_DIM = N_STICKERS * N_COLORS  # 324


class ValueNet(nn.Module):
    """
    Cost-to-go estimator J(s). Maps a one-hot cube state to a single scalar
    (estimated moves-to-solve). A plain ReLU MLP — bigger than the PPO net,
    because here the network has to memorize a global distance field over the
    whole state space, not just a local policy.
    """

    def __init__(self, hidden_size: int = 1024, n_layers: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(OBS_DIM, hidden_size), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers += [nn.Linear(hidden_size, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x_int: torch.Tensor) -> torch.Tensor:
        return self.net(_onehot(x_int)).squeeze(-1)


# ----------------------------------------------------------------------------
# Vectorized cube ops (numpy). We avoid cube.apply_move's per-state python loop
# so we can scramble and expand tens of thousands of states per update.
# ----------------------------------------------------------------------------
PERMS = get_perms()           # (12, 54) int — PERMS[m] is the permutation for move m
SOLVED = solved_state()       # (54,) int


def random_scramble_batch(batch_size: int, max_depth: int, rng: np.random.Generator,
                          return_depths: bool = False):
    """
    Generate `batch_size` scrambled states, each a random walk of length
    Uniform[1, max_depth] from solved. Returns (batch_size, 54) int array, or
    (states, depths) if return_depths=True (depths is the walk length per state).

    Vectorized: all states walk together; at step t we only move the states
    whose sampled length is still >= t. (Random walk, not the de-duplicated
    scramble() used for episodes — redundant moves just make the true distance
    shorter than the walk length, which DAVI handles fine.)
    """
    depths = rng.integers(1, max_depth + 1, size=batch_size)
    states = np.tile(SOLVED, (batch_size, 1))          # (B, 54)
    for t in range(1, max_depth + 1):
        active = depths >= t
        n = int(active.sum())
        if n == 0:
            break
        moves = rng.integers(0, N_MOVES, size=n)        # (n,)
        idx = PERMS[moves]                              # (n, 54)
        states[active] = np.take_along_axis(states[active], idx, axis=1)
    if return_depths:
        return states, depths
    return states


def expand_children(states: np.ndarray) -> np.ndarray:
    """
    Apply all 12 moves to each state.
    states: (B, 54) -> children: (B, 12, 54).
    """
    return states[:, PERMS]   # fancy index: new[b, m, i] = states[b, PERMS[m, i]]


def compute_targets(states_np: np.ndarray, target_net: ValueNet, device) -> torch.Tensor:
    """
    One step of value iteration: y(s) = min_a [ 1 + J_target(child_a) ],
    with the child term forced to 0 when the child is the solved state.
    Returns a (B,) float tensor (detached).
    """
    B = states_np.shape[0]
    children = expand_children(states_np)                       # (B, 12, 54)
    is_goal = (children == SOLVED).all(axis=2)                  # (B, 12)

    flat = torch.as_tensor(children.reshape(-1, N_STICKERS), device=device)
    with torch.no_grad():
        child_J = target_net(flat).reshape(B, N_MOVES)         # (B, 12)
    child_J = torch.clamp(child_J, min=0.0)                     # cost-to-go is non-negative

    cost = 1.0 + child_J
    cost[torch.as_tensor(is_goal, device=device)] = 1.0        # J(goal)=0, so moving to goal costs 1
    target = cost.min(dim=1).values                            # (B,)

    # Anchor: a state that is already solved has cost-to-go 0.
    input_is_goal = torch.as_tensor((states_np == SOLVED).all(axis=1), device=device)
    target[input_is_goal] = 0.0
    return target


def main():
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__s{args.seed}__{int(time.time())}"
    os.makedirs(f"runs/{run_name}", exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    online = ValueNet(args.hidden_size, args.n_layers).to(device)
    target = ValueNet(args.hidden_size, args.n_layers).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = optim.Adam(online.parameters(), lr=args.learning_rate)

    def save_checkpoint():
        torch.save(
            {
                "model_state_dict": online.state_dict(),
                "hidden_size": args.hidden_size,
                "n_layers": args.n_layers,
                "args": vars(args),
            },
            f"runs/{run_name}/value_net.pt",
        )

    start = time.time()
    for update in range(1, args.total_updates + 1):
        states_np, depths_np = random_scramble_batch(
            args.batch_size, args.max_scramble_depth, rng, return_depths=True
        )
        states_t = torch.as_tensor(states_np, device=device)

        target_vals = compute_targets(states_np, target, device)
        pred = online(states_t)
        loss = F.mse_loss(pred, target_vals)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Periodically refresh the target network from the online network.
        # (DeepCubeA refreshes when loss drops below a threshold; periodic is
        # simpler and works well here.)
        if update % args.target_update_interval == 0:
            target.load_state_dict(online.state_dict())

        # --- TensorBoard logging (mirrors train/ppo_cube.py) ---
        writer.add_scalar("losses/value_loss", loss.item(), update)
        writer.add_scalar("charts/mean_pred", pred.mean().item(), update)
        writer.add_scalar("charts/mean_target", target_vals.mean().item(), update)

        if update % 100 == 0 or update == 1:
            ups = update / (time.time() - start)
            writer.add_scalar("charts/updates_per_sec", ups, update)
            # DAVI-specific diagnostic: mean predicted cost-to-go bucketed by the
            # scramble walk length. A healthy heuristic grows monotonically with
            # depth — this is the curve to watch, more telling than loss alone.
            pred_np = pred.detach().cpu().numpy()
            for d in range(1, args.max_scramble_depth + 1):
                mask = depths_np == d
                if mask.any():
                    writer.add_scalar(f"pred_by_depth/d{d:02d}", float(pred_np[mask].mean()), update)
            print(f"update {update:6d}/{args.total_updates}  "
                  f"loss={loss.item():.4f}  "
                  f"mean_pred={pred.mean().item():.2f}  "
                  f"updates/s={ups:.1f}")
            if args.save_model and update % 1000 == 0:
                save_checkpoint()

    if args.save_model:
        save_checkpoint()
        print(f"saved value net to runs/{run_name}/value_net.pt")

    writer.close()


if __name__ == "__main__":
    main()

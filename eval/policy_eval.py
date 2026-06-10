"""
Evaluate a trained PPO agent's solve rate as a function of scramble depth.

This is the frontier curve: for each depth, run N fresh scrambles with the
greedy policy (argmax over action logits) and report success rate and
steps-to-solve. Compare against eval/random_baseline.py for the floor.

Usage:
    python -m eval.policy_eval --checkpoint runs/<run_name>/agent.pt
    python -m eval.policy_eval --checkpoint runs/<run_name>/agent.pt \
        --max-depth 14 --episodes 200 --max-steps 40 --stochastic
"""
import argparse

import numpy as np
import torch

from cube.gym_env import RubikEnv
from train.ppo_cube import Agent, _onehot
from cube.cube import N_MOVES


def load_agent(checkpoint_path: str, device: torch.device) -> Agent:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    agent = Agent(N_MOVES, hidden_size=ckpt.get("hidden_size", 128)).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


@torch.no_grad()
def evaluate_depth(
    agent: Agent,
    depth: int,
    n_episodes: int,
    max_steps: int,
    device: torch.device,
    seed: int,
    stochastic: bool = False,
):
    """Run `n_episodes` at a fixed scramble depth; return (success_rate, solved_step_counts)."""
    env = RubikEnv(scramble_depth=depth, max_steps=max_steps)
    n_solved = 0
    solved_step_counts = []
    for episode in range(n_episodes):
        # Seed once to get a reproducible stream; later resets (seed=None)
        # draw fresh scrambles from the same RNG.
        obs, info = env.reset(seed=seed if episode == 0 else None)
        while True:
            x = torch.as_tensor(obs, dtype=torch.long, device=device).unsqueeze(0)
            if stochastic:
                action, _, _, _ = agent.get_action_and_value(x)
                action = int(action.item())
            else:
                logits = agent.actor(_onehot(x))
                action = int(logits.argmax(dim=-1).item())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if terminated:
                    n_solved += 1
                    solved_step_counts.append(info["step_count"])
                break
    return n_solved / n_episodes, solved_step_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to agent.pt")
    parser.add_argument("--max-depth", type=int, default=12, help="evaluate depths 1..max-depth")
    parser.add_argument("--episodes", type=int, default=200, help="episodes per depth")
    parser.add_argument("--max-steps", type=int, default=30, help="step budget per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true",
                        help="sample from the policy instead of taking argmax")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = load_agent(args.checkpoint, device)

    mode = "stochastic" if args.stochastic else "greedy"
    print(f"checkpoint: {args.checkpoint}  ({mode} policy, "
          f"{args.episodes} episodes/depth, max {args.max_steps} steps)\n")
    print(f"{'depth':>6} {'success%':>10} {'mean_steps':>12} {'median_steps':>14}")
    print("-" * 46)

    for depth in range(1, args.max_depth + 1):
        success_rate, steps = evaluate_depth(
            agent, depth, args.episodes, args.max_steps, device,
            seed=args.seed + 1000 * depth, stochastic=args.stochastic,
        )
        mean_str = f"{np.mean(steps):.1f}" if steps else "n/a"
        med_str = f"{np.median(steps):.1f}" if steps else "n/a"
        print(f"{depth:>6} {success_rate * 100:>9.1f}% {mean_str:>12} {med_str:>14}")


if __name__ == "__main__":
    main()

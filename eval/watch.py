"""
Watch a policy run on the cube.

Usage:
    # Open an interactive window (your local machine):
    python -m eval.watch --depth 3 --max-steps 30

    # Save an animated GIF (useful in headless environments):
    python -m eval.watch --depth 3 --max-steps 30 --save out.gif

    # ANSI in your terminal (no GUI needed):
    python -m eval.watch --depth 3 --max-steps 30 --ansi
"""
import argparse
import numpy as np

from cube.gym_env import RubikEnv
from cube.cube import MOVES
from cube.render import animate, render_ansi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=3, help="scramble depth")
    parser.add_argument("--max-steps", type=int, default=30, help="step budget for the policy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default=None, help="GIF output path; if omitted, opens a window")
    parser.add_argument("--ansi", action="store_true", help="print every step to terminal instead of plotting")
    parser.add_argument("--interval-ms", type=int, default=600)
    args = parser.parse_args()

    env = RubikEnv(scramble_depth=args.depth, max_steps=args.max_steps)
    rng = np.random.default_rng(args.seed)

    obs, info = env.reset(seed=args.seed)
    states = [obs.copy()]
    titles = [f"Step 0 — scrambled (depth {args.depth})"]

    if args.ansi:
        print(titles[0])
        print(render_ansi(obs))

    last_status = "ran out of steps"
    for t in range(1, args.max_steps + 1):
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        titles.append(f"Step {t} — move {MOVES[action]}")

        if args.ansi:
            print(titles[-1])
            print(render_ansi(obs))

        if terminated:
            last_status = "SOLVED"
            titles[-1] += "  ✓ SOLVED"
            break
        if truncated:
            last_status = "truncated"
            break

    print(f"\nFinal: {last_status} after {len(states) - 1} steps.")

    if args.ansi:
        return

    animate(states, titles=titles, interval_ms=args.interval_ms, save_path=args.save)
    if args.save:
        print(f"Saved animation to {args.save}")


if __name__ == "__main__":
    main()

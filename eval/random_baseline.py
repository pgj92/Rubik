"""
Random-policy baseline for the cube env.

Runs N episodes per scramble depth, uses a uniformly random action policy,
and reports success rate + average steps-to-solve.

This is the "floor" any trained policy must beat. It also builds intuition
for why sparse-reward RL on the cube is hard.

Usage:
    python -m eval.random_baseline
"""

import numpy as np
from cube.gym_env import RubikEnv


def run_random_policy(env: RubikEnv, n_episodes: int, rng: np.random.Generator):
    """
    Run `n_episodes` of the random policy on `env`.

    Returns a dict with:
      - n_episodes:   total episodes run
      - n_solved:     count of episodes that ended in `terminated=True`
      - success_rate: n_solved / n_episodes
      - mean_steps_when_solved:  average step count for the solved episodes
      - median_steps_when_solved: median, useful when distribution is skewed
    """
    # TODO 1: For each episode:
    #   - call env.reset() (no need to override scramble_depth; the env was
    #     constructed with the depth we want)
    #   - loop until terminated or truncated:
    #       - pick an action UNIFORMLY at random using `rng.integers(0, env.action_space.n)`
    #         (do NOT use env.action_space.sample() — it uses its own RNG and is
    #          harder to seed reproducibly)
    #       - call env.step(action)
    #   - if terminated: record the step count
    #
    # Then compute and return the stats dict.
    raise NotImplementedError("TODO 1: implement the random-policy rollout loop")


def main():
    depths = [1, 3, 5, 7]
    n_episodes = 1000
    max_steps = 50

    print(f"{'depth':>6} {'success%':>10} {'mean_steps':>12} {'median_steps':>14}")
    print("-" * 46)

    for depth in depths:
        env = RubikEnv(scramble_depth=depth, max_steps=max_steps)
        rng = np.random.default_rng(seed=42 + depth)
        stats = run_random_policy(env, n_episodes, rng)

        pct = stats["success_rate"] * 100
        mean_s = stats["mean_steps_when_solved"]
        med_s = stats["median_steps_when_solved"]
        mean_str = f"{mean_s:.1f}" if mean_s is not None else "n/a"
        med_str  = f"{med_s:.1f}"  if med_s  is not None else "n/a"
        print(f"{depth:>6} {pct:>9.2f}% {mean_str:>12} {med_str:>14}")


if __name__ == "__main__":
    main()

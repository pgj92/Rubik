"""
Reverse-curriculum scheduler for the Rubik's cube.

The Curriculum class is the brain:
    - Tracks per-depth success history (sliding window).
    - On reset, samples a depth: usually the current depth, sometimes a lower
      one (to keep skills fresh).
    - Promotes to depth+1 once the success rate at the current depth is high.

The CurriculumWrapper plugs it into a gym env: on reset, the wrapper consults
the curriculum and tells the underlying RubikEnv what scramble depth to use.

In PPO training, the same Curriculum instance is shared across all parallel
envs (we run SyncVectorEnv, so no concurrency to worry about).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np


class Curriculum:
    """
    Reverse-curriculum scheduler.

    State:
      - depth:        current target scramble depth
      - history[d]:   deque of recent solved/unsolved (1/0) outcomes at depth d

    Policy decisions:
      - sample_depth() is called every env reset. It returns the depth to use
        for THIS episode. With probability mix_current we use self.depth;
        otherwise we sample uniformly from {1, ..., self.depth - 1}.
        (At depth 1 there's nothing lower, so we always return 1.)
      - record(depth_used, solved) is called every time an episode ends.
      - maybe_promote() is called once per training iteration. If the
        sliding-window success rate at the CURRENT depth is >= threshold,
        it bumps self.depth by 1 and returns True.
    """

    def __init__(
        self,
        initial_depth: int = 1,
        max_depth: int = 7,
        window: int = 200,
        promote_threshold: float = 0.80,
        mix_current: float = 0.80,
        rng: np.random.Generator | None = None,
    ):
        self.depth = initial_depth
        self.max_depth = max_depth
        self.window = window
        self.promote_threshold = promote_threshold
        self.mix_current = mix_current
        self.rng = rng or np.random.default_rng()
        # Per-depth sliding window of success/fail outcomes.
        self.history: dict[int, deque[int]] = {}

    # ------------------------------------------------------------------
    # TODO: implement the four methods below.
    # ------------------------------------------------------------------

    def sample_depth(self) -> int:
        """
        Return a scramble depth for the next episode.

        Rules:
          - If self.depth == 1: always return 1 (no lower depth to mix in).
          - Otherwise:
              * with probability self.mix_current, return self.depth
              * with probability 1 - self.mix_current, return an integer
                sampled uniformly from {1, 2, ..., self.depth - 1}

        Use self.rng for randomness:
            self.rng.random()                 -> float in [0, 1)
            self.rng.integers(low, high)      -> int in [low, high)   (high exclusive)
        """
        raise NotImplementedError("TODO: implement sample_depth")

    def record(self, depth_used: int, solved: bool) -> None:
        """
        Record one episode outcome at the depth it was played.

        Append 1 if solved else 0 to self.history[depth_used] (creating the
        deque if this depth is new). The deque has maxlen=self.window so it
        automatically forgets old outcomes.
        """
        raise NotImplementedError("TODO: implement record")

    def current_success_rate(self) -> float | None:
        """
        Return the sliding-window success rate at the CURRENT depth, or None
        if we don't have enough data yet (require at least self.window
        samples — don't trust a noisy half-full window).
        """
        raise NotImplementedError("TODO: implement current_success_rate")

    def maybe_promote(self) -> bool:
        """
        If we're not already at max_depth, and the sliding-window success rate
        at the current depth is >= self.promote_threshold, increment self.depth
        and return True. Otherwise return False.

        IMPORTANT after promoting: do NOT clear the history at the new depth.
        The next-depth history starts empty until episodes are played there.
        """
        raise NotImplementedError("TODO: implement maybe_promote")


class CurriculumWrapper(gym.Wrapper):
    """
    Env wrapper that consults a shared Curriculum on every reset to decide
    this episode's scramble depth. Passes it through to RubikEnv via
    options={"scramble_depth": k}.
    """

    def __init__(self, env, curriculum: Curriculum):
        super().__init__(env)
        self.curriculum = curriculum

    def reset(self, *, seed=None, options=None):
        depth = self.curriculum.sample_depth()
        options = dict(options or {})
        options["scramble_depth"] = depth
        return self.env.reset(seed=seed, options=options)

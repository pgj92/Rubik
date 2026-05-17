"""
Gymnasium environment wrapper for the Rubik's Cube.

Docs to skim before implementing:
  - Custom env tutorial: https://gymnasium.farama.org/introduction/create_custom_env/
  - Env API:             https://gymnasium.farama.org/api/env/
  - Spaces:              https://gymnasium.farama.org/api/spaces/

This file is a scaffold. Fill in the TODOs.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .cube import (
    N_MOVES, N_STICKERS, N_COLORS, MOVES,
    solved_state, apply_move, is_solved, scramble,
)


class RubikEnv(gym.Env):
    """
    Observation: the cube state as a length-54 int array with values in 0..5.
    Action:      an integer in 0..11 (index into cube.MOVES).
    Reward:      +1 when the cube is solved, 0 otherwise.
    Episode ends:
      - terminated=True when the cube is solved
      - truncated=True when step count reaches max_steps without solving

    Scramble depth (how many random moves are applied from solved at reset)
    is configurable per-instance (constructor) and per-episode (reset options).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, scramble_depth: int = 1, max_steps: int = 50):
        super().__init__()

        # TODO 1: Declare action_space.
        #   Hint: spaces.Discrete(N_MOVES)
        # self.action_space = ...

        # TODO 2: Declare observation_space.
        #   We want a length-54 int array, values 0..5.
        #   Hint: spaces.Box(low=0, high=N_COLORS - 1, shape=(N_STICKERS,), dtype=np.int8)
        # self.observation_space = ...

        # TODO 3: Store config.
        # self.scramble_depth = ...
        # self.max_steps = ...

        # Will be initialized in reset().
        self._state: np.ndarray | None = None
        self._step_count: int = 0
        self._current_depth: int = scramble_depth

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode.

        `options` may include {"scramble_depth": int} to override the default
        scramble depth for this specific episode. This is the hook the reverse
        curriculum scheduler will use.

        Returns: (observation, info)
        """
        # TODO 4: Seed self.np_random by calling super().reset(seed=seed).
        # super().reset(seed=seed)

        # TODO 5: Determine this episode's scramble depth.
        #   - Default to self.scramble_depth
        #   - Override if options is not None and "scramble_depth" in options
        # depth = ...
        # self._current_depth = depth

        # TODO 6: Generate the scrambled state.
        #   Use cube.scramble(depth, self.np_random) — it returns (state, moves).
        #   You only need the state.
        # self._state, _ = ...

        # TODO 7: Reset the step counter.
        # self._step_count = 0

        # TODO 8: Return (observation, info).
        #   - observation: self._state (already a length-54 int8 numpy array)
        #   - info: dict containing useful debug data e.g. {"scramble_depth": depth}
        # return ..., {...}

        raise NotImplementedError("TODO: implement reset")

    def step(self, action: int):
        """
        Apply a move and return the result.

        Returns: (observation, reward, terminated, truncated, info)
        """
        # TODO 9: Apply the move to self._state using cube.apply_move.
        # self._state = ...

        # TODO 10: Increment step count.
        # self._step_count += 1

        # TODO 11: Compute terminated and truncated.
        #   - terminated = cube is solved
        #   - truncated = NOT terminated AND step count >= max_steps
        # terminated = ...
        # truncated  = ...

        # TODO 12: Compute reward.
        #   +1.0 if solved this step, else 0.0.
        # reward = ...

        # TODO 13: Return (obs, reward, terminated, truncated, info).
        # return self._state, reward, terminated, truncated, {}

        raise NotImplementedError("TODO: implement step")

    def render(self):
        """Optional: pretty-print the cube state. Skip this for now."""
        return None

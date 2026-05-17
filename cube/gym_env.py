"""
Gymnasium environment wrapper for the Rubik's Cube.

Docs to skim before implementing:
  - Custom env tutorial: https://gymnasium.farama.org/introduction/create_custom_env/
  - Env API:             https://gymnasium.farama.org/api/env/
  - Spaces:              https://gymnasium.farama.org/api/spaces/
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .cube import (
    N_MOVES,
    N_STICKERS,
    N_COLORS,
    MOVES,
    apply_move,
    is_solved,
    scramble,
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

    metadata = {"render_modes": ["ansi", "rgb_array", "human"], "render_fps": 4}

    def __init__(
        self,
        scramble_depth: int = 1,
        max_steps: int = 50,
        render_mode: str | None = None,
    ):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"render_mode must be one of {self.metadata['render_modes']} or None"
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(N_MOVES)

        self.observation_space = spaces.Box(
            low=0, high=N_COLORS - 1, shape=(N_STICKERS,), dtype=np.int8
        )

        self.scramble_depth = scramble_depth
        self.max_steps = max_steps

        # Will be initialized in reset().
        self._state: np.ndarray | None = None
        self._step_count: int = 0
        self._current_depth: int = scramble_depth

    def _get_info(self):
        return {"scramble_depth": self._current_depth, "step_count": self._step_count}

    def reset(self, seed: int | None = None, options: dict | None = None):
        """
        Start a new episode.

        `options` may include {"scramble_depth": int} to override the default
        scramble depth for this specific episode. This is the hook the reverse
        curriculum scheduler will use.

        Returns: (observation, info)
        """
        super().reset(seed=seed)

        depth = (
            options.get("scramble_depth", self.scramble_depth)
            if options
            else self.scramble_depth
        )
        self._current_depth = depth

        self._state, _ = scramble(depth, self.np_random)

        self._step_count = 0

        return self._state, self._get_info()

    def step(self, action: int):
        """
        Apply a move and return the result.

        Returns: (observation, reward, terminated, truncated, info)
        """
        self._state = apply_move(self._state, action)

        self._step_count += 1

        terminated = is_solved(self._state)
        truncated = self._step_count >= self.max_steps and not terminated

        # Reward: +1.0 if solved this step, else 0.0.
        reward = 1.0 if terminated else 0.0

        return self._state, reward, terminated, truncated, self._get_info()

    def render(self):
        """
        Render the current cube state according to self.render_mode.
          - "ansi":      return colored text string
          - "rgb_array": return (H, W, 3) uint8 image
          - "human":     print the ANSI rendering to stdout, return None
          - None:        return None
        """
        if self.render_mode is None:
            return None
        # Import here to avoid pulling matplotlib at env construction time.
        from .render import render_ansi, render_rgb_array
        if self.render_mode == "ansi":
            return render_ansi(self._state)
        if self.render_mode == "rgb_array":
            return render_rgb_array(self._state)
        if self.render_mode == "human":
            print(render_ansi(self._state))
            return None

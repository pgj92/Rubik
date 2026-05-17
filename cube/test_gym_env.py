"""
Sanity tests for the Gymnasium wrapper.
Run with: python -m cube.test_gym_env
"""
import numpy as np
from cube.gym_env import RubikEnv
from cube.cube import is_solved, MOVE_TO_IDX, MOVES


def test_reset_returns_obs_and_info():
    env = RubikEnv(scramble_depth=3)
    obs, info = env.reset(seed=0)
    assert obs.shape == (54,), f"obs shape {obs.shape}"
    assert obs.dtype == np.int8, f"obs dtype {obs.dtype}"
    assert isinstance(info, dict)


def test_reset_depth_0_gives_solved_state():
    """If we scramble with 0 moves, the cube starts solved."""
    env = RubikEnv(scramble_depth=0)
    obs, _ = env.reset(seed=0)
    assert is_solved(obs)


def test_reset_depth_options_override():
    """options={'scramble_depth': k} should override the default."""
    env = RubikEnv(scramble_depth=10)
    obs0, info0 = env.reset(seed=0, options={"scramble_depth": 0})
    assert is_solved(obs0)
    assert info0.get("scramble_depth") == 0


def test_action_space_size():
    env = RubikEnv()
    assert env.action_space.n == 12


def test_step_returns_5_tuple():
    env = RubikEnv(scramble_depth=1)
    env.reset(seed=0)
    out = env.step(0)
    assert len(out) == 5, f"step should return 5 things, got {len(out)}"


def test_solving_terminates_with_reward_1():
    """Scramble by one known move, undo it, expect reward 1 + terminated."""
    env = RubikEnv(scramble_depth=0, max_steps=10)
    env.reset(seed=0)
    # Apply a known move so the cube becomes non-solved
    obs, reward, terminated, truncated, _ = env.step(MOVE_TO_IDX["U"])
    assert not terminated, "after one U from solved, cube is not solved"
    assert reward == 0.0
    # Now apply U' to solve it
    obs, reward, terminated, truncated, _ = env.step(MOVE_TO_IDX["U'"])
    assert terminated, "U then U' should leave cube solved"
    assert reward == 1.0


def test_truncation_after_max_steps():
    env = RubikEnv(scramble_depth=5, max_steps=3)
    env.reset(seed=0)
    for i in range(3):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        if terminated:
            return  # got lucky; not a failure
    assert truncated, "should be truncated after max_steps with no solve"


ALL_TESTS = [
    test_reset_returns_obs_and_info,
    test_reset_depth_0_gives_solved_state,
    test_reset_depth_options_override,
    test_action_space_size,
    test_step_returns_5_tuple,
    test_solving_terminates_with_reward_1,
    test_truncation_after_max_steps,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except NotImplementedError as e:
            print(f"  SKIP  {t.__name__}  (TODO not done: {e})")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

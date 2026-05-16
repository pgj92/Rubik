"""
Sanity tests for the cube env. Run with: python -m cube.test_cube
These will FAIL until Generative Task #2 (the other 5 face perms) is done.

The U-face tests will pass right now though — useful for checking your
understanding before tackling the rest.
"""
import numpy as np
from cube.cube import (
    MOVES, MOVE_TO_IDX, N_MOVES,
    solved_state, apply_move, apply_sequence, is_solved, scramble,
    get_perms,
)


def test_solved_starts_solved():
    assert is_solved(solved_state())


def test_U_four_times_is_identity():
    """Applying U four times returns to solved."""
    s = solved_state()
    for _ in range(4):
        s = apply_move(s, MOVE_TO_IDX["U"])
    assert is_solved(s), "U^4 should be identity"


def test_U_prime_inverts_U():
    """U then U' returns to solved."""
    s = solved_state()
    s = apply_move(s, MOVE_TO_IDX["U"])
    s = apply_move(s, MOVE_TO_IDX["U'"])
    assert is_solved(s), "U U' should be identity"


def test_all_faces_four_turns():
    """For every face X, X^4 == identity."""
    for face in ["U", "L", "F", "R", "B", "D"]:
        s = solved_state()
        for _ in range(4):
            s = apply_move(s, MOVE_TO_IDX[face])
        assert is_solved(s), f"{face}^4 should be identity"


def test_all_inverses():
    """For every move M, M then M^{-1} == identity."""
    for face in ["U", "L", "F", "R", "B", "D"]:
        s = solved_state()
        s = apply_move(s, MOVE_TO_IDX[face])
        s = apply_move(s, MOVE_TO_IDX[face + "'"])
        assert is_solved(s), f"{face} {face}' should be identity"


def test_scramble_then_reverse_solves():
    """Scrambling and then applying the reverse inverse sequence solves it."""
    rng = np.random.default_rng(0)
    for depth in [1, 5, 20]:
        state, moves = scramble(depth, rng)
        # Invert: reverse order, and flip each move (U <-> U')
        def invert(m):
            name = MOVES[m]
            if name.endswith("'"):
                return MOVE_TO_IDX[name[:-1]]
            return MOVE_TO_IDX[name + "'"]
        inv = [invert(m) for m in reversed(moves)]
        state = apply_sequence(state, inv)
        assert is_solved(state), f"reversing depth-{depth} scramble should solve"


def test_perms_are_permutations():
    """Each move's perm array should be a valid permutation of 0..53."""
    perms = get_perms()
    for i in range(N_MOVES):
        assert sorted(perms[i].tolist()) == list(range(54)), f"move {MOVES[i]} perm is not a permutation"


ALL_TESTS = [
    test_solved_starts_solved,
    test_U_four_times_is_identity,
    test_U_prime_inverts_U,
    test_all_faces_four_turns,
    test_all_inverses,
    test_scramble_then_reverse_solves,
    test_perms_are_permutations,
]

if __name__ == "__main__":
    passed, failed = 0, 0
    for t in ALL_TESTS:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except NotImplementedError as e:
            print(f"  SKIP  {t.__name__}  (TODO not done yet: {e})")
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}  {e}")
            failed += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}  {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")

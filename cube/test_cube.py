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


def test_sexy_move_order_6():
    """(R U R' U')^6 = identity. Classic cube identity; catches strip-direction bugs."""
    seq = [MOVE_TO_IDX[m] for m in ["R", "U", "R'", "U'"]] * 6
    s = apply_sequence(solved_state(), seq)
    assert is_solved(s), "(R U R' U')^6 should solve"
    seq5 = [MOVE_TO_IDX[m] for m in ["R", "U", "R'", "U'"]] * 5
    s5 = apply_sequence(solved_state(), seq5)
    assert not is_solved(s5), "(R U R' U')^5 should NOT solve"


def test_RU_order_105():
    """(R U) has order 105. Strong joint test of R and U."""
    seq_one = [MOVE_TO_IDX["R"], MOVE_TO_IDX["U"]]
    s = solved_state()
    for i in range(1, 106):
        s = apply_sequence(s, seq_one)
        if is_solved(s):
            assert i == 105, f"(R U) returned to identity at step {i}, expected 105"
            return
    assert False, "(R U) did not return to identity within 105 applications"


def test_color_counts_preserved():
    """Random scrambles must keep exactly 9 of each color."""
    rng = np.random.default_rng(42)
    state, _ = scramble(50, rng)
    counts = np.bincount(state, minlength=6)
    assert (counts == 9).all(), f"color counts wrong: {counts}"


ALL_TESTS = [
    test_solved_starts_solved,
    test_U_four_times_is_identity,
    test_U_prime_inverts_U,
    test_all_faces_four_turns,
    test_all_inverses,
    test_scramble_then_reverse_solves,
    test_perms_are_permutations,
    test_sexy_move_order_6,
    test_RU_order_105,
    test_color_counts_preserved,
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

"""
Rubik's Cube state and moves, sticker representation.

Sticker layout (54 stickers, indices 0..53):

The cube has 6 faces: U(up), D(down), L(left), R(right), F(front), B(back).
Each face has 9 stickers, numbered 0..8 in reading order when looking
straight at that face:

         0 1 2
         3 4 5         <- U face (indices 0..8 in the state array)
         6 7 8

  9 10 11  18 19 20  27 28 29  36 37 38
  12 13 14 21 22 23  30 31 32  39 40 41   <- L, F, R, B faces
  15 16 17 24 25 26  33 34 35  42 43 44      (indices 9..44)

        45 46 47
        48 49 50       <- D face (indices 45..53)
        51 52 53

So the state is a length-54 numpy array. Each entry is a color 0..5
(one per face). The SOLVED state is:
    [0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9
i.e. U=0, L=1, F=2, R=3, B=4, D=5.

A "move" is a permutation of these 54 positions: applying move m
means new_state[i] = old_state[PERM[m][i]].
"""

import numpy as np

# Face color indices
U, L, F, R, B, D = 0, 1, 2, 3, 4, 5

# 12 moves in QTM. Index 0..11 for the network; string for humans.
MOVES = ["U", "U'", "L", "L'", "F", "F'", "R", "R'", "B", "B'", "D", "D'"]
MOVE_TO_IDX = {m: i for i, m in enumerate(MOVES)}
N_MOVES = len(MOVES)
N_STICKERS = 54
N_COLORS = 6


def solved_state() -> np.ndarray:
    """Return the solved cube as a length-54 int array."""
    return np.repeat(np.arange(6, dtype=np.int8), 9)


def _identity_perm() -> np.ndarray:
    return np.arange(N_STICKERS, dtype=np.int32)


def _rotate_face_cw(perm: np.ndarray, face_start: int) -> None:
    """
    In-place: rotate the 3x3 stickers of one face 90° clockwise.
    The 9 stickers of a face occupy indices [face_start..face_start+8]
    laid out as:
        0 1 2
        3 4 5
        6 7 8
    A CW rotation maps:
        new[0]=old[6], new[1]=old[3], new[2]=old[0],
        new[3]=old[7], new[4]=old[4], new[5]=old[1],
        new[6]=old[8], new[7]=old[5], new[8]=old[2]
    """
    s = face_start
    old = perm[s:s+9].copy()
    mapping = [6, 3, 0, 7, 4, 1, 8, 5, 2]
    for i, j in enumerate(mapping):
        perm[s + i] = old[j]


def _build_U_perm() -> np.ndarray:
    """
    Build the permutation array for a single U (up, 90° CW) move.

    A U move:
      - rotates the 9 stickers on the U face clockwise
      - cycles the TOP ROW of the four side faces:
            F top row -> L top row
            L top row -> B top row
            B top row -> R top row
            R top row -> F top row
        (when viewed from above, the top edges move CW)

    Top rows are:
        L: indices 9, 10, 11
        F: indices 18, 19, 20
        R: indices 27, 28, 29
        B: indices 36, 37, 38
    """
    perm = _identity_perm()

    # Rotate U face stickers CW (U face starts at index 0)
    _rotate_face_cw(perm, 0)

    # Cycle the top rows: F -> L -> B -> R -> F
    # perm[i] tells us "where to read the new value from"
    # So if F's top row goes TO L's top row, then new L top row reads from old F top row:
    #     perm[L_top[k]] = F_top[k]
    L_top = [9, 10, 11]
    F_top = [18, 19, 20]
    R_top = [27, 28, 29]
    B_top = [36, 37, 38]

    for k in range(3):
        perm[L_top[k]] = F_top[k]
        perm[B_top[k]] = L_top[k]
        perm[R_top[k]] = B_top[k]
        perm[F_top[k]] = R_top[k]

    return perm


# === TODO: Generative Task #2 ===
# Build the permutation for each of the remaining 5 face moves.
# Use _build_U_perm above as your template. You'll need to:
#   1. Rotate the face's own 9 stickers CW.
#   2. Identify the four 3-sticker strips on adjacent faces that get cycled.
#   3. Wire up the cycle.
#
# Hint: drawing the cube on paper and labeling sticker indices is by far
# the fastest way. Don't try to do it in your head.

def _build_D_perm() -> np.ndarray:
    raise NotImplementedError("Generative Task #2: implement D move")

def _build_L_perm() -> np.ndarray:
    raise NotImplementedError("Generative Task #2: implement L move")

def _build_R_perm() -> np.ndarray:
    raise NotImplementedError("Generative Task #2: implement R move")

def _build_F_perm() -> np.ndarray:
    raise NotImplementedError("Generative Task #2: implement F move")

def _build_B_perm() -> np.ndarray:
    raise NotImplementedError("Generative Task #2: implement B move")


_BASE_BUILDERS = {
    "U": _build_U_perm,
    "L": _build_L_perm,
    "F": _build_F_perm,
    "R": _build_R_perm,
    "B": _build_B_perm,
    "D": _build_D_perm,
}

# Lazy per-move cache so U-related tests can pass before the other faces
# are implemented.
_PERM_CACHE: dict[int, np.ndarray] = {}

def _get_perm(move_idx: int) -> np.ndarray:
    if move_idx in _PERM_CACHE:
        return _PERM_CACHE[move_idx]
    name = MOVES[move_idx]
    if name.endswith("'"):
        face = name[:-1]
        p = _BASE_BUILDERS[face]()
        p = p[p][p]              # X' = X^3
    else:
        p = _BASE_BUILDERS[name]()
    _PERM_CACHE[move_idx] = p
    return p


def get_perms() -> np.ndarray:
    """Returns a (12, 54) array of all move permutations. Builds all of them."""
    return np.stack([_get_perm(i) for i in range(N_MOVES)], axis=0)


def apply_move(state: np.ndarray, move_idx: int) -> np.ndarray:
    """Apply move `move_idx` to `state` and return the new state."""
    return state[_get_perm(move_idx)]


def apply_sequence(state: np.ndarray, moves: list[int]) -> np.ndarray:
    """Apply a list of move indices in order."""
    for m in moves:
        state = apply_move(state, m)
    return state


def is_solved(state: np.ndarray) -> bool:
    return bool(np.array_equal(state, solved_state()))


def scramble(depth: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Return (scrambled_state, move_sequence) from solved, `depth` random moves."""
    state = solved_state()
    moves = []
    for _ in range(depth):
        m = int(rng.integers(0, N_MOVES))
        state = apply_move(state, m)
        moves.append(m)
    return state, moves

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
  12 13 14  21 22 23  30 31 32  39 40 41   <- L, F, R, B faces
  15 16 17  24 25 26  33 34 35  42 43 44      (indices 9..44)

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


def _build_D_perm() -> np.ndarray:
    """
    Build the permutation array for a single D (down, 90° CW) move.

    A D move:
      - rotates the 9 stickers on the D face clockwise
      - cycles the BOTTOM ROW of the four side faces:
            F bottom row -> L bottom row
            L bottom row -> B bottom row
            B bottom row -> R bottom row
            R bottom row -> F bottom row
        (when viewed from top, the bottom edges move CCW)

    Bottom rows are:
        L: indices 15, 16, 17
        F: indices 24, 25, 26
        R: indices 33, 34, 35
        B: indices 42, 43, 44
    """
    perm = _identity_perm()

    # Rotate D face stickers CW (D face starts at 45)
    _rotate_face_cw(perm, 45)

    # Cycle the bottom rows: F -> R -> B -> L -> F
    # perm[i] tells us "where to read the new value from"
    # So if F's bottom row goes TO R's bottom row, then new R bottom row reads from old F bottom row:
    #     perm[R_bottom[k]] = F_bottom[k]
    L_bottom = [15, 16, 17]
    F_bottom = [24, 25, 26]
    R_bottom = [33, 34, 35]
    B_bottom = [42, 43, 44]

    for k in range(3):
        perm[R_bottom[k]] = F_bottom[k]
        perm[B_bottom[k]] = R_bottom[k]
        perm[L_bottom[k]] = B_bottom[k]
        perm[F_bottom[k]] = L_bottom[k]

    return perm


def _build_L_perm() -> np.ndarray:
    """
    Build the permutation array for a single L (left, 90° CW) move.

    A L move:
      - rotates the 9 stickers on the L face clockwise
      - cycles one COLUMN of the four side faces:
            U left column -> F left column
            F left column -> D left column
            D left column (reversed!) -> B right! column 
            B right! column (reversed!) -> U left column
            

    Rotated columns are:
        U: indices 0, 3, 6
        F: indices 18, 21, 24
        D: indices 45, 48, 51
        B: indices 38, 41, 44
    """
    perm = _identity_perm()

    # Rotate L face stickers CW (L face starts at 9)
    _rotate_face_cw(perm, 9)

    # Cycle the columns: U -> F -> D -(r)-> B -(r)-> U
    # perm[i] tells us "where to read the new value from"
    # So if F's left column goes to D's left column, then new D left column reads from old F left column:
    #     perm[D_left[k]] = F_left[k]
    # For rotations with B's right column indices are reversed, thus:
    #     perm[B_right[k]] = D_left[::-1][k]
    U_left = [0, 3, 6]
    F_left = [18, 21, 24]
    D_left = [45, 48, 51]
    B_right = [38, 41, 44]

    for k in range(3):
        perm[F_left[k]] = U_left[k]
        perm[D_left[k]] = F_left[k]
        perm[B_right[k]] = D_left[::-1][k]
        perm[U_left[k]] = B_right[::-1][k]

    return perm


def _build_R_perm() -> np.ndarray:
    """
    Build the permutation array for a single R (right, 90° CW) move.

    A R move:
      - rotates the 9 stickers on the R face clockwise
      - cycles one COLUMN of the four side faces:
            U right column (reversed!) -> B left! column
            B left! column (reversed!) -> D right column
            D right column -> F right column
            F right column -> U right column

    Rotated columns are:
        U: indices 2, 5, 8
        B: indices 36, 39, 42
        D: indices 47, 50, 53
        F: indices 20, 23, 26
    """
    perm = _identity_perm()

    # Rotate R face stickers CW (R face starts at 27)
    _rotate_face_cw(perm, 27)

    # Cycle the columns: U -(r)-> B -(r)-> D -> F -> U
    # perm[i] tells us "where to read the new value from"
    # So if D's right column goes to F's right column, then new F right column reads from old D right column:
    #     perm[F_right[k]] = D_right[k]
    # For rotations with B's left column indices are reversed, thus:
    #     perm[B_left[k]] = U_right[::-1][k]
    U_right = [2, 5, 8]
    B_left = [36, 39, 42]
    D_right = [47, 50, 53]
    F_right = [20, 23, 26]

    for k in range(3):
        perm[B_left[k]] = U_right[::-1][k]
        perm[D_right[k]] = B_left[::-1][k]
        perm[F_right[k]] = D_right[k]
        perm[U_right[k]] = F_right[k]

    return perm


def _build_F_perm() -> np.ndarray:
    """
    Build the permutation array for a single F (front, 90° CW) move.

    A F move:
    - rotates the 9 stickers on the F face clockwise
    - cycles rows and columns of four faces:
        U bottom row -> R left column
        R left column -> D top row (reversed!)
        D top row -> L right column
        L right column -> U bottom row (reversed!)

    Rotated rows and columns are:
        U indices: 6, 7, 8
        R indices: 27, 39, 33
        D indices: 45, 46, 47
        L indices: 11, 14, 17
    """
    perm = _identity_perm()

    # Rotate F face stickers CW.
    _rotate_face_cw(perm, 18)

    # Cycle rows and columns: U -> R -(r)-> D -> L -(r)-> U
    U_bottom = [6, 7, 8]
    R_left = [27, 30, 33]
    D_top = [45, 46, 47]
    L_right = [11, 14, 17]

    for k in range(3):
        perm[R_left[k]] = U_bottom[k]
        perm[D_top[k]] = R_left[::-1][k]
        perm[L_right[k]] = D_top[k]
        perm[U_bottom[k]] = L_right[::-1][k]

    return perm


def _build_B_perm() -> np.ndarray:
    """
    Build the permutation array for a single B (back, 90° CW) move.

    A B move:
    - rotates the 9 stickers on the B face clockwise
    - cycles rows and columns of four faces:
        U top row -> L left column (reversed!)
        L left column -> D bottom row
        D bottom row -> R right column (reversed!)
        R right column -> U top row

    Rotated rows and columns are:
        U indices: 0, 1, 2
        L indices: 9, 12, 15
        D indices: 51, 52, 53
        R indices: 29, 32, 35
    """
    perm = _identity_perm()

    # Rotate B face stickers CW.
    _rotate_face_cw(perm, 36)

    # Cycle rows and columns: U -(r)-> L -> D -(r)-> R -> U
    U_top = [0, 1, 2]
    L_left = [9, 12, 15]
    D_bottom = [51, 52, 53]
    R_right = [29, 32, 35]

    for k in range(3):
        perm[L_left[k]] = U_top[::-1][k]
        perm[D_bottom[k]] = L_left[k]
        perm[R_right[k]] = D_bottom[::-1][k]
        perm[U_top[k]] = R_right[k]

    return perm


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


def inverse_move(move_id: int):
    move = MOVES[move_id]
    if len(move) == 2:
        return MOVE_TO_IDX[move[0]]
    else:
        return MOVE_TO_IDX[move + "'"]


def scramble(depth: int, rng: np.random.Generator) -> tuple[np.ndarray, list[int]]:
    """Return (scrambled_state, move_sequence) from solved, `depth` random moves."""
    state = solved_state()
    moves = []

    def valid_scramble(m: int):
        if moves and moves[-1] == inverse_move(m):
            return False
        if len(moves) >= 2 and moves[-2] == moves[-1]:
            return moves[-1] != m
        return True

    def sample_move():
        while True:
            m = int(rng.integers(0, N_MOVES))
            if valid_scramble(m):
                return m

    for _ in range(depth):
        m = sample_move()
        state = apply_move(state, m)
        moves.append(m)
    return state, moves

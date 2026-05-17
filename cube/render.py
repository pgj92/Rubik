"""
Rendering helpers for the cube. Two flavors:

  - render_ansi(state) -> str
        Colored text suitable for printing in a terminal that supports ANSI
        256-color escape codes (most modern terminals).

  - render_rgb_array(state) -> np.ndarray
        An (H, W, 3) uint8 image via matplotlib, drawn as the unfolded
        cube cross. Used for "rgb_array" render_mode and for GIF saving.

  - animate(states, interval_ms, save_path=None)
        Step through a list of states either by opening an interactive
        matplotlib window (save_path=None) or saving an animated GIF.
"""
from __future__ import annotations

import numpy as np

from .cube import N_STICKERS

# Standard western cube colors
COLORS = [
    "#FFFFFF",  # 0 U  white
    "#FF8C00",  # 1 L  orange
    "#1FAB47",  # 2 F  green
    "#D32F2F",  # 3 R  red
    "#1E88E5",  # 4 B  blue
    "#FFE600",  # 5 D  yellow
]

# ANSI 256-color background codes
ANSI = {
    0: "\033[48;5;255m",  # white
    1: "\033[48;5;208m",  # orange
    2: "\033[48;5;34m",   # green
    3: "\033[48;5;196m",  # red
    4: "\033[48;5;21m",   # blue
    5: "\033[48;5;226m",  # yellow
}
ANSI_RESET = "\033[0m"

# Where each face lives in the unfolded 9-row x 12-col grid.
# Layout:
#         U U U
#         U U U
#         U U U
#   L L L F F F R R R B B B
#   L L L F F F R R R B B B
#   L L L F F F R R R B B B
#         D D D
#         D D D
#         D D D
FACE_OFFSETS = {
    0: (0, 3),  # U
    1: (3, 0),  # L
    2: (3, 3),  # F
    3: (3, 6),  # R
    4: (3, 9),  # B
    5: (6, 3),  # D
}


def _sticker_cells(state: np.ndarray):
    """Yield (row, col, color_idx) for each of the 54 stickers."""
    for i in range(N_STICKERS):
        face = i // 9
        within = i % 9
        r_off, c_off = FACE_OFFSETS[face]
        yield r_off + within // 3, c_off + within % 3, int(state[i])


def render_ansi(state: np.ndarray) -> str:
    """Colored text rendering for terminals."""
    grid: list[list[int | None]] = [[None] * 12 for _ in range(9)]
    for r, c, color in _sticker_cells(state):
        grid[r][c] = color

    lines = []
    for row in grid:
        line = ""
        for cell in row:
            if cell is None:
                line += "   "
            else:
                line += ANSI[cell] + "   " + ANSI_RESET
        lines.append(line)
    return "\n".join(lines)


def _draw_unfolded(ax, state: np.ndarray, title: str | None = None):
    """Draw the unfolded cube cross on a matplotlib Axes."""
    import matplotlib.patches as patches

    ax.clear()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

    for r, c, color_idx in _sticker_cells(state):
        rect = patches.Rectangle(
            (c, r), 1, 1,
            edgecolor="black", facecolor=COLORS[color_idx], linewidth=1.5,
        )
        ax.add_patch(rect)

    if title:
        ax.set_title(title, fontsize=12)


def render_rgb_array(state: np.ndarray, figsize=(6, 4.5), dpi=100) -> np.ndarray:
    """Single-frame RGB rendering. Returns (H, W, 3) uint8."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plt.Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    _draw_unfolded(ax, state)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def animate(
    states: list[np.ndarray],
    titles: list[str] | None = None,
    interval_ms: int = 500,
    save_path: str | None = None,
):
    """
    Animate a sequence of cube states.

    If save_path is given, saves an animated GIF there.
    Otherwise opens an interactive matplotlib window (blocking).
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 4.5))

    def update(i):
        title = titles[i] if titles else f"Step {i}"
        _draw_unfolded(ax, states[i], title=title)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(states),
        interval=interval_ms, blit=False, repeat=False,
    )

    if save_path:
        fps = max(1, 1000 // interval_ms)
        anim.save(save_path, writer=PillowWriter(fps=fps))
        plt.close(fig)
    else:
        plt.show()

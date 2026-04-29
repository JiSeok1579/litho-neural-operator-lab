"""Binary mask pattern generators.

Each function returns a 2D ``torch.float32`` tensor with values in {0, 1}
defined on the grid passed in. ``1`` means "open / clear" (full
transmission), ``0`` means "blocked / opaque". The complex transmission
function used in optics is built from these binary masks via
:mod:`src.mask.transmission`.

All sizes (pitch, width, radius, gap, length) are given in the same
normalized length units used by the grid.
"""

from __future__ import annotations

import math

import torch

from src.common.grid import Grid2D


def _zeros(grid: Grid2D) -> torch.Tensor:
    return torch.zeros((grid.n, grid.n), device=grid.device, dtype=grid.dtype)


def line_space(grid: Grid2D, pitch: float, duty: float = 0.5,
               orientation: str = "vertical") -> torch.Tensor:
    """Periodic line-space grating.

    ``pitch`` is the period (one line + one space) in normalized units.
    ``duty`` is the fraction of the pitch that is open. ``orientation``
    is "vertical" (lines run along y) or "horizontal" (lines run along x).
    """
    if pitch <= 0:
        raise ValueError("pitch must be positive")
    if not (0.0 < duty < 1.0):
        raise ValueError("duty must be in (0, 1)")

    x, y = grid.axes()
    coord = x if orientation == "vertical" else y
    # phase in [0, 1) along the periodic direction
    u = (coord / pitch) % 1.0
    line_1d = (u < duty).to(grid.dtype)
    if orientation == "vertical":
        return line_1d.unsqueeze(0).expand(grid.n, grid.n).contiguous()
    return line_1d.unsqueeze(1).expand(grid.n, grid.n).contiguous()


def contact_hole(grid: Grid2D, radius: float, center: tuple[float, float] = (0.0, 0.0)) -> torch.Tensor:
    if radius <= 0:
        raise ValueError("radius must be positive")
    X, Y = grid.meshgrid()
    cx, cy = center
    return ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).to(grid.dtype)


def isolated_line(grid: Grid2D, width: float, orientation: str = "vertical") -> torch.Tensor:
    if width <= 0:
        raise ValueError("width must be positive")
    X, Y = grid.meshgrid()
    coord = X if orientation == "vertical" else Y
    return (coord.abs() <= width / 2).to(grid.dtype)


def two_bar(grid: Grid2D, width: float, gap: float, orientation: str = "vertical") -> torch.Tensor:
    if width <= 0 or gap <= 0:
        raise ValueError("width and gap must be positive")
    X, Y = grid.meshgrid()
    coord = X if orientation == "vertical" else Y
    half = width / 2
    offset = (gap + width) / 2
    bar_left = ((coord + offset).abs() <= half)
    bar_right = ((coord - offset).abs() <= half)
    return (bar_left | bar_right).to(grid.dtype)


def elbow(grid: Grid2D, width: float, length: float) -> torch.Tensor:
    """L-shaped pattern: a horizontal arm and a vertical arm sharing a corner.

    The corner sits at the origin; arms extend in +x and +y by ``length``.
    """
    if width <= 0 or length <= 0:
        raise ValueError("width and length must be positive")
    X, Y = grid.meshgrid()
    horizontal_arm = (Y.abs() <= width / 2) & (X >= 0) & (X <= length)
    vertical_arm = (X.abs() <= width / 2) & (Y >= 0) & (Y <= length)
    return (horizontal_arm | vertical_arm).to(grid.dtype)


def random_binary(grid: Grid2D, fill_fraction: float = 0.5,
                  block_size: int = 1, seed: int | None = None) -> torch.Tensor:
    """Random binary pattern at the requested density.

    ``block_size`` lets you generate larger features (each block is
    block_size x block_size pixels with the same value).
    """
    if not (0.0 < fill_fraction < 1.0):
        raise ValueError("fill_fraction must be in (0, 1)")
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    g = torch.Generator(device=grid.device)
    if seed is not None:
        g.manual_seed(seed)
    nb = math.ceil(grid.n / block_size)
    blocks = (torch.rand((nb, nb), generator=g, device=grid.device) < fill_fraction).to(grid.dtype)
    if block_size == 1:
        return blocks
    expanded = blocks.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    return expanded[: grid.n, : grid.n].contiguous()

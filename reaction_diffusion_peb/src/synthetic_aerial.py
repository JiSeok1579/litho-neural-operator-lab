"""Synthetic aerial / exposure intensity maps for the PEB submodule.

These functions deliberately do not import from the main repo's optics
or mask modules. The submodule starts self-contained and only later
plugs into the main pipeline through a file-based interface
(`outputs/aerial_image.npy`).

Every function returns a 2D ``torch.float32`` tensor of shape
``(grid_size, grid_size)`` with values in ``[0, 1]`` after the final
``normalize_intensity`` step.
"""

from __future__ import annotations

import math

import torch


def _coord_axis(grid_size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Pixel-centered axis ranging from ``-grid_size/2 + 0.5`` to
    ``grid_size/2 - 0.5``."""
    return torch.arange(grid_size, dtype=dtype) - (grid_size - 1) / 2.0


def gaussian_spot(
    grid_size: int,
    sigma_px: float,
    center: tuple[float, float] | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """2D Gaussian intensity centered at ``center`` (default: middle).

    ``sigma_px`` is the Gaussian sigma in pixels. Peak value is 1.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    if center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = float(center[0]), float(center[1])
    axis = _coord_axis(grid_size, dtype)
    X, Y = torch.meshgrid(axis, axis, indexing="xy")
    return torch.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma_px ** 2))


def line_space(
    grid_size: int,
    pitch_px: float,
    duty: float = 0.5,
    contrast: float = 1.0,
    orientation: str = "vertical",
    smooth_px: float | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Periodic line-space exposure pattern.

    A "line" is the bright phase (intensity ~ ``contrast``); a "space"
    is the dark phase (intensity ~ ``1 - contrast``). With
    ``contrast = 1`` the dark phase is exactly 0.

    ``smooth_px`` (optional) replaces the binary edges with a
    half-cosine of that pixel width — useful when a sharp IC stresses
    the PINN unnecessarily.
    """
    if pitch_px <= 0:
        raise ValueError("pitch_px must be positive")
    if not (0.0 < duty < 1.0):
        raise ValueError("duty must be in (0, 1)")
    if not (0.0 < contrast <= 1.0):
        raise ValueError("contrast must be in (0, 1]")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("orientation must be 'vertical' or 'horizontal'")

    axis = _coord_axis(grid_size, dtype)
    coord = axis if orientation == "vertical" else axis
    # Periodic phase u in [0, 1)
    u = (coord / pitch_px) % 1.0  # shape (grid_size,)

    if smooth_px is None or smooth_px <= 0:
        line_1d = (u < duty).to(dtype) * contrast + (u >= duty).to(dtype) * (1.0 - contrast)
    else:
        # Smooth transitions of width ``smooth_px`` around u=0 and u=duty
        edge_width = smooth_px / pitch_px
        # Distance into the edge band, normalized to [0, 1]
        d_open_edge = ((duty - u) / edge_width).clamp(min=0.0, max=1.0)
        d_close_edge = ((u - 0.0) / edge_width).clamp(min=0.0, max=1.0)
        # Cosine-roll-off between the two transitions; result in [0, 1]
        weight_open = 0.5 * (1.0 - torch.cos(math.pi * (d_open_edge.minimum(d_close_edge))))
        line_1d = weight_open * contrast + (1.0 - weight_open) * (1.0 - contrast)

    if orientation == "vertical":
        return line_1d.unsqueeze(0).expand(grid_size, grid_size).contiguous().to(dtype)
    return line_1d.unsqueeze(1).expand(grid_size, grid_size).contiguous().to(dtype)


def contact_array(
    grid_size: int,
    pitch_px: float,
    sigma_px: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Periodic 2D Gaussian array — a "contact-hole-like" exposure map.

    Centers sit on a regular grid spaced by ``pitch_px``; each is a
    Gaussian of standard deviation ``sigma_px``. Result peaks at 1
    (when sigma_px is small enough that adjacent Gaussians do not
    overlap appreciably) and is then renormalized.
    """
    if pitch_px <= 0:
        raise ValueError("pitch_px must be positive")
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    axis = _coord_axis(grid_size, dtype)
    X, Y = torch.meshgrid(axis, axis, indexing="xy")
    # Wrap onto [-pitch/2, pitch/2)
    Xc = ((X + pitch_px / 2.0) % pitch_px) - pitch_px / 2.0
    Yc = ((Y + pitch_px / 2.0) % pitch_px) - pitch_px / 2.0
    out = torch.exp(-(Xc ** 2 + Yc ** 2) / (2.0 * sigma_px ** 2))
    return normalize_intensity(out)


def two_spot(
    grid_size: int,
    sigma_px: float,
    separation_px: float,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Two Gaussian spots separated horizontally by ``separation_px``.

    Useful for studying the diffusion / reaction interaction between
    neighbouring acid distributions.
    """
    if separation_px <= 0:
        raise ValueError("separation_px must be positive")
    half = separation_px / 2.0
    g1 = gaussian_spot(grid_size, sigma_px=sigma_px, center=(-half, 0.0), dtype=dtype)
    g2 = gaussian_spot(grid_size, sigma_px=sigma_px, center=(+half, 0.0), dtype=dtype)
    return normalize_intensity(g1 + g2)


def normalize_intensity(I: torch.Tensor) -> torch.Tensor:
    """Affine-rescale a non-negative intensity field to ``[0, 1]``."""
    I_min = I.min()
    I_max = I.max()
    span = (I_max - I_min)
    if span.item() < 1e-12:
        return torch.zeros_like(I)
    return (I - I_min) / span

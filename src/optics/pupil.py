"""Projection-lens pupil for scalar / coherent imaging.

The pupil is a low-pass filter in spatial-frequency space that models the
finite numerical aperture of the projection lens:

    P(fx, fy) = 1 if sqrt(fx^2 + fy^2) <= NA / wavelength, else 0

Units convention. The grid extent is expressed in the same length units as
the ``wavelength`` argument. The default ``wavelength = 1.0`` means
``Grid2D.extent`` is already counted in wavelengths, so the cutoff
``NA / wavelength`` is just ``NA`` in cycles per wavelength.

Two variants are provided:

- :func:`circular_pupil` — the hard binary cutoff specified by plain
  Fourier optics. Use this whenever exact NA-cutoff behavior matters.
- :func:`apodized_circular_pupil` — a cosine-tapered pupil. The smooth
  edge produces continuous gradients, which helps Phase-3 inverse mask
  optimization and Phase-8 surrogate training avoid the pathologies of a
  step-function filter.
"""

from __future__ import annotations

import math

import torch

from src.common.grid import Grid2D


def _validate(NA: float, wavelength: float) -> None:
    if NA <= 0:
        raise ValueError("NA must be positive")
    if wavelength <= 0:
        raise ValueError("wavelength must be positive")


def circular_pupil(grid: Grid2D, NA: float, wavelength: float = 1.0) -> torch.Tensor:
    """Hard circular pupil with cutoff at ``NA / wavelength``."""
    _validate(NA, wavelength)
    cutoff = NA / wavelength
    fr = grid.radial_freq()
    return (fr <= cutoff).to(grid.dtype)


def circular_pupil_at(
    grid: Grid2D,
    NA: float,
    center_freq: tuple[float, float],
    wavelength: float = 1.0,
) -> torch.Tensor:
    """Hard circular pupil centered at ``(cx, cy)`` in cycles per length unit.

    Used by :mod:`src.optics.partial_coherence` to build off-axis pupils for
    each source point in the Hopkins integral.
    """
    _validate(NA, wavelength)
    cx, cy = center_freq
    fx, fy = grid.freq_meshgrid()
    cutoff = NA / wavelength
    fr = torch.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
    return (fr <= cutoff).to(grid.dtype)


def apodized_circular_pupil(
    grid: Grid2D,
    NA: float,
    wavelength: float = 1.0,
    roll_off: float = 0.05,
) -> torch.Tensor:
    """Circular pupil with a cosine-tapered edge.

    Inside ``(1 - roll_off) * cutoff`` the response is 1; outside ``cutoff``
    it is 0; in between it follows a half-cosine roll-off. ``roll_off`` is
    a fraction of the cutoff radius. ``roll_off = 0`` reduces to
    :func:`circular_pupil`.
    """
    _validate(NA, wavelength)
    if not (0.0 <= roll_off <= 1.0):
        raise ValueError("roll_off must be in [0, 1]")
    cutoff = NA / wavelength
    inner = cutoff * (1.0 - roll_off)
    fr = grid.radial_freq()
    out = torch.zeros_like(fr)
    out = torch.where(fr <= inner, torch.ones_like(fr), out)
    if roll_off > 0:
        in_taper = (fr > inner) & (fr <= cutoff)
        u = (fr - inner) / (cutoff - inner + 1e-12)
        taper = 0.5 * (1.0 + torch.cos(math.pi * u))
        out = torch.where(in_taper, taper, out)
    return out

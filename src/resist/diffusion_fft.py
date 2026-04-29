"""Exact diffusion via the analytic Fourier heat kernel.

The 2D heat equation has closed-form solution in Fourier space:

    A_hat(fx, fy, t) = A_hat(fx, fy, 0) * exp(-4 * pi**2 * D * t * |f|**2)

which is equivalent to a real-space convolution with a Gaussian of
standard deviation ``sigma = sqrt(2 * D * t)`` along each axis. The
"diffusion length" is conventionally ``L = sqrt(2 * D * t)``.

Two helpers are provided:

- :func:`diffuse_fft` — explicit ``D, t`` parameterization.
- :func:`diffuse_fft_by_length` — single-knob ``L`` parameterization
  (more convenient for sweeps because it absorbs ``D`` and ``t`` into
  one geometric scale).
"""

from __future__ import annotations

import math

import torch

from src.common.fft_utils import fft2c, ifft2c
from src.common.grid import Grid2D


def diffuse_fft(
    A0: torch.Tensor,
    grid: Grid2D,
    D: float,
    t: float,
) -> torch.Tensor:
    """Exact heat-kernel diffusion of ``A0`` for time ``t`` at coefficient ``D``.

    Returns the real part of the inverse FFT; for real ``A0`` the
    imaginary part is zero up to round-off.
    """
    if D < 0:
        raise ValueError("D must be non-negative")
    if t < 0:
        raise ValueError("t must be non-negative")
    if D == 0 or t == 0:
        return A0.clone()
    fr = grid.radial_freq()
    decay = torch.exp(-4.0 * math.pi ** 2 * D * t * fr ** 2)
    A_hat = fft2c(A0.to(torch.complex64))
    return ifft2c(A_hat * decay).real


def diffuse_fft_by_length(
    A0: torch.Tensor,
    grid: Grid2D,
    diffusion_length: float,
) -> torch.Tensor:
    """Same diffusion as :func:`diffuse_fft` but parameterized by
    ``L = sqrt(2 D t)``.

    Equivalent decay factor: ``exp(-2 * pi**2 * L**2 * |f|**2)`` since
    ``4 * pi**2 * D * t = 2 * pi**2 * L**2``.
    """
    if diffusion_length < 0:
        raise ValueError("diffusion_length must be non-negative")
    if diffusion_length == 0:
        return A0.clone()
    fr = grid.radial_freq()
    decay = torch.exp(-2.0 * math.pi ** 2 * (diffusion_length ** 2) * fr ** 2)
    A_hat = fft2c(A0.to(torch.complex64))
    return ifft2c(A_hat * decay).real

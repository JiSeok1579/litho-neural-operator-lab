"""Exact heat-kernel diffusion via the analytic Fourier solution.

Solves the same equation as ``diffusion_fd`` (pure 2D diffusion with
periodic boundary conditions), but evaluates the closed-form Fourier-
domain solution in one FFT round-trip:

    H_hat(fx, fy, t) = H_hat(fx, fy, 0) * exp(-4 * pi^2 * D_H * t * |f|^2)

Equivalent to a real-space convolution with a Gaussian of standard
deviation ``sigma = sqrt(2 * D_H * t)``. Conventionally the
"diffusion length" is ``L = sqrt(2 * D_H * t)``.

Units:
    DH_nm2_s : [nm^2 / s]
    t_end_s  : [s]
    dx_nm    : [nm]
    L_nm     : [nm]
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.fft_utils import fft2c, freq_grid_nm, ifft2c


def diffuse_fft(
    H0: torch.Tensor,
    DH_nm2_s: float,
    t_end_s: float,
    dx_nm: float,
) -> torch.Tensor:
    """Exact heat-kernel diffusion of ``H0`` for ``t_end_s`` seconds.

    Returns the real part of the inverse FFT — for a real-valued ``H0``
    the imaginary part is zero up to round-off.
    """
    if DH_nm2_s < 0:
        raise ValueError("DH_nm2_s must be non-negative")
    if t_end_s < 0:
        raise ValueError("t_end_s must be non-negative")
    if DH_nm2_s == 0 or t_end_s == 0:
        return H0.clone()
    grid_size = H0.shape[-1]
    fx, fy = freq_grid_nm(grid_size, dx_nm, dtype=H0.dtype, device=H0.device)
    fr2 = fx * fx + fy * fy
    decay = torch.exp(-4.0 * math.pi ** 2 * DH_nm2_s * t_end_s * fr2)
    H_hat = fft2c(H0.to(torch.complex64))
    return ifft2c(H_hat * decay).real


def diffuse_fft_by_length(
    H0: torch.Tensor,
    L_nm: float,
    dx_nm: float,
) -> torch.Tensor:
    """Same diffusion but parameterized by the diffusion length
    ``L = sqrt(2 * D_H * t)``.

    Equivalent decay factor is ``exp(-2 * pi^2 * L^2 * |f|^2)`` since
    ``4 * pi^2 * D * t = 2 * pi^2 * L^2``.
    """
    if L_nm < 0:
        raise ValueError("L_nm must be non-negative")
    if L_nm == 0:
        return H0.clone()
    grid_size = H0.shape[-1]
    fx, fy = freq_grid_nm(grid_size, dx_nm, dtype=H0.dtype, device=H0.device)
    fr2 = fx * fx + fy * fy
    decay = torch.exp(-2.0 * math.pi ** 2 * L_nm * L_nm * fr2)
    H_hat = fft2c(H0.to(torch.complex64))
    return ifft2c(H_hat * decay).real

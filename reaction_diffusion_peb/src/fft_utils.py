"""Centered 2D FFT helpers for the PEB submodule.

Self-contained: not imported from the main repo's ``src/common``. The
submodule keeps its own copy so it can stay decoupled while the
convention (``fftshift(fft2(ifftshift(x)))`` with ``norm='ortho'``)
matches the main project.
"""

from __future__ import annotations

import torch


def fft2c(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D forward FFT over the last two dims."""
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm=norm),
        dim=(-2, -1),
    )


def ifft2c(X: torch.Tensor, norm: str = "ortho") -> torch.Tensor:
    """Centered 2D inverse FFT over the last two dims."""
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), norm=norm),
        dim=(-2, -1),
    )


def freq_grid_nm(grid_size: int, grid_spacing_nm: float,
                 dtype: torch.dtype = torch.float32,
                 device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(fx, fy)`` 2D meshgrids in cycles per nanometer.

    Compatible with the centered FFT convention: DC is at the centered
    index ``(N/2, N/2)``. Spacing in frequency is
    ``df = 1 / (N * dx_nm)`` cycles per nm.
    """
    if grid_size < 1:
        raise ValueError("grid_size must be positive")
    if grid_spacing_nm <= 0:
        raise ValueError("grid_spacing_nm must be positive")
    df = 1.0 / (grid_size * grid_spacing_nm)
    coord = (torch.arange(grid_size, dtype=dtype, device=device)
             - grid_size / 2.0) * df
    return torch.meshgrid(coord, coord, indexing="xy")

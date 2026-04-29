"""Centered 2D FFT helpers.

The lab uses a single FFT convention everywhere:

    fft2c(x)  = fftshift( fft2( ifftshift(x) ) )
    ifft2c(X) = fftshift( ifft2( ifftshift(X) ) )

This places the DC component at index (N/2, N/2) in both real and frequency
space, which matches the centered grids in :mod:`grid` and matplotlib's
``imshow`` extent convention. No raw ``torch.fft.fft2`` calls should appear
outside this module.
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


def amplitude(z: torch.Tensor) -> torch.Tensor:
    return z.abs()


def phase(z: torch.Tensor) -> torch.Tensor:
    return torch.angle(z)


def log_amplitude(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return torch.log10(z.abs() + eps)

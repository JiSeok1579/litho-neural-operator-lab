"""Scalar diffraction in 2D.

Given a thin-mask complex transmission function t(x, y), the diffraction
spectrum is its 2D Fourier transform

    T(fx, fy) = F{ t(x, y) }

This module currently provides only the spectrum computation — pupil
filtering, coherent imaging, and aerial intensity belong to the Phase 2
modules under :mod:`src.optics.coherent_imaging`.
"""

from __future__ import annotations

import torch

from src.common.fft_utils import fft2c, ifft2c


def diffraction_spectrum(transmission: torch.Tensor) -> torch.Tensor:
    """Centered 2D FFT of a complex transmission function."""
    if not torch.is_complex(transmission):
        raise TypeError("transmission must be a complex tensor (use binary_transmission first)")
    return fft2c(transmission)


def reconstruct_field(spectrum: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`diffraction_spectrum` (with no pupil filtering)."""
    return ifft2c(spectrum)

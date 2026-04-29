"""Partial-coherent imaging via Hopkins source integration.

For a source distribution ``J_s(sigma_x, sigma_y)`` the aerial image is

    I(r) = sum over source points  w_s * |E_s(r)|^2,
    E_s(r) = F^-1{ T(f) * P_s(f) }

where ``P_s`` is a circular pupil **centered at**
``(sigma_x, sigma_y) * NA / wavelength`` in spatial-frequency space.

This module evaluates the sum in batched form: all shifted pupils are
stacked along a leading batch dimension and the FFT chain runs once on
``(K, N, N)`` tensors. For typical demo settings (a few dozen non-zero
source points on a 256^2 grid) this comfortably fits in 16 GB of VRAM.
"""

from __future__ import annotations

import torch

from src.common.fft_utils import fft2c, ifft2c
from src.common.grid import Grid2D
from src.optics.source import source_points


def _build_shifted_pupils(
    grid: Grid2D,
    NA: float,
    sigmas: torch.Tensor,
    wavelength: float,
) -> torch.Tensor:
    """Return ``(K, N, N)`` stack of pupils centered at each source sigma.

    Sign convention. We use ``center_freq = +sigma * NA / wavelength``
    (matches the pseudocode in study plan §4.5). For symmetric source
    distributions (annular, dipole, quadrupole, random-with-symmetry) this
    convention is identical to the equivalent spectrum-shift formulation.
    """
    K = sigmas.shape[0]
    fx, fy = grid.freq_meshgrid()  # each (N, N)
    cutoff = NA / wavelength
    cx = sigmas[:, 0] * NA / wavelength    # (K,)
    cy = sigmas[:, 1] * NA / wavelength    # (K,)
    fxK = fx.unsqueeze(0)                  # (1, N, N)
    fyK = fy.unsqueeze(0)
    cxK = cx.view(K, 1, 1)
    cyK = cy.view(K, 1, 1)
    fr = torch.sqrt((fxK - cxK) ** 2 + (fyK - cyK) ** 2)
    return (fr <= cutoff).to(grid.dtype)   # (K, N, N)


def partial_coherent_aerial_image(
    transmission: torch.Tensor,
    grid: Grid2D,
    source: torch.Tensor,
    NA: float,
    wavelength: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Hopkins-integrated aerial image.

    With ``normalize=True`` the result is divided by the source's total
    weight so each source produces an aerial on a comparable scale. With
    ``normalize=False`` the absolute intensity scales with the number /
    weight of source points, which is the right behavior for inverse
    optimization.
    """
    if not torch.is_complex(transmission):
        raise TypeError("transmission must be complex")
    if torch.is_complex(source):
        raise TypeError("source must be real-valued")
    if source.ndim != 2:
        raise ValueError("source must be a 2D tensor")

    sigmas, weights = source_points(source.to(grid.device))
    if sigmas.shape[0] == 0:
        return torch.zeros((grid.n, grid.n), device=grid.device, dtype=grid.dtype)

    pupils = _build_shifted_pupils(grid, NA=NA, sigmas=sigmas, wavelength=wavelength)
    # Spectrum is (N, N) -> broadcast against (K, N, N) pupils.
    spectrum = fft2c(transmission)
    fields = ifft2c(spectrum.unsqueeze(0) * pupils)             # (K, N, N)
    intensities = fields.real ** 2 + fields.imag ** 2           # (K, N, N)
    aerial = (weights.view(-1, 1, 1) * intensities).sum(dim=0)  # (N, N)

    if normalize:
        total_weight = weights.sum()
        aerial = aerial / total_weight.clamp(min=1e-12)
    return aerial

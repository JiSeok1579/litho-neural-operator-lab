"""Source-shape factories for partial-coherent imaging.

A source is a 2D non-negative tensor in normalized partial-coherence
coordinates ``(sigma_x, sigma_y) in [-1, 1]^2``. Each non-zero pixel
represents a source point with the corresponding weight; the weight is
later distributed across the Hopkins integral in
:mod:`src.optics.partial_coherence`.

``sigma`` is the standard lithography partial-coherence factor:
``sin(theta_source) = sigma * NA``. ``sigma = 0`` is on-axis (coherent)
illumination; ``sigma = 1`` is at the lens NA boundary; ``sigma > 1`` does
not pass the lens and is unused.
"""

from __future__ import annotations

import torch


def sigma_axis(n_sigma: int = 31, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """1D axis of length ``n_sigma`` evenly spanning ``[-1, 1]``.

    For ``n_sigma`` odd, the center pixel is bit-exactly ``sigma = 0``: built
    via ``arange(-half, half+1) / half`` rather than ``linspace`` so the
    center value is the integer ``0``, not a rounded ``5e-8``. The
    bit-exactness matters for the partial-coherence sanity test that
    compares a delta source against direct coherent imaging — a tiny
    nonzero center shifts the pupil enough to swap a few boundary pixels.
    """
    if n_sigma < 3:
        raise ValueError("n_sigma must be >= 3")
    if n_sigma % 2 == 1:
        half = n_sigma // 2
        idx = torch.arange(-half, half + 1, dtype=dtype)
        return idx / half
    return torch.linspace(-1.0, 1.0, n_sigma, dtype=dtype)


def sigma_meshgrid(n_sigma: int = 31, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    s = sigma_axis(n_sigma, dtype=dtype)
    return torch.meshgrid(s, s, indexing="xy")


def coherent_source(n_sigma: int = 31, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Single point at the origin (the on-axis coherent illumination)."""
    if n_sigma % 2 == 0:
        raise ValueError("coherent_source needs an odd n_sigma so the center pixel exists")
    src = torch.zeros((n_sigma, n_sigma), dtype=dtype)
    src[n_sigma // 2, n_sigma // 2] = 1.0
    return src


def annular_source(
    n_sigma: int = 31,
    sigma_inner: float = 0.3,
    sigma_outer: float = 0.7,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if not (0.0 <= sigma_inner < sigma_outer <= 1.0):
        raise ValueError("require 0 <= sigma_inner < sigma_outer <= 1")
    sx, sy = sigma_meshgrid(n_sigma, dtype=dtype)
    sr = torch.sqrt(sx ** 2 + sy ** 2)
    return ((sr >= sigma_inner) & (sr <= sigma_outer)).to(dtype)


def dipole_source(
    n_sigma: int = 31,
    sigma_center: float = 0.5,
    sigma_radius: float = 0.2,
    axis: str = "x",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Two circular pole regions centered at ``(+/-sigma_center, 0)`` (axis='x')
    or ``(0, +/-sigma_center)`` (axis='y')."""
    if axis not in ("x", "y"):
        raise ValueError("axis must be 'x' or 'y'")
    if sigma_radius <= 0:
        raise ValueError("sigma_radius must be positive")
    sx, sy = sigma_meshgrid(n_sigma, dtype=dtype)
    if axis == "x":
        d1 = torch.sqrt((sx - sigma_center) ** 2 + sy ** 2)
        d2 = torch.sqrt((sx + sigma_center) ** 2 + sy ** 2)
    else:
        d1 = torch.sqrt(sx ** 2 + (sy - sigma_center) ** 2)
        d2 = torch.sqrt(sx ** 2 + (sy + sigma_center) ** 2)
    return ((d1 <= sigma_radius) | (d2 <= sigma_radius)).to(dtype)


def quadrupole_source(
    n_sigma: int = 31,
    sigma_center: float = 0.5,
    sigma_radius: float = 0.2,
    diagonal: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Four circular pole regions.

    With ``diagonal=False``: at (+/-c, 0) and (0, +/-c) (cross / plus-shape).
    With ``diagonal=True``: at (+/-c, +/-c) / sqrt(2) (X-shape).
    """
    if sigma_radius <= 0:
        raise ValueError("sigma_radius must be positive")
    sx, sy = sigma_meshgrid(n_sigma, dtype=dtype)
    out = torch.zeros((n_sigma, n_sigma), dtype=dtype)
    if diagonal:
        c = sigma_center / (2 ** 0.5)
        centers = [(c, c), (-c, c), (c, -c), (-c, -c)]
    else:
        c = sigma_center
        centers = [(c, 0.0), (-c, 0.0), (0.0, c), (0.0, -c)]
    for cx, cy in centers:
        d = torch.sqrt((sx - cx) ** 2 + (sy - cy) ** 2)
        out = torch.maximum(out, (d <= sigma_radius).to(dtype))
    return out


def random_source(
    n_sigma: int = 31,
    fill_fraction: float = 0.05,
    sigma_max: float = 0.7,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Random binary source within a disk of radius ``sigma_max``.

    Each pixel inside the disk is independently set to 1 with probability
    ``fill_fraction``; the rest stay 0. Useful as a stress test or as
    "freeform" illumination for source-mask co-optimization studies.
    """
    if not (0.0 < fill_fraction < 1.0):
        raise ValueError("fill_fraction must be in (0, 1)")
    if not (0.0 < sigma_max <= 1.0):
        raise ValueError("sigma_max must be in (0, 1]")
    sx, sy = sigma_meshgrid(n_sigma, dtype=dtype)
    sr = torch.sqrt(sx ** 2 + sy ** 2)
    g = torch.Generator().manual_seed(seed)
    rand = torch.rand((n_sigma, n_sigma), generator=g, dtype=dtype)
    return ((rand < fill_fraction) & (sr <= sigma_max)).to(dtype)


def source_points(source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a 2D source tensor into ``(sigmas, weights)``.

    Returns:
        sigmas: ``(K, 2)`` tensor of (sigma_x, sigma_y) values for each
                non-zero source pixel.
        weights: ``(K,)`` tensor of the corresponding weights.
    """
    if source.ndim != 2:
        raise ValueError("source must be a 2D tensor")
    n = source.shape[-1]
    s_axis = sigma_axis(n, dtype=source.dtype).to(source.device)
    nz = (source > 0).nonzero(as_tuple=False)  # (K, 2) [row, col]
    if nz.numel() == 0:
        return torch.zeros((0, 2), dtype=source.dtype, device=source.device), \
               torch.zeros((0,), dtype=source.dtype, device=source.device)
    rows = nz[:, 0]
    cols = nz[:, 1]
    # Indexing convention from sigma_meshgrid(..., indexing="xy"): rows -> sy, cols -> sx
    sigma_x = s_axis[cols]
    sigma_y = s_axis[rows]
    sigmas = torch.stack([sigma_x, sigma_y], dim=-1)
    weights = source[rows, cols]
    return sigmas, weights

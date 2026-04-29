"""Aerial-image metrics shared across phases.

Kept deliberately small: any metric used by more than one phase lives here,
anything used in a single phase stays in that phase's module.
"""

from __future__ import annotations

import torch


def image_contrast(intensity: torch.Tensor, region: torch.Tensor | None = None) -> torch.Tensor:
    """Michelson contrast ``(I_max - I_min) / (I_max + I_min)`` over the
    full field, or restricted to ``region`` (non-zero pixels) if given.
    """
    if region is None:
        I = intensity
    else:
        I = intensity[region > 0]
        if I.numel() == 0:
            return torch.tensor(0.0, dtype=intensity.dtype, device=intensity.device)
    Imax, Imin = I.max(), I.min()
    return (Imax - Imin) / (Imax + Imin + 1e-12)


def peak_intensity_in_region(intensity: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    if region.sum() < 1:
        return torch.tensor(0.0, dtype=intensity.dtype, device=intensity.device)
    return (intensity * (region > 0).to(intensity.dtype)).max()


def integrated_leakage(intensity: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Sum of intensity inside ``region`` (the higher, the worse the
    suppression). Sums rather than averages so that a wider forbidden
    region carries proportionally more weight."""
    return (intensity * (region > 0).to(intensity.dtype)).sum()


def normalized_image_log_slope(
    intensity: torch.Tensor,
    edge_index_x: int,
    feature_width: float,
    extent: float,
    row_index: int | None = None,
) -> torch.Tensor:
    """Approximate NILS = w * (1/I) * |dI/dx| at the requested edge column.

    A study-grade estimate that uses central finite differences along x.
    ``feature_width`` (in length units) sets the ``w`` scaling. For real
    lithography NILS the window is the resist threshold cross; we pick the
    geometric feature size as a reasonable proxy here.
    """
    n = intensity.shape[-1]
    if row_index is None:
        row_index = n // 2
    # Central difference; clamp to grid bounds.
    i = max(1, min(n - 2, edge_index_x))
    I_at = intensity[row_index, i]
    dI_dx = (intensity[row_index, i + 1] - intensity[row_index, i - 1]) / (2.0 * extent / n)
    return feature_width * dI_dx.abs() / (I_at + 1e-12)

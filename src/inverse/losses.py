"""Region-based losses for inverse aerial optimization.

The Phase-3 loss decomposes the wafer plane into a *target* region (where
the aerial intensity should match a target value, typically 1) and a
*forbidden* / background region (where the intensity should be zero).
Both losses are masked mean-squared errors normalized by the region area.
"""

from __future__ import annotations

import torch


def masked_mse(field: torch.Tensor, region: torch.Tensor, target: float) -> torch.Tensor:
    """Mean-squared error of ``field`` against ``target``, weighted by ``region``.

    ``region`` is a non-negative mask; pixels with weight 0 do not contribute.
    Normalization is by the region's total weight (clamped at 1) so the loss
    has the same scale regardless of region size.
    """
    weight_sum = region.sum().clamp(min=1.0)
    return (((field - target) ** 2) * region).sum() / weight_sum


def target_loss(
    intensity: torch.Tensor,
    target_region: torch.Tensor,
    target_value: float = 1.0,
) -> torch.Tensor:
    """Loss that pushes the aerial intensity toward ``target_value`` inside
    ``target_region``."""
    return masked_mse(intensity, target_region, target_value)


def background_loss(
    intensity: torch.Tensor,
    forbidden_region: torch.Tensor,
) -> torch.Tensor:
    """Loss that suppresses aerial intensity inside ``forbidden_region``."""
    return masked_mse(intensity, forbidden_region, 0.0)


def mean_intensity_in_region(
    intensity: torch.Tensor,
    region: torch.Tensor,
) -> torch.Tensor:
    """Average intensity over the pixels selected by ``region``.

    Useful as a diagnostic signal during optimization; not a loss term.
    """
    weight_sum = region.sum().clamp(min=1.0)
    return (intensity * region).sum() / weight_sum

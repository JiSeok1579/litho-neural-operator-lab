"""Regularizers for the inverse-mask optimization parameter.

Both regularizers operate on the continuous mask ``m`` in [0, 1] (i.e. the
output of ``sigmoid(alpha * theta)``), not on the raw parameter ``theta``.
"""

from __future__ import annotations

import torch


def total_variation(m: torch.Tensor) -> torch.Tensor:
    """L1 total variation of a 2D mask.

    Penalizes large pixel-to-pixel jumps so the mask stays geometrically
    coherent rather than collapsing into a high-frequency speckle. Returns
    the mean absolute first difference along x and y.
    """
    if m.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {tuple(m.shape)}")
    dx = m[:, 1:] - m[:, :-1]
    dy = m[1:, :] - m[:-1, :]
    return dx.abs().mean() + dy.abs().mean()


def binarization_penalty(m: torch.Tensor) -> torch.Tensor:
    """Quadratic penalty on intermediate mask values.

    Equal to ``mean(m * (1 - m))``. The function is zero when ``m`` is
    everywhere 0 or 1 and reaches its maximum 0.25 at ``m = 0.5``. Used
    together with sigmoid sharpening to drive the final mask binary.
    """
    return (m * (1.0 - m)).mean()

"""Coherent imaging through a thin mask and a projection-lens pupil.

The thin-mask approximation treats the mask as a 2D complex transmission
function ``t(x, y)``. The wafer-plane field after the projection lens is

    E(x, y) = F^-1{ T(fx, fy) * P(fx, fy) }

and the aerial image (intensity recorded by the resist) is

    I(x, y) = |E(x, y)|^2

This module is the bridge between Phase 1 (mask / spectrum) and Phase 3
(differentiable mask optimization). Every operation is differentiable in
PyTorch, so a wrapping `loss.backward()` propagates through to the mask
parameter.
"""

from __future__ import annotations

import torch

from src.common.fft_utils import fft2c, ifft2c


def coherent_field(transmission: torch.Tensor, pupil: torch.Tensor) -> torch.Tensor:
    """Pupil-filtered wafer-plane complex field.

    ``transmission`` must already be complex; build it with
    :func:`src.mask.transmission.binary_transmission` or one of the
    phase-shift variants. ``pupil`` is real-valued (binary or apodized).
    """
    if not torch.is_complex(transmission):
        raise TypeError(
            "transmission must be complex; use src.mask.transmission helpers"
        )
    if torch.is_complex(pupil):
        raise TypeError("pupil must be real-valued")
    spectrum = fft2c(transmission)
    return ifft2c(spectrum * pupil)


def coherent_aerial_image(
    transmission: torch.Tensor,
    pupil: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Wafer-plane intensity after coherent imaging through ``pupil``.

    With ``normalize=True`` the intensity is divided by its own maximum, so
    the result is in [0, 1]. For inverse optimization (Phase 3) and for any
    relative-intensity comparison across NAs, use ``normalize=False`` and
    pick the reference yourself.
    """
    field = coherent_field(transmission, pupil)
    intensity = (field.real ** 2 + field.imag ** 2)  # |E|^2 stays in real autograd graph
    if normalize:
        intensity = intensity / (intensity.max() + 1e-12)
    return intensity

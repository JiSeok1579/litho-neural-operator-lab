"""Aerial-intensity to initial acid-concentration mapping (Dill style).

The PEB submodule's exposure model:

    H_0(x, y) = H_max * (1 - exp(-eta * dose * I(x, y)))

with units:

    H_0   : [mol / dm^3]    (peaks at H_max for fully exposed pixels)
    H_max : [mol / dm^3]    (saturating acid density)
    eta   : dimensionless   (acid generation efficiency)
    dose  : dimensionless   (normalized exposure dose)
    I     : in [0, 1]       (normalized aerial intensity)

The function is differentiable in ``I`` so a downstream loss can
still backprop into the exposure map (useful only when this submodule
is later wired into an inverse-design loop; not used in Phase 1).
"""

from __future__ import annotations

import torch


def acid_generation(
    I: torch.Tensor,
    dose: float = 1.0,
    eta: float = 1.0,
    Hmax: float = 0.2,
) -> torch.Tensor:
    """Initial acid concentration ``H_0`` from a normalized aerial map.

    Args:
        I    : 2D intensity tensor with values expected in ``[0, 1]``.
        dose : normalized exposure dose. Must be ``>= 0``.
        eta  : acid generation efficiency. Must be ``>= 0``.
        Hmax : maximum acid concentration in ``mol/dm^3``. Must be
               ``>= 0``.

    Returns:
        ``H_0 = Hmax * (1 - exp(-eta * dose * I))``.
    """
    if dose < 0:
        raise ValueError("dose must be non-negative")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    if Hmax < 0:
        raise ValueError("Hmax must be non-negative")
    return Hmax * (1.0 - torch.exp(-eta * dose * I))

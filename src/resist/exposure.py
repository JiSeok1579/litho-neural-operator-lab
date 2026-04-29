"""Aerial-image to acid-concentration mapping (Dill-style).

Photoresist exposure is modelled as saturating exponential acid generation:

    A0(x, y) = 1 - exp(-eta * dose * I(x, y))

where ``I`` is the (non-negative) aerial intensity, ``dose`` is the
exposure dose, and ``eta`` is the acid generation coefficient. The result
``A0`` is in [0, 1]; ``A0 = 1`` corresponds to fully saturated acid.

The function is differentiable, so an inverse-design loss applied to
``A0`` (or to the post-diffusion latent image) propagates back to the
mask through the optics chain unchanged.
"""

from __future__ import annotations

import torch


def acid_from_aerial(
    aerial: torch.Tensor,
    dose: float,
    eta: float = 1.0,
) -> torch.Tensor:
    """Saturating exponential acid generation.

    For small ``dose * I``, the model linearizes to ``A0 ~ eta * dose * I``
    (Beer-Lambert in the dilute regime). At large dose it saturates at 1.
    """
    if dose < 0:
        raise ValueError("dose must be non-negative")
    if eta < 0:
        raise ValueError("eta must be non-negative")
    return 1.0 - torch.exp(-eta * dose * aerial)

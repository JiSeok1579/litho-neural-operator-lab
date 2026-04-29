"""Complex transmission functions built from binary masks.

A binary mask m(x, y) in {0, 1} is converted to a complex transmission
function t(x, y) = A(x, y) * exp(i * phi(x, y)) appropriate for scalar
Fourier optics.
"""

from __future__ import annotations

import math

import torch


def binary_transmission(mask: torch.Tensor) -> torch.Tensor:
    """Pure binary transmission: open = 1, blocked = 0 (no phase shift)."""
    if torch.is_complex(mask):
        return mask.to(torch.complex64)
    return mask.to(torch.complex64)


def attenuated_phase_shift(
    mask: torch.Tensor,
    attenuation: float = 0.06,
    phase_rad: float = math.pi,
) -> torch.Tensor:
    """Att-PSM transmission.

    Open regions transmit fully (1 + 0j). Blocked regions transmit a small
    fraction sqrt(attenuation) with a phase shift of ``phase_rad`` radians
    (default pi). This corresponds to a 6% half-tone phase-shift mask when
    ``attenuation = 0.06``.
    """
    if not (0.0 <= attenuation <= 1.0):
        raise ValueError("attenuation must be in [0, 1]")
    open_value = torch.complex(torch.ones_like(mask), torch.zeros_like(mask))
    amp = math.sqrt(attenuation)
    abs_value = torch.complex(
        torch.full_like(mask, amp * math.cos(phase_rad)),
        torch.full_like(mask, amp * math.sin(phase_rad)),
    )
    is_open = (mask > 0.5).to(torch.bool)
    return torch.where(is_open, open_value, abs_value).to(torch.complex64)


def alternating_phase_shift(
    mask: torch.Tensor,
    region_a: torch.Tensor,
    region_b: torch.Tensor,
) -> torch.Tensor:
    """Alt-PSM transmission.

    ``region_a`` keeps phase 0; ``region_b`` flips by pi. Both regions must
    be subsets of the open mask. Blocked regions stay at 0.
    """
    zero = torch.zeros_like(mask)
    is_open = (mask > 0.5).to(torch.bool)
    is_a = is_open & (region_a > 0.5)
    is_b = is_open & (region_b > 0.5)
    out = torch.complex(zero.clone(), zero.clone())
    out = torch.where(is_a, torch.complex(torch.ones_like(mask), zero), out)
    out = torch.where(is_b, torch.complex(-torch.ones_like(mask), zero), out)
    return out.to(torch.complex64)

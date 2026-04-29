"""End-to-end Phase-1 sanity checks for diffraction spectra."""

from __future__ import annotations

import torch

from src.common.grid import Grid2D
from src.mask import patterns, transmission
from src.optics.scalar_diffraction import diffraction_spectrum, reconstruct_field


def test_real_mask_spectrum_is_hermitian():
    """For a real-valued transmission, the centered spectrum is Hermitian:
    T(-f) = conj(T(f)). Check by flipping both axes about the DC bin.
    """
    g = Grid2D(n=64, extent=1.0)
    m = patterns.contact_hole(g, radius=0.1)
    t = transmission.binary_transmission(m)
    T = diffraction_spectrum(t)
    # Drop the DC row/col (index n/2) so flip is exact about the center.
    n = T.shape[-1]
    c = n // 2
    sub = T[1:, 1:]  # indices 1..n-1, symmetric about center c-1
    flipped = torch.flip(sub, dims=(-2, -1)).conj()
    rel = (sub - flipped).abs().max().item() / (sub.abs().max().item() + 1e-12)
    assert rel < 1e-4, f"Hermitian symmetry broken: rel max diff {rel}"


def test_diffraction_round_trip():
    g = Grid2D(n=64, extent=1.0)
    m = patterns.line_space(g, pitch=0.25, duty=0.5)
    t = transmission.binary_transmission(m)
    T = diffraction_spectrum(t)
    t_rec = reconstruct_field(T)
    err = (t - t_rec).abs().max().item()
    assert err < 1e-4


def test_pitch_changes_diffraction_order_spacing():
    """Halving the line-space pitch doubles the spacing between diffraction
    orders along the perpendicular axis. We check that the second-strongest
    nonzero frequency is roughly 2x further from DC when pitch is halved.
    """
    g = Grid2D(n=128, extent=1.0)

    def first_order_offset(pitch: float) -> float:
        m = patterns.line_space(g, pitch=pitch, duty=0.5, orientation="vertical")
        T = diffraction_spectrum(transmission.binary_transmission(m))
        amp = T.abs()
        # Look only along the central row (fy = 0): orders sit at integer
        # multiples of 1/pitch along fx.
        center_row = amp[amp.shape[-2] // 2]
        # Suppress DC
        center_row = center_row.clone()
        center_row[center_row.shape[-1] // 2] = 0.0
        peak_idx = int(torch.argmax(center_row).item())
        # Convert pixel offset from center into a frequency value
        return abs(peak_idx - center_row.shape[-1] // 2) * g.df

    f_a = first_order_offset(0.20)
    f_b = first_order_offset(0.10)
    # Smaller pitch -> larger first-order frequency
    assert f_b > f_a
    # Should be roughly 2x (allow 25% tolerance for DFT discretization)
    ratio = f_b / f_a
    assert 1.5 < ratio < 2.5, f"order-spacing ratio off: {ratio}"

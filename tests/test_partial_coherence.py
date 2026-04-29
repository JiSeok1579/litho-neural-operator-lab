"""Tests for :mod:`src.optics.partial_coherence`."""

from __future__ import annotations

import torch

from src.common.grid import Grid2D
from src.mask import patterns, transmission
from src.optics.coherent_imaging import coherent_aerial_image
from src.optics.partial_coherence import partial_coherent_aerial_image
from src.optics.pupil import circular_pupil
from src.optics.source import (
    annular_source,
    coherent_source,
    dipole_source,
)


def _grid() -> Grid2D:
    return Grid2D(n=128, extent=10.0)


def test_coherent_source_matches_coherent_imaging():
    """A delta source at the origin must reproduce the on-axis coherent
    aerial image exactly (up to floating-point round-off)."""
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    p = circular_pupil(g, NA=0.5)
    I_ref = coherent_aerial_image(t, p, normalize=False)

    src = coherent_source(31)
    I_pc = partial_coherent_aerial_image(t, g, src, NA=0.5, normalize=False)

    err = (I_pc - I_ref).abs().max().item()
    rel = err / (I_ref.max().item() + 1e-12)
    assert rel < 1e-4, f"PC with delta source diverged from coherent: rel {rel}"


def test_partial_coherent_real_nonneg():
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    src = annular_source(31, sigma_inner=0.3, sigma_outer=0.7)
    I = partial_coherent_aerial_image(t, g, src, NA=0.5, normalize=True)
    assert I.dtype == g.dtype
    assert (I >= 0).all().item()


def test_normalize_makes_scale_independent():
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)

    src1 = annular_source(31, sigma_inner=0.3, sigma_outer=0.7)
    src2 = src1 * 5.0  # same pattern, 5x weight

    I1 = partial_coherent_aerial_image(t, g, src1, NA=0.5, normalize=True)
    I2 = partial_coherent_aerial_image(t, g, src2, NA=0.5, normalize=True)
    err = (I1 - I2).abs().max().item() / (I1.max().item() + 1e-12)
    assert err < 1e-6, "normalize should remove source-weight scale"


def test_dipole_x_helps_vertical_line_space():
    """For vertical line-space at a pitch where on-axis fails to resolve,
    a dipole-x source can shift the diffraction order into the pupil and
    recover contrast. The x-dipole vs y-dipole comparison isolates the
    orientation effect."""
    g = Grid2D(n=256, extent=20.0)
    # Pitch 1.5 lambda -> fundamental at fx = 0.667 cycles/lambda.
    # At NA=0.4, on-axis fails (cutoff 0.4 < 0.667).
    # Dipole-x at sigma=0.7 places the pupil center at fx ~ 0.28; the
    # right pole's pupil reaches fx = 0.28 + 0.4 = 0.68 -> just captures
    # the +1 order.
    m = patterns.line_space(g, pitch=1.5, duty=0.5, orientation="vertical")
    t = transmission.binary_transmission(m)

    src_dx = dipole_source(31, sigma_center=0.7, sigma_radius=0.15, axis="x")
    src_dy = dipole_source(31, sigma_center=0.7, sigma_radius=0.15, axis="y")

    Idx = partial_coherent_aerial_image(t, g, src_dx, NA=0.4, normalize=True)
    Idy = partial_coherent_aerial_image(t, g, src_dy, NA=0.4, normalize=True)

    rel_std_dx = (Idx.std() / (Idx.mean() + 1e-12)).item()
    rel_std_dy = (Idy.std() / (Idy.mean() + 1e-12)).item()
    assert rel_std_dx > 5 * rel_std_dy, (
        f"dipole-x should produce far more contrast than dipole-y for vertical "
        f"lines, got dx={rel_std_dx} dy={rel_std_dy}"
    )


def test_partial_coherence_autograd():
    g = _grid()
    raw = torch.zeros((g.n, g.n), dtype=g.dtype, requires_grad=True)
    t = torch.sigmoid(raw + 1.0).to(torch.complex64)
    src = annular_source(31, sigma_inner=0.3, sigma_outer=0.7)
    I = partial_coherent_aerial_image(t, g, src, NA=0.5, normalize=False)
    loss = I.sum()
    loss.backward()
    assert raw.grad is not None
    assert raw.grad.abs().max().item() > 0


def test_empty_source_returns_zero():
    g = _grid()
    m = patterns.contact_hole(g, radius=1.0)
    t = transmission.binary_transmission(m)
    src = torch.zeros((31, 31))
    I = partial_coherent_aerial_image(t, g, src, NA=0.5, normalize=False)
    assert torch.all(I == 0)

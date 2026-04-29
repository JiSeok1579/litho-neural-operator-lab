"""Sanity tests for :mod:`src.optics.pupil`."""

from __future__ import annotations

import math

import torch

from src.common.grid import Grid2D
from src.optics.pupil import apodized_circular_pupil, circular_pupil


def _grid() -> Grid2D:
    return Grid2D(n=128, extent=10.0)


def test_pupil_radial_symmetry():
    g = _grid()
    p = circular_pupil(g, NA=0.5)
    # The pupil is rotationally symmetric, so a 90-degree rotation
    # (transpose for square images) leaves it unchanged.
    assert torch.equal(p, p.T)


def test_pupil_dc_passes():
    """Even an arbitrarily small NA must let the DC bin through."""
    g = _grid()
    p = circular_pupil(g, NA=0.01)
    c = g.n // 2
    assert p[c, c].item() == 1.0


def test_pupil_area_scales_with_NA_squared():
    """For a circular pupil, open area scales as NA^2."""
    g = Grid2D(n=256, extent=20.0)
    a_small = circular_pupil(g, NA=0.2).sum().item()
    a_large = circular_pupil(g, NA=0.8).sum().item()
    measured = a_large / a_small
    expected = (0.8 / 0.2) ** 2  # 16
    assert abs(measured - expected) / expected < 0.10


def test_pupil_wavelength_scales_cutoff():
    """Doubling lambda halves the cutoff frequency, which roughly quarters
    the open area."""
    g = Grid2D(n=256, extent=20.0)
    a1 = circular_pupil(g, NA=0.6, wavelength=1.0).sum().item()
    a2 = circular_pupil(g, NA=0.6, wavelength=2.0).sum().item()
    ratio = a1 / a2
    assert 3.5 < ratio < 4.5  # ~4x, allow DFT discretization slack


def test_pupil_invalid_args():
    g = _grid()
    for bad in (0.0, -0.1):
        try:
            circular_pupil(g, NA=bad)
        except ValueError:
            pass
        else:
            assert False, f"NA={bad} should raise"
        try:
            circular_pupil(g, NA=0.5, wavelength=bad)
        except ValueError:
            pass
        else:
            assert False, f"wavelength={bad} should raise"


def test_apodized_zero_rolloff_matches_hard():
    g = _grid()
    p_hard = circular_pupil(g, NA=0.5)
    p_apo = apodized_circular_pupil(g, NA=0.5, roll_off=0.0)
    assert torch.allclose(p_hard, p_apo, atol=1e-6)


def test_apodized_smooth_taper_present():
    g = _grid()
    p = apodized_circular_pupil(g, NA=0.5, roll_off=0.2)
    # The taper region must contain values strictly between 0 and 1.
    has_taper = ((p > 0.01) & (p < 0.99)).any().item()
    assert has_taper
    assert (p >= 0).all() and (p <= 1).all()


def test_apodized_invalid_rolloff():
    g = _grid()
    for bad in (-0.1, 1.5):
        try:
            apodized_circular_pupil(g, NA=0.5, roll_off=bad)
        except ValueError:
            pass
        else:
            assert False, f"roll_off={bad} should raise"

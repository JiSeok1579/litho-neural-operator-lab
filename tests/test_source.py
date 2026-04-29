"""Tests for :mod:`src.optics.source`."""

from __future__ import annotations

import torch

from src.optics.source import (
    annular_source,
    coherent_source,
    dipole_source,
    quadrupole_source,
    random_source,
    sigma_axis,
    sigma_meshgrid,
    source_points,
)


def test_sigma_axis_endpoints():
    s = sigma_axis(31)
    assert abs(s[0].item() + 1.0) < 1e-6
    assert abs(s[-1].item() - 1.0) < 1e-6
    assert abs(s[15].item()) < 1e-6


def test_coherent_single_pixel():
    src = coherent_source(31)
    assert src.sum().item() == 1.0
    assert src[15, 15].item() == 1.0


def test_coherent_rejects_even():
    try:
        coherent_source(30)
    except ValueError:
        pass
    else:
        assert False, "even n_sigma should raise"


def test_annular_inner_outer():
    src = annular_source(101, sigma_inner=0.3, sigma_outer=0.7)
    sx, sy = sigma_meshgrid(101)
    sr = torch.sqrt(sx ** 2 + sy ** 2)
    inside_inner = src[(sr < 0.3 - 0.05)]
    outside_outer = src[(sr > 0.7 + 0.05)]
    assert (inside_inner == 0).all()
    assert (outside_outer == 0).all()
    in_ring = src[(sr > 0.4) & (sr < 0.6)]
    assert (in_ring == 1).all()


def test_dipole_x_centers():
    src = dipole_source(101, sigma_center=0.5, sigma_radius=0.1, axis="x")
    # The two center pixels should be inside the poles
    sx, sy = sigma_meshgrid(101)
    plus_pole = src[((sx - 0.5).abs() < 0.05) & (sy.abs() < 0.05)]
    minus_pole = src[((sx + 0.5).abs() < 0.05) & (sy.abs() < 0.05)]
    assert plus_pole.max().item() == 1.0
    assert minus_pole.max().item() == 1.0
    # No mass at +/-y poles
    y_pole = src[(sx.abs() < 0.05) & ((sy - 0.5).abs() < 0.05)]
    assert y_pole.max().item() == 0.0


def test_dipole_axis_swap_is_transpose():
    a = dipole_source(31, sigma_center=0.5, sigma_radius=0.15, axis="x")
    b = dipole_source(31, sigma_center=0.5, sigma_radius=0.15, axis="y")
    assert torch.equal(a, b.T)


def test_quadrupole_four_lobes():
    src = quadrupole_source(101, sigma_center=0.5, sigma_radius=0.1)
    # Four mass concentrations expected; check each
    sx, sy = sigma_meshgrid(101)
    for cx, cy in [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5)]:
        pole = src[((sx - cx).abs() < 0.05) & ((sy - cy).abs() < 0.05)]
        assert pole.max().item() == 1.0


def test_random_source_respects_sigma_max():
    src = random_source(51, fill_fraction=0.2, sigma_max=0.5, seed=0)
    sx, sy = sigma_meshgrid(51)
    sr = torch.sqrt(sx ** 2 + sy ** 2)
    # Anything outside sigma_max must be zero
    outside = src[sr > 0.5]
    assert (outside == 0).all()


def test_source_points_decoding():
    src = dipole_source(31, sigma_center=0.5, sigma_radius=0.05, axis="x")
    sigmas, weights = source_points(src)
    assert sigmas.shape[1] == 2
    assert sigmas.shape[0] == weights.shape[0]
    # The source has equal weights (binary), so all weights are 1
    assert torch.all(weights == 1.0)
    # x positions are clustered around +/-0.5; y around 0
    sx_set = sigmas[:, 0]
    sy_set = sigmas[:, 1]
    assert (sx_set.abs() > 0.3).all()
    assert (sy_set.abs() < 0.1).all()


def test_source_points_empty_source():
    src = torch.zeros((31, 31))
    sigmas, weights = source_points(src)
    assert sigmas.shape == (0, 2)
    assert weights.shape == (0,)

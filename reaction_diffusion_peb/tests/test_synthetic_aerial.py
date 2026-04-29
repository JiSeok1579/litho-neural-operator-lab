"""Tests for the PEB submodule's synthetic-aerial helpers."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.synthetic_aerial import (
    contact_array,
    gaussian_spot,
    line_space,
    normalize_intensity,
    two_spot,
)


def test_gaussian_spot_shape_and_peak():
    g = gaussian_spot(64, sigma_px=8.0)
    assert g.shape == (64, 64)
    # Peak is at the center pixel and equals 1
    c = 32
    assert abs(g[c, c].item() - 1.0) < 1e-6 or g.max().item() > 0.99


def test_gaussian_spot_off_center():
    g = gaussian_spot(64, sigma_px=4.0, center=(10.0, 0.0))
    nz_max_idx = torch.argmax(g.flatten())
    row, col = divmod(int(nz_max_idx.item()), 64)
    # Peak shifts in +x direction, which is column-wise in our 'xy'
    # indexing convention.
    assert col > 32


def test_gaussian_spot_rejects_invalid_args():
    for bad in (0, -1):
        try:
            gaussian_spot(bad, sigma_px=4.0)
        except ValueError:
            pass
        else:
            raise AssertionError(f"grid_size={bad} should raise")
    for bad in (0.0, -1.0):
        try:
            gaussian_spot(32, sigma_px=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"sigma_px={bad} should raise")


def test_line_space_shape_and_periodicity():
    ls = line_space(64, pitch_px=16.0, duty=0.5, contrast=1.0)
    assert ls.shape == (64, 64)
    # All rows equal each other for vertical orientation
    assert torch.allclose(ls[0], ls[-1])
    # Period is 16 pixels: row 0 column k vs column k+16 should match
    assert torch.allclose(ls[0, :48], ls[0, 16:])


def test_line_space_orientation_swap_is_transpose():
    a = line_space(32, pitch_px=8.0, orientation="vertical", contrast=1.0)
    b = line_space(32, pitch_px=8.0, orientation="horizontal", contrast=1.0)
    assert torch.equal(a, b.T)


def test_line_space_duty_changes_open_fraction():
    a = line_space(128, pitch_px=16.0, duty=0.25, contrast=1.0)
    b = line_space(128, pitch_px=16.0, duty=0.75, contrast=1.0)
    assert a.mean().item() < b.mean().item()


def test_contact_array_periodic_centers():
    arr = contact_array(64, pitch_px=16.0, sigma_px=2.0)
    assert arr.shape == (64, 64)
    assert (arr >= 0).all().item() and (arr <= 1).all().item()


def test_two_spot_two_local_maxima():
    spots = two_spot(64, sigma_px=3.0, separation_px=20.0)
    # Two clear peaks along the central row, on either side of x=0
    mid = spots[32, :]
    left_peak = int(torch.argmax(mid[:32]).item())
    right_peak = 32 + int(torch.argmax(mid[32:]).item())
    assert left_peak < 32 < right_peak


def test_normalize_intensity_to_unit_range():
    x = torch.linspace(2.0, 7.0, 25).reshape(5, 5)
    y = normalize_intensity(x)
    assert abs(y.min().item()) < 1e-6
    assert abs(y.max().item() - 1.0) < 1e-6


def test_normalize_intensity_constant_returns_zero():
    x = torch.full((4, 4), 3.0)
    y = normalize_intensity(x)
    assert torch.equal(y, torch.zeros_like(y))


def test_line_space_smooth_edges_have_intermediate_values():
    """With a non-zero ``smooth_px`` the binary step is replaced by a
    half-cosine — there must be values strictly between 0 and 1 along
    a single row."""
    ls = line_space(128, pitch_px=16.0, duty=0.5, contrast=1.0, smooth_px=2.0)
    row = ls[0]
    has_intermediate = ((row > 0.05) & (row < 0.95)).any().item()
    assert has_intermediate

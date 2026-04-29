"""Tests for :mod:`src.common.metrics`."""

from __future__ import annotations

import torch

from src.common.metrics import (
    image_contrast,
    integrated_leakage,
    peak_intensity_in_region,
)


def test_contrast_uniform_field_is_zero():
    I = torch.full((8, 8), 0.5)
    assert image_contrast(I).item() < 1e-6


def test_contrast_full_swing_is_one():
    I = torch.zeros((8, 8))
    I[0, 0] = 1.0
    assert abs(image_contrast(I).item() - 1.0) < 1e-6


def test_contrast_with_region():
    I = torch.zeros((4, 4))
    I[0, :] = 1.0
    region = torch.zeros((4, 4))
    region[2:, :] = 1.0  # selects only rows 2-3 where I = 0
    # Within the selected region the field is uniform, so contrast = 0.
    assert image_contrast(I, region=region).item() < 1e-6


def test_peak_intensity_in_region():
    I = torch.tensor([[0.5, 0.9], [0.1, 0.0]])
    region = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    # Considered values: 0.5, 0.1 -> peak 0.5
    assert abs(peak_intensity_in_region(I, region).item() - 0.5) < 1e-6


def test_integrated_leakage():
    I = torch.full((4, 4), 0.25)
    region = torch.tensor([[1.0, 1.0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    # Sum over the 4 selected pixels, each at 0.25 -> 1.0
    assert abs(integrated_leakage(I, region).item() - 1.0) < 1e-6

"""Tests for :mod:`src.inverse.losses` and :mod:`src.inverse.regularizers`."""

from __future__ import annotations

import torch

from src.inverse.losses import (
    background_loss,
    masked_mse,
    mean_intensity_in_region,
    target_loss,
)
from src.inverse.regularizers import binarization_penalty, total_variation


def test_masked_mse_zero_when_match():
    f = torch.full((8, 8), 0.5)
    region = torch.ones((8, 8))
    assert masked_mse(f, region, target=0.5).item() < 1e-12


def test_masked_mse_outside_region_ignored():
    """Pixels with weight 0 must not contribute to the loss."""
    f = torch.zeros((4, 4))
    f[0, 0] = 1.0  # this pixel is outside the region
    region = torch.zeros((4, 4))
    region[1, 1] = 1.0  # only this pixel counts; field there is 0, target 0
    loss = masked_mse(f, region, target=0.0).item()
    assert loss < 1e-12


def test_target_loss_alias():
    f = torch.full((4, 4), 0.8)
    region = torch.ones((4, 4))
    assert abs(target_loss(f, region, target_value=1.0).item() - 0.04) < 1e-6


def test_background_loss_zero_when_dark():
    f = torch.zeros((4, 4))
    region = torch.ones((4, 4))
    assert background_loss(f, region).item() < 1e-12


def test_mean_intensity():
    f = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    region = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    # average of (0, 1) = 0.5
    assert abs(mean_intensity_in_region(f, region).item() - 0.5) < 1e-6


def test_total_variation_constant_field():
    m = torch.full((16, 16), 0.5)
    assert total_variation(m).item() < 1e-12


def test_total_variation_nonzero_for_step():
    m = torch.zeros((16, 16))
    m[:, 8:] = 1.0  # vertical step in the middle
    assert total_variation(m).item() > 0


def test_binarization_zero_for_binary():
    m = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert binarization_penalty(m).item() < 1e-12


def test_binarization_max_at_half():
    m = torch.full((4, 4), 0.5)
    # m * (1 - m) = 0.25 everywhere
    assert abs(binarization_penalty(m).item() - 0.25) < 1e-6


def test_total_variation_rejects_non_2d():
    m = torch.zeros((4, 4, 4))
    try:
        total_variation(m)
    except ValueError:
        pass
    else:
        assert False, "total_variation should reject non-2D inputs"


def test_losses_differentiable():
    """All region losses + regularizers must propagate gradients."""
    field = torch.zeros((8, 8), requires_grad=True)
    region = torch.ones((8, 8))
    loss = masked_mse(field, region, target=1.0) + total_variation(field) + binarization_penalty(field)
    loss.backward()
    assert field.grad is not None
    assert field.grad.abs().max().item() > 0

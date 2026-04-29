"""Tests for the PEB submodule's exposure model."""

from __future__ import annotations

import torch

from reaction_diffusion_peb.src.exposure import acid_generation


def test_zero_intensity_gives_zero_acid():
    I = torch.zeros(8, 8)
    H0 = acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2)
    assert torch.allclose(H0, torch.zeros_like(H0))


def test_zero_dose_gives_zero_acid():
    I = torch.rand(8, 8)
    H0 = acid_generation(I, dose=0.0, eta=1.0, Hmax=0.2)
    assert torch.allclose(H0, torch.zeros_like(H0))


def test_acid_does_not_exceed_hmax():
    I = torch.full((4, 4), 1.0)
    H0 = acid_generation(I, dose=20.0, eta=1.0, Hmax=0.2)
    assert (H0 <= 0.2 + 1e-6).all()


def test_acid_saturates_at_hmax_for_large_dose():
    I = torch.full((4, 4), 1.0)
    H0 = acid_generation(I, dose=20.0, eta=1.0, Hmax=0.2)
    # 1 - exp(-20) ~ 1 -> H0 should be ~Hmax
    assert (H0 > 0.999 * 0.2).all()


def test_acid_monotonic_in_dose():
    I = torch.rand(16, 16)
    H_low = acid_generation(I, dose=0.5, eta=1.0, Hmax=0.2)
    H_high = acid_generation(I, dose=2.0, eta=1.0, Hmax=0.2)
    assert (H_high >= H_low - 1e-6).all()


def test_acid_monotonic_in_eta():
    I = torch.rand(16, 16)
    H_low = acid_generation(I, dose=1.0, eta=0.5, Hmax=0.2)
    H_high = acid_generation(I, dose=1.0, eta=2.0, Hmax=0.2)
    assert (H_high >= H_low - 1e-6).all()


def test_acid_scales_with_hmax():
    I = torch.rand(16, 16)
    Ha = acid_generation(I, dose=1.0, eta=1.0, Hmax=0.1)
    Hb = acid_generation(I, dose=1.0, eta=1.0, Hmax=0.3)
    # Both saturate to their respective Hmax; the ratio everywhere is
    # the Hmax ratio (3:1) because the (1 - exp(...)) factor cancels.
    ratio = (Hb / (Ha + 1e-12)).mean().item()
    assert 2.9 < ratio < 3.1


def test_acid_generation_differentiable():
    I = torch.rand(8, 8, requires_grad=True)
    H0 = acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2)
    H0.sum().backward()
    assert I.grad is not None
    assert I.grad.abs().max().item() > 0


def test_acid_generation_invalid_args():
    I = torch.rand(4, 4)
    for bad in (-0.1,):
        for kw in ("dose", "eta", "Hmax"):
            kwargs = {"dose": 1.0, "eta": 1.0, "Hmax": 0.2}
            kwargs[kw] = bad
            try:
                acid_generation(I, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"{kw}={bad} should raise")

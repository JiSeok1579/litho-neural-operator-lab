"""Tests for the Phase-4 PINN with acid-loss term."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.pinn_diffusion import (
    gaussian_spot_acid_callable,
)
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDiffusionLoss


def _grid_kwargs():
    return dict(x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
                hidden=8, n_hidden_layers=1, n_fourier=4)


def test_loss_pinn_hard_ic_at_t_zero():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusionLoss(
        DH_nm2_s=0.8, kloss_s_inv=0.005,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = torch.linspace(-30, 30, 7)
    y = torch.zeros_like(x); t = torch.zeros_like(x)
    H_pred = pinn(x, y, t)
    H_truth = H0_fn(x, y)
    assert torch.allclose(H_pred, H_truth, atol=1e-6)


def test_loss_pinn_residual_runs():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusionLoss(
        DH_nm2_s=0.8, kloss_s_inv=0.005,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = (torch.rand(8) - 0.5) * 60.0
    y = (torch.rand(8) - 0.5) * 60.0
    t = torch.rand(8) * 60.0
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    r = pinn.pde_residual(x, y, t)
    assert r.shape == (8,)


def test_loss_pinn_residual_zero_kloss_matches_diffusion_only():
    """With kloss=0 the residual should equal the bare diffusion residual.

    We construct two PINNs with the same Fourier seed / weights (same
    init), call both residuals on the same points, and check equality.
    """
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)

    pinn_with_loss = PINNDiffusionLoss(
        DH_nm2_s=0.8, kloss_s_inv=0.0,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(), seed=42,
    )
    # Cross-check by also computing the bare diffusion residual using
    # PINNDiffusion (parent class) on the very same network state.
    from reaction_diffusion_peb.src.pinn_diffusion import PINNDiffusion
    pinn_bare = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        **_grid_kwargs(), seed=42,
    )
    pinn_bare.load_state_dict(pinn_with_loss.state_dict())

    x = (torch.rand(8) - 0.5) * 60.0
    y = (torch.rand(8) - 0.5) * 60.0
    t = torch.rand(8) * 60.0
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    r_loss = pinn_with_loss.pde_residual(x, y, t)
    r_bare = pinn_bare.pde_residual(x.detach().requires_grad_(True),
                                     y.detach().requires_grad_(True),
                                     t.detach().requires_grad_(True))
    # Both should be the same since kloss=0 makes the extra term vanish.
    assert torch.allclose(r_loss, r_bare, atol=1e-5)


def test_loss_pinn_invalid_kloss():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    try:
        PINNDiffusionLoss(
            DH_nm2_s=0.8, kloss_s_inv=-0.1,
            hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative kloss should raise")

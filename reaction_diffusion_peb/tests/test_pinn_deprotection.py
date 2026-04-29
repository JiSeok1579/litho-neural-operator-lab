"""Tests for the Phase-5 PINN ``PINNDeprotection``."""

from __future__ import annotations

import torch

from reaction_diffusion_peb.src.pinn_diffusion import gaussian_spot_acid_callable
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection


def _grid_kwargs():
    return dict(x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
                hidden=8, n_hidden_layers=1, n_fourier=4)


def test_deprotection_pinn_forward_shape():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = torch.randn(8) * 30.0
    y = torch.randn(8) * 30.0
    t = torch.rand(8) * 60.0
    H_pred, P_pred = pinn(x, y, t)
    assert H_pred.shape == (8,)
    assert P_pred.shape == (8,)


def test_hard_ic_enforces_initial_conditions():
    """At t=0 the hard-IC parameterization gives H = H_0 and P = 0
    exactly, regardless of weights."""
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0, Hmax=0.2)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = torch.linspace(-30, 30, 7)
    y = torch.zeros_like(x); t = torch.zeros_like(x)
    H_pred, P_pred = pinn(x, y, t)
    H_truth = H0_fn(x, y)
    assert torch.allclose(H_pred, H_truth, atol=1e-6)
    assert torch.allclose(P_pred, torch.zeros_like(P_pred), atol=1e-6)


def test_residuals_run_with_grad_inputs():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = (torch.rand(8) - 0.5) * 60.0
    y = (torch.rand(8) - 0.5) * 60.0
    t = torch.rand(8) * 60.0
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    r_H, r_P = pinn.pde_residuals(x, y, t)
    assert r_H.shape == (8,)
    assert r_P.shape == (8,)


def test_residuals_require_grad_inputs():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDeprotection(
        DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
    )
    x = torch.zeros(4); y = torch.zeros(4); t = torch.zeros(4)
    try:
        pinn.pde_residuals(x, y, t)
    except RuntimeError:
        pass
    else:
        raise AssertionError("residuals should require grad-tracked inputs")


def test_invalid_kdep():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    try:
        PINNDeprotection(
            DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=-0.5,
            hard_ic=True, H0_callable=H0_fn, **_grid_kwargs(),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("negative kdep should raise")

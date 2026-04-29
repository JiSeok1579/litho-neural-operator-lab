"""Tests for the PEB submodule's PINN diffusion model + trainer."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.pinn_base import (
    FourierFeatures,
    MLP,
    PINNBase,
)
from reaction_diffusion_peb.src.pinn_diffusion import (
    PINNDiffusion,
    gaussian_spot_acid_callable,
    pinn_to_grid,
    train_pinn_diffusion,
)


def test_fourier_features_shape():
    f = FourierFeatures(in_dim=3, num_features=8)
    x = torch.randn(50, 3)
    y = f(x)
    assert y.shape == (50, 16)


def test_mlp_layer_count():
    m = MLP(in_dim=4, out_dim=1, hidden=16, n_hidden_layers=3)
    linears = [m for m in m.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4


def test_pinn_base_forward_shape():
    pinn = PINNBase(
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=16, n_hidden_layers=2, n_fourier=4,
    )
    x = torch.randn(10) * 30.0
    y = torch.randn(10) * 30.0
    t = torch.rand(10) * 60.0
    out = pinn(x, y, t)
    assert out.shape == (10,)


def test_pinn_base_shape_mismatch_raises():
    pinn = PINNBase(
        x_range_nm=(-1.0, 1.0), t_range_s=(0.0, 1.0),
        hidden=8, n_hidden_layers=1, n_fourier=4,
    )
    try:
        pinn(torch.randn(10), torch.randn(11), torch.randn(10))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch should raise")


def test_gaussian_spot_acid_callable_at_origin():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0, Hmax=0.2, eta=1.0, dose=1.0)
    H0 = H0_fn(torch.tensor([0.0]), torch.tensor([0.0]))
    # At origin, I=1, so H0 = Hmax * (1 - exp(-1)) ~= 0.2 * 0.632 = 0.1264
    assert abs(H0.item() - 0.2 * (1 - math.exp(-1.0))) < 1e-6


def test_gaussian_spot_acid_callable_far_away():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0, Hmax=0.2)
    H0 = H0_fn(torch.tensor([100.0]), torch.tensor([100.0]))
    # I ~ 0 -> H0 ~ 0
    assert abs(H0.item()) < 1e-6


def test_pinn_diffusion_hard_ic_at_t_zero():
    """With hard_ic=True the PINN output at t=t_low must equal H0
    exactly, regardless of weights."""
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0, Hmax=0.2)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=8, n_hidden_layers=1, n_fourier=4,
    )
    x = torch.linspace(-30, 30, 7)
    y = torch.zeros_like(x)
    t = torch.zeros_like(x)
    H_pred = pinn(x, y, t)
    H_truth = H0_fn(x, y)
    assert torch.allclose(H_pred, H_truth, atol=1e-6)


def test_pinn_diffusion_residual_runs():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=8, n_hidden_layers=1, n_fourier=4,
    )
    x = (torch.rand(8) - 0.5) * 60.0
    y = (torch.rand(8) - 0.5) * 60.0
    t = torch.rand(8) * 60.0
    x.requires_grad_(True); y.requires_grad_(True); t.requires_grad_(True)
    r = pinn.pde_residual(x, y, t)
    assert r.shape == (8,)


def test_pinn_diffusion_residual_requires_grad_inputs():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=8, n_hidden_layers=1, n_fourier=4,
    )
    x = torch.zeros(8); y = torch.zeros(8); t = torch.zeros(8)
    try:
        pinn.pde_residual(x, y, t)
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing requires_grad should raise")


def test_pinn_to_grid_shape():
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=8, n_hidden_layers=1, n_fourier=4,
    )
    out = pinn_to_grid(pinn, grid_size=16, dx_nm=8.0, t_s=30.0)
    assert out.shape == (16, 16)


def test_pinn_to_grid_at_t_zero_matches_h0():
    """At t=0 the grid evaluation should equal the analytic H0 grid."""
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0, Hmax=0.2)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=16, n_hidden_layers=2, n_fourier=4,
    )
    grid_size = 16; dx = 8.0
    coord = (torch.arange(grid_size, dtype=torch.float32) - grid_size / 2.0) * dx
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    H0_truth = H0_fn(X.flatten(), Y.flatten()).view(grid_size, grid_size)
    H_pred = pinn_to_grid(pinn, grid_size=grid_size, dx_nm=dx, t_s=0.0)
    assert torch.allclose(H_pred, H0_truth, atol=1e-6)


def test_short_training_run_finite_loss():
    """Smoke test: 30 iters on a small PINN, loss must stay finite."""
    H0_fn = gaussian_spot_acid_callable(sigma_nm=10.0)
    pinn = PINNDiffusion(
        DH_nm2_s=0.8, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        hidden=16, n_hidden_layers=2, n_fourier=8,
    )
    h = train_pinn_diffusion(
        pinn, H0_callable=H0_fn,
        x_range_nm=(-64.0, 64.0), t_range_s=(0.0, 60.0),
        n_iters=30, lr=1e-3, n_collocation=128, n_ic=64, n_ic_grid_side=4,
        weight_pde=1.0, weight_ic=0.0,
        log_every=10,
    )
    assert math.isfinite(h.loss_total[-1])
    assert h.loss_total[-1] < h.loss_total[0] + 1.0

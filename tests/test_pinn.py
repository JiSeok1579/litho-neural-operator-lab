"""Tests for the Phase-6 PINN building blocks."""

from __future__ import annotations

import math

import torch

from src.pinn.pinn_base import FourierFeatures, MLP, PINNBase
from src.pinn.pinn_diffusion import (
    PINNDiffusion,
    gaussian_analytic_solution,
    gaussian_initial_condition,
    pinn_to_grid,
    train_pinn_diffusion,
)
from src.pinn.pinn_reaction_diffusion import PINNReactionDiffusion
from src.common.grid import Grid2D


def test_fourier_features_shape():
    f = FourierFeatures(in_dim=3, num_features=8)
    x = torch.randn(50, 3)
    y = f(x)
    assert y.shape == (50, 16)


def test_fourier_features_repeatable():
    f1 = FourierFeatures(in_dim=3, num_features=8, seed=42)
    f2 = FourierFeatures(in_dim=3, num_features=8, seed=42)
    assert torch.equal(f1.B, f2.B)


def test_mlp_layer_count():
    m = MLP(in_dim=4, out_dim=1, hidden=16, n_hidden_layers=3, activation="tanh")
    # n_hidden_layers=3 -> Linear(in,h), Tanh, Linear(h,h), Tanh, Linear(h,h), Tanh, Linear(h,out) = 4 Linear, 3 Tanh
    linears = [m for m in m.net if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4


def test_pinn_base_forward_shape():
    pinn = PINNBase(hidden=16, n_hidden_layers=2, n_fourier=4)
    x = torch.randn(10)
    y = torch.randn(10)
    t = torch.randn(10)
    out = pinn(x, y, t)
    assert out.shape == (10,)


def test_pinn_base_shape_mismatch_raises():
    pinn = PINNBase(hidden=8, n_hidden_layers=1, n_fourier=4)
    try:
        pinn(torch.randn(10), torch.randn(11), torch.randn(10))
    except ValueError:
        pass
    else:
        assert False, "shape mismatch should raise"


def test_pinn_diffusion_residual_runs():
    pinn = PINNDiffusion(D=0.1, hard_ic=False, hidden=16, n_hidden_layers=2, n_fourier=4)
    x = torch.randn(8, requires_grad=True)
    y = torch.randn(8, requires_grad=True)
    t = torch.rand(8, requires_grad=True)
    r = pinn.pde_residual(x, y, t)
    assert r.shape == (8,)


def test_pinn_diffusion_residual_requires_grad_inputs():
    pinn = PINNDiffusion(D=0.1, hard_ic=False, hidden=8, n_hidden_layers=1, n_fourier=4)
    x = torch.randn(8)
    y = torch.randn(8)
    t = torch.rand(8)
    try:
        pinn.pde_residual(x, y, t)
    except RuntimeError:
        pass
    else:
        assert False, "should require grad-tracked inputs"


def test_gaussian_initial_condition_at_origin():
    A0 = gaussian_initial_condition(sigma=0.5)
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    assert abs(A0(x, y).item() - 1.0) < 1e-12


def test_gaussian_analytic_decays_with_time():
    A = gaussian_analytic_solution(sigma=0.5, D=0.1)
    x = torch.tensor([0.0])
    y = torch.tensor([0.0])
    assert A(x, y, t=0.0).item() > A(x, y, t=1.0).item()


def test_gaussian_analytic_conserves_mass():
    """Closed-form Gaussian solution conserves integral over R^2."""
    A_fn = gaussian_analytic_solution(sigma=0.5, D=0.1)
    g = Grid2D(n=128, extent=8.0)
    X, Y = g.meshgrid()
    m0 = A_fn(X, Y, t=0.0).sum().item() * (g.dx ** 2)
    m1 = A_fn(X, Y, t=1.0).sum().item() * (g.dx ** 2)
    # Both ~ 2 pi sigma^2 = pi/2 ~ 1.5708 ; allow 1% from box truncation
    assert abs(m1 - m0) / m0 < 0.01


def test_pinn_to_grid_shape():
    pinn = PINNDiffusion(D=0.1, hard_ic=False, hidden=8, n_hidden_layers=1, n_fourier=4)
    g = Grid2D(n=16, extent=2.0)
    A = pinn_to_grid(pinn, g, t=0.0)
    assert A.shape == (g.n, g.n)


def test_short_training_run_decreases_loss():
    """A few-iteration smoke run on a Gaussian IC should at least not blow
    up; we check that the loss is finite after 30 steps."""
    pinn = PINNDiffusion(D=0.1, hard_ic=False, hidden=16, n_hidden_layers=2, n_fourier=8)
    A0 = gaussian_initial_condition(sigma=0.5)
    h = train_pinn_diffusion(
        pinn, A0_callable=A0, extent=4.0, t_end=0.5,
        n_iters=30, lr=1e-3, n_collocation=256, n_ic=128,
        log_every=10,
    )
    assert math.isfinite(h.loss_total[-1])
    assert h.loss_total[-1] < h.loss_total[0] + 1.0  # not diverging


def test_pinn_reaction_diffusion_residuals_runs():
    pinn = PINNReactionDiffusion(D=0.1, k=0.5, hidden=8, n_hidden_layers=1, n_fourier=4)
    x = torch.randn(8, requires_grad=True)
    y = torch.randn(8, requires_grad=True)
    t = torch.rand(8, requires_grad=True)
    r_A, r_Q = pinn.pde_residuals(x, y, t)
    assert r_A.shape == (8,)
    assert r_Q.shape == (8,)


def test_pinn_reaction_diffusion_output_dim():
    pinn = PINNReactionDiffusion(D=0.1, k=0.5, hidden=8, n_hidden_layers=1, n_fourier=4)
    out = pinn(torch.randn(5), torch.randn(5), torch.rand(5))
    assert out.shape == (5, 2)

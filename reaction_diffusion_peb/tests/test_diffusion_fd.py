"""Tests for the PEB submodule's FD diffusion solver."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.diffusion_fd import (
    diffuse_fd,
    laplacian_5pt,
    step_diffusion_fd,
)


def test_laplacian_constant_field_is_zero():
    H = torch.full((16, 16), 0.7)
    L = laplacian_5pt(H, dx_nm=1.0)
    assert L.abs().max().item() < 1e-6


def test_laplacian_quadratic_constant():
    """For H(x, y) = (x^2 + y^2) / scale the Laplacian is 4 / scale
    everywhere in the interior. Periodic boundaries make the wrap-around
    rows / cols invalid; check only the interior."""
    n = 32
    coord = torch.arange(n, dtype=torch.float32) - n / 2
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    scale = 100.0
    H = (X * X + Y * Y) / scale
    L = laplacian_5pt(H, dx_nm=1.0)
    interior = L[4:-4, 4:-4]
    expected = torch.full_like(interior, 4.0 / scale)
    assert torch.allclose(interior, expected, atol=1e-6)


def test_diffuse_fd_zero_time_identity():
    H = torch.rand(16, 16)
    out = diffuse_fd(H, DH_nm2_s=0.8, t_end_s=0.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_diffuse_fd_zero_diffusivity_identity():
    H = torch.rand(16, 16)
    out = diffuse_fd(H, DH_nm2_s=0.0, t_end_s=10.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_diffuse_fd_conserves_mass_periodic():
    """Periodic BC + interior Laplacian conserves total ``H``."""
    H = torch.rand(32, 32)
    s_before = H.sum().item()
    out = diffuse_fd(H, DH_nm2_s=0.8, t_end_s=20.0, dx_nm=1.0)
    s_after = out.sum().item()
    assert abs(s_after - s_before) / s_before < 1e-4


def test_diffuse_fd_smooths_with_time():
    """A noisy field becomes smoother (lower L1 first-difference) over
    time."""
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    out = diffuse_fd(H, DH_nm2_s=0.8, t_end_s=10.0, dx_nm=1.0)
    grad_H = (H[:, 1:] - H[:, :-1]).abs().mean()
    grad_out = (out[:, 1:] - out[:, :-1]).abs().mean()
    assert grad_out < grad_H


def test_diffuse_fd_blur_grows_with_DH():
    """Larger DH → more smoothing, captured by a smaller L1 first-
    difference of the result."""
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    a = diffuse_fd(H, DH_nm2_s=0.3, t_end_s=10.0, dx_nm=1.0)
    b = diffuse_fd(H, DH_nm2_s=1.5, t_end_s=10.0, dx_nm=1.0)
    grad_a = (a[:, 1:] - a[:, :-1]).abs().mean()
    grad_b = (b[:, 1:] - b[:, :-1]).abs().mean()
    assert grad_b < grad_a


def test_diffuse_fd_rejects_cfl_violation():
    """Forcing a single huge step that violates CFL must raise."""
    H = torch.rand(8, 8)
    try:
        diffuse_fd(H, DH_nm2_s=1.0, t_end_s=10.0, dx_nm=1.0, n_steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("CFL violation should raise")


def test_diffuse_fd_step_function():
    """Single-step helper agrees with one iteration of the full solver."""
    torch.manual_seed(0)
    H = torch.rand(16, 16)
    dx = 1.0
    DH = 0.8
    dt = 0.1 * (dx * dx) / (4.0 * DH)  # well below CFL
    direct = step_diffusion_fd(H, DH, dt, dx)
    via_solver = diffuse_fd(H, DH_nm2_s=DH, t_end_s=dt, dx_nm=dx, n_steps=1)
    assert torch.allclose(direct, via_solver)

"""Tests for the PEB Phase-4 acid-loss FD / FFT solvers."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.diffusion_fft import diffuse_fft
from reaction_diffusion_peb.src.reaction_diffusion import (
    diffuse_acid_loss_fd,
    diffuse_acid_loss_fft,
    expected_mass_decay_factor,
    step_acid_loss_fd,
    total_mass,
)


def test_acid_loss_fd_zero_kloss_matches_diffusion_only():
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    a = diffuse_acid_loss_fd(H, DH_nm2_s=0.8, kloss_s_inv=0.0,
                             t_end_s=20.0, dx_nm=1.0)
    b = diffuse_fd(H, DH_nm2_s=0.8, t_end_s=20.0, dx_nm=1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_acid_loss_fft_zero_kloss_matches_diffusion_only():
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    a = diffuse_acid_loss_fft(H, DH_nm2_s=0.8, kloss_s_inv=0.0,
                              t_end_s=20.0, dx_nm=1.0)
    b = diffuse_fft(H, DH_nm2_s=0.8, t_end_s=20.0, dx_nm=1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_acid_loss_fft_zero_DH_pure_decay():
    """With D=0 the equation becomes dH/dt = -k_loss H whose solution is
    H(t) = H(0) * exp(-k_loss t) at every point."""
    torch.manual_seed(0)
    H = torch.rand(16, 16)
    out = diffuse_acid_loss_fft(H, DH_nm2_s=0.0, kloss_s_inv=0.05,
                                t_end_s=10.0, dx_nm=1.0)
    expected = H * math.exp(-0.05 * 10.0)
    assert torch.allclose(out, expected, atol=1e-5)


def test_acid_loss_fft_mass_decays_exactly():
    """Total mass under FFT should follow exp(-k_loss t) exactly."""
    H = torch.rand(32, 32)
    DH = 0.5
    kloss = 0.02
    t = 30.0
    m0 = total_mass(H, dx_nm=1.0)
    H_t = diffuse_acid_loss_fft(H, DH, kloss, t_end_s=t, dx_nm=1.0)
    m_t = total_mass(H_t, dx_nm=1.0)
    expected = m0 * expected_mass_decay_factor(kloss, t)
    rel = abs(m_t - expected) / expected
    assert rel < 1e-4


def test_acid_loss_fd_mass_decays_close_to_analytic():
    """FD mass decay should match the analytic exp(-k_loss t) up to
    discretization error (a few percent for moderate dt)."""
    H = torch.rand(32, 32)
    DH = 0.5
    kloss = 0.02
    t = 30.0
    m0 = total_mass(H, dx_nm=1.0)
    H_t = diffuse_acid_loss_fd(H, DH, kloss, t_end_s=t, dx_nm=1.0,
                               cfl_safety=0.5)
    m_t = total_mass(H_t, dx_nm=1.0)
    expected = m0 * expected_mass_decay_factor(kloss, t)
    rel = abs(m_t - expected) / expected
    assert rel < 1e-2


def test_acid_loss_fd_zero_time_identity():
    H = torch.rand(8, 8)
    out = diffuse_acid_loss_fd(H, DH_nm2_s=0.8, kloss_s_inv=0.005,
                               t_end_s=0.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_acid_loss_fft_zero_time_identity():
    H = torch.rand(8, 8)
    out = diffuse_acid_loss_fft(H, DH_nm2_s=0.8, kloss_s_inv=0.005,
                                t_end_s=0.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_acid_loss_fd_step_function():
    """Single-step helper agrees with one iteration of the full solver."""
    torch.manual_seed(0)
    H = torch.rand(16, 16)
    DH, kloss = 0.5, 0.01
    dt_max = (1.0 ** 2) / (4.0 * DH)
    dt = 0.1 * dt_max
    direct = step_acid_loss_fd(H, DH, kloss, dt, dx_nm=1.0)
    via_solver = diffuse_acid_loss_fd(H, DH, kloss,
                                      t_end_s=dt, dx_nm=1.0, n_steps=1)
    assert torch.allclose(direct, via_solver)


def test_acid_loss_fd_rejects_cfl_violation():
    H = torch.rand(8, 8)
    try:
        diffuse_acid_loss_fd(H, DH_nm2_s=1.0, kloss_s_inv=0.0,
                             t_end_s=10.0, dx_nm=1.0, n_steps=1)
    except ValueError:
        pass
    else:
        raise AssertionError("CFL violation should raise")


def test_fd_vs_fft_agree_on_smooth_gaussian_with_loss():
    n = 64
    dx = 1.0
    coord = (torch.arange(n, dtype=torch.float32) - n / 2) * dx
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    sigma = 4.0
    H = torch.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma))
    DH, kloss, t = 0.4, 0.02, 1.0
    H_fft = diffuse_acid_loss_fft(H, DH, kloss, t, dx_nm=dx)
    H_fd = diffuse_acid_loss_fd(H, DH, kloss, t, dx_nm=dx, cfl_safety=0.25)
    rel = (H_fft - H_fd).abs().max().item() / (H_fft.abs().max().item() + 1e-12)
    assert rel < 5e-2, f"FD vs FFT mismatch with loss too large: {rel}"

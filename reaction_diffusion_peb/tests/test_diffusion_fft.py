"""Tests for the PEB submodule's FFT diffusion solver."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.diffusion_fft import (
    diffuse_fft,
    diffuse_fft_by_length,
)
from reaction_diffusion_peb.src.fft_utils import fft2c, freq_grid_nm, ifft2c


def test_fft2c_round_trip():
    x = torch.complex(torch.rand(32, 32), torch.rand(32, 32))
    y = ifft2c(fft2c(x))
    assert torch.allclose(x, y, atol=1e-5)


def test_freq_grid_dc_at_center():
    fx, fy = freq_grid_nm(32, grid_spacing_nm=1.0)
    assert abs(fx[16, 16].item()) < 1e-6
    assert abs(fy[16, 16].item()) < 1e-6


def test_diffuse_fft_zero_time_identity():
    H = torch.rand(16, 16)
    out = diffuse_fft(H, DH_nm2_s=0.8, t_end_s=0.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_diffuse_fft_zero_diffusivity_identity():
    H = torch.rand(16, 16)
    out = diffuse_fft(H, DH_nm2_s=0.0, t_end_s=10.0, dx_nm=1.0)
    assert torch.equal(H, out)


def test_diffuse_fft_constant_field_unchanged():
    H = torch.full((32, 32), 0.5)
    out = diffuse_fft(H, DH_nm2_s=0.8, t_end_s=20.0, dx_nm=1.0)
    assert torch.allclose(out, H, atol=1e-6)


def test_diffuse_fft_conserves_mass():
    H = torch.rand(32, 32)
    out = diffuse_fft(H, DH_nm2_s=0.8, t_end_s=20.0, dx_nm=1.0)
    s_before = H.sum().item()
    s_after = out.sum().item()
    assert abs(s_after - s_before) / s_before < 1e-4


def test_diffuse_fft_blur_grows_with_DH():
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    a = diffuse_fft(H, DH_nm2_s=0.3, t_end_s=10.0, dx_nm=1.0)
    b = diffuse_fft(H, DH_nm2_s=1.5, t_end_s=10.0, dx_nm=1.0)
    grad_a = (a[:, 1:] - a[:, :-1]).abs().mean()
    grad_b = (b[:, 1:] - b[:, :-1]).abs().mean()
    assert grad_b < grad_a


def test_fft_by_length_matches_explicit_DH_t():
    """``diffuse_fft_by_length(L)`` must match ``diffuse_fft(D, t)``
    when ``L = sqrt(2 * D * t)``."""
    torch.manual_seed(0)
    H = torch.rand(32, 32)
    DH, t = 0.5, 8.0
    L = math.sqrt(2.0 * DH * t)
    a = diffuse_fft(H, DH, t, dx_nm=1.0)
    b = diffuse_fft_by_length(H, L_nm=L, dx_nm=1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_fft_matches_fd_on_smooth_gaussian():
    """FD and FFT must agree to a few percent on a smooth, well-resolved
    Gaussian for a short diffusion time."""
    n = 64
    dx = 1.0
    coord = (torch.arange(n, dtype=torch.float32) - n / 2) * dx
    X, Y = torch.meshgrid(coord, coord, indexing="xy")
    sigma = 4.0
    H = torch.exp(-(X * X + Y * Y) / (2.0 * sigma * sigma))
    DH, t = 0.4, 1.0
    H_fft = diffuse_fft(H, DH, t, dx_nm=dx)
    H_fd = diffuse_fd(H, DH, t, dx_nm=dx, cfl_safety=0.25)
    rel_err = (H_fft - H_fd).abs().max().item() / (H_fft.abs().max().item() + 1e-12)
    assert rel_err < 5e-2, f"FD vs FFT mismatch too large: {rel_err}"


def test_diffuse_fft_real_input_real_output():
    H = torch.rand(16, 16)
    out = diffuse_fft(H, DH_nm2_s=0.8, t_end_s=5.0, dx_nm=1.0)
    assert out.dtype == torch.float32
    assert not torch.is_complex(out)


def test_diffuse_fft_invalid_args():
    H = torch.rand(8, 8)
    for bad in (-0.1,):
        try:
            diffuse_fft(H, DH_nm2_s=bad, t_end_s=1.0, dx_nm=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("negative DH should raise")
        try:
            diffuse_fft(H, DH_nm2_s=0.5, t_end_s=bad, dx_nm=1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("negative t should raise")

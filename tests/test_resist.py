"""Tests for the Phase-5 resist modules."""

from __future__ import annotations

import math

import torch

from src.common.grid import Grid2D
from src.resist.diffusion_fd import diffuse_fd, laplacian_5pt
from src.resist.diffusion_fft import diffuse_fft, diffuse_fft_by_length
from src.resist.exposure import acid_from_aerial
from src.resist.reaction_diffusion import evolve_reaction_diffusion
from src.resist.threshold import (
    hard_threshold,
    measure_cd_horizontal,
    soft_threshold,
    thresholded_area,
)


def _grid(n: int = 64, extent: float = 4.0):
    return Grid2D(n=n, extent=extent)


# ---- exposure -------------------------------------------------------------

def test_acid_from_aerial_zero_dose_zero_acid():
    aerial = torch.rand(8, 8)
    A0 = acid_from_aerial(aerial, dose=0.0)
    assert torch.allclose(A0, torch.zeros_like(A0))


def test_acid_from_aerial_saturates_at_one():
    aerial = torch.ones(8, 8)
    A0 = acid_from_aerial(aerial, dose=20.0, eta=1.0)
    # 1 - exp(-20) ~ 1
    assert (A0 > 0.999).all()


def test_acid_from_aerial_monotonic_in_dose():
    aerial = torch.rand(8, 8)
    A_low = acid_from_aerial(aerial, dose=0.5)
    A_high = acid_from_aerial(aerial, dose=2.0)
    assert (A_high >= A_low - 1e-6).all()


def test_acid_from_aerial_differentiable():
    aerial = torch.rand(8, 8, requires_grad=True)
    A0 = acid_from_aerial(aerial, dose=1.0)
    A0.sum().backward()
    assert aerial.grad is not None and aerial.grad.abs().max().item() > 0


# ---- FD diffusion ---------------------------------------------------------

def test_laplacian_constant_field_zero():
    A = torch.full((16, 16), 0.7)
    assert laplacian_5pt(A, dx=0.1).abs().max().item() < 1e-6


def test_laplacian_quadratic_constant():
    """For A(x, y) = x**2 + y**2 the Laplacian is 4 everywhere (in the
    interior; periodic BCs introduce wrap-around so we only check the
    interior)."""
    n = 32
    x = torch.arange(n, dtype=torch.float32) - n / 2
    X, Y = torch.meshgrid(x, x, indexing="xy")
    A = (X * X + Y * Y) * 0.01  # scale so values stay small
    L = laplacian_5pt(A, dx=1.0)
    interior = L[4:-4, 4:-4]
    assert torch.allclose(interior, torch.full_like(interior, 0.04), atol=1e-5)


def test_diffuse_fd_zero_time_identity():
    A = torch.rand(16, 16)
    out = diffuse_fd(A, D=1.0, t_end=0.0, dx=0.1)
    assert torch.equal(A, out)


def test_diffuse_fd_conservation_periodic():
    """Periodic BC + interior Laplacian conserves total mass."""
    A = torch.rand(32, 32)
    s_before = A.sum().item()
    out = diffuse_fd(A, D=0.05, t_end=0.5, dx=0.1)
    s_after = out.sum().item()
    assert abs(s_after - s_before) / s_before < 1e-4


def test_diffuse_fd_rejects_cfl_violation():
    A = torch.rand(8, 8)
    # dt = 1.0 with D=1, dx=0.1 -> dt > dx^2/(4D) = 0.0025
    try:
        diffuse_fd(A, D=1.0, t_end=1.0, dx=0.1, n_steps=1)
    except ValueError:
        pass
    else:
        assert False, "should reject CFL-violating step"


# ---- FFT diffusion --------------------------------------------------------

def test_diffuse_fft_zero_returns_input():
    g = _grid()
    A = torch.rand(g.n, g.n)
    out = diffuse_fft(A, g, D=0.0, t=1.0)
    assert torch.equal(A, out)
    out2 = diffuse_fft(A, g, D=1.0, t=0.0)
    assert torch.equal(A, out2)


def test_diffuse_fft_constant_field_unchanged():
    g = _grid()
    A = torch.full((g.n, g.n), 0.5)
    out = diffuse_fft(A, g, D=1.0, t=0.5)
    assert torch.allclose(out, A, atol=1e-6)


def test_diffuse_fft_conserves_mass():
    g = _grid(n=64, extent=8.0)
    A = torch.rand(g.n, g.n)
    out = diffuse_fft(A, g, D=0.1, t=0.5)
    s_before = A.sum().item()
    s_after = out.sum().item()
    assert abs(s_after - s_before) / s_before < 1e-3


def test_fft_matches_fd_for_small_t():
    """FD and FFT must agree to a few percent on a smooth, well-resolved
    initial field after a short diffusion. The 5-point FD Laplacian is
    second-order accurate in dx, so a Gaussian with sigma = 5 dx is in the
    regime where the truncation error is small.

    Random initial fields (high-frequency content right at Nyquist) are
    intentionally avoided here — they amplify FD truncation by ~1e-1 even
    for one explicit Euler step.
    """
    g = _grid(n=64, extent=4.0)
    X, Y = g.meshgrid()
    A = torch.exp(-(X * X + Y * Y) / (2 * 0.25 ** 2))  # sigma 0.25, ~4 dx wide
    D, t = 0.02, 0.05
    A_fft = diffuse_fft(A, g, D=D, t=t)
    A_fd = diffuse_fd(A, D=D, t_end=t, dx=g.dx, cfl_safety=0.25)
    rel = (A_fft - A_fd).abs().max().item() / (A_fft.abs().max().item() + 1e-12)
    assert rel < 5e-2, f"FD vs FFT mismatch too large: rel {rel}"


def test_diffuse_fft_by_length_matches_explicit():
    g = _grid()
    A = torch.rand(g.n, g.n)
    D, t = 0.02, 0.5
    L = math.sqrt(2 * D * t)
    a1 = diffuse_fft(A, g, D=D, t=t)
    a2 = diffuse_fft_by_length(A, g, diffusion_length=L)
    assert torch.allclose(a1, a2, atol=1e-5)


# ---- reaction diffusion ---------------------------------------------------

def test_reaction_diffusion_consumes_acid():
    """With D=0 (no diffusion) and large k, A and Q should both decrease
    where they overlap, with their difference (A - Q) preserved in
    each pixel."""
    n = 16
    A = torch.full((n, n), 1.0)
    Q = torch.full((n, n), 0.5)
    diff_before = (A - Q).clone()
    A_out, Q_out = evolve_reaction_diffusion(A, Q, D_A=0.0, k=2.0,
                                             t_end=1.0, dx=0.1)
    assert (A_out <= A + 1e-6).all()
    assert (Q_out <= Q + 1e-6).all()
    diff_after = A_out - Q_out
    assert torch.allclose(diff_before, diff_after, atol=1e-3)


def test_reaction_diffusion_zero_quencher_pure_diffusion():
    """With Q == 0 the reaction term vanishes and the result must match
    pure diffusion."""
    g = _grid(n=32, extent=2.0)
    A = torch.rand(g.n, g.n)
    Q = torch.zeros_like(A)
    A_rd, _ = evolve_reaction_diffusion(A, Q, D_A=0.05, k=1.0,
                                        t_end=0.05, dx=g.dx)
    A_pure = diffuse_fd(A, D=0.05, t_end=0.05, dx=g.dx, cfl_safety=0.25)
    rel = (A_rd - A_pure).abs().max().item() / (A_pure.abs().max().item() + 1e-12)
    assert rel < 0.1


# ---- threshold ------------------------------------------------------------

def test_soft_threshold_monotonic():
    A = torch.linspace(0, 1, 11)
    R = soft_threshold(A, A_th=0.5, beta=20.0)
    diffs = R[1:] - R[:-1]
    assert (diffs >= -1e-6).all()  # monotonic non-decreasing


def test_soft_threshold_limits():
    A = torch.tensor([-10.0, 0.0, 0.5, 1.0, 10.0])
    R = soft_threshold(A, A_th=0.5, beta=20.0)
    assert R[0].item() < 1e-3
    assert R[-1].item() > 1.0 - 1e-3


def test_hard_threshold_binary():
    A = torch.tensor([[0.1, 0.6], [0.5, 0.9]])
    R = hard_threshold(A, A_th=0.5)
    assert torch.equal(R, torch.tensor([[0.0, 1.0], [0.0, 1.0]]))


def test_measure_cd_isolated_line():
    """A 6-pixel-wide stripe along the middle row should give CD = 6 * dx."""
    n = 16
    field = torch.zeros((n, n))
    field[n // 2, 5:11] = 1.0
    cd = measure_cd_horizontal(field, threshold=0.5, dx=0.1)
    assert abs(cd - 0.6) < 1e-6


def test_thresholded_area():
    field = torch.tensor([[0.1, 0.7], [0.5, 0.9]])
    assert thresholded_area(field, threshold=0.5) == 2

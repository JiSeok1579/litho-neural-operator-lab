"""Tests for the PEB Phase-5 deprotection FD solver.

The user's six verification criteria are covered explicitly:

  1. kdep = 0  ->  P stays at 0 everywhere
  2. H = 0 region  ->  P does not grow there
  3. Region with larger H ramps P up faster
  4. P stays in [0, 1]
  5. Larger kdep -> larger P_final
  6. Threshold contour widens with kdep / PEB time
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.deprotection import (
    deprotected_fraction_from_H_integral,
    evolve_acid_loss_deprotection_fd,
    step_acid_loss_deprotection_fd,
    thresholded_area,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


# ---- helpers -----------------------------------------------------------

def _gaussian_H0(grid_size: int = 64, sigma_px: float = 6.0,
                Hmax: float = 0.2) -> torch.Tensor:
    I = gaussian_spot(grid_size, sigma_px=sigma_px)
    return acid_generation(I, dose=1.0, eta=1.0, Hmax=Hmax)


# ---- 1. kdep = 0 -> P stays at 0 ---------------------------------------

def test_zero_kdep_gives_zero_P_everywhere():
    H0 = _gaussian_H0()
    H_t, P_t = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert torch.allclose(P_t, torch.zeros_like(P_t))


# ---- 2. H = 0 region -> P does not grow --------------------------------

def test_no_acid_no_deprotection_locally():
    """Far from the Gaussian spot the initial acid is exactly 0; without
    diffusion bringing acid to those pixels, P must stay at 0."""
    H0 = _gaussian_H0()
    # Set DH=0 so acid never spreads — the dark periphery has H=0 forever.
    H_t, P_t = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.0, kloss_s_inv=0.0, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0,
    )
    n = H0.shape[-1]
    # Corner is far from the centered Gaussian -> H_0[corner] ~ 0.
    corner_H = H0[0, 0].item()
    corner_P = P_t[0, 0].item()
    assert corner_H < 1e-6, f"corner H_0 should be near 0, got {corner_H}"
    assert corner_P < 1e-6, f"corner P after evolution should be 0, got {corner_P}"


# ---- 3. Region with larger H ramps P up faster -------------------------

def test_high_H_grows_P_first():
    """At every time, the pixel with maximum H_0 should have the largest
    P. Test by stopping early and comparing peak vs corner."""
    H0 = _gaussian_H0()
    n = H0.shape[-1]
    center = (n // 2, n // 2)
    H_t, P_t = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.0, kloss_s_inv=0.0, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0,  # short run
    )
    peak_P = P_t[center].item()
    corner_P = P_t[0, 0].item()
    assert peak_P > corner_P + 1e-3


# ---- 4. P stays in [0, 1] ---------------------------------------------

def test_P_in_unit_interval():
    H0 = _gaussian_H0()
    for kdep in [0.01, 0.1, 0.5, 5.0]:
        H_t, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=kdep,
            t_end_s=60.0, dx_nm=1.0,
        )
        assert (P_t >= 0).all().item(), f"kdep={kdep}: P went below 0"
        assert (P_t <= 1).all().item(), f"kdep={kdep}: P went above 1"


# ---- 5. Larger kdep -> larger P_final ---------------------------------

def test_larger_kdep_increases_P_final():
    H0 = _gaussian_H0()
    P_means = []
    P_maxes = []
    for kdep in [0.0, 0.05, 0.5, 5.0]:
        _, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=kdep,
            t_end_s=60.0, dx_nm=1.0,
        )
        P_means.append(P_t.mean().item())
        P_maxes.append(P_t.max().item())
    # Strictly non-decreasing in kdep
    for k in range(len(P_means) - 1):
        assert P_means[k] <= P_means[k + 1] + 1e-9
        assert P_maxes[k] <= P_maxes[k + 1] + 1e-9
    # And the largest kdep should have visibly larger P
    assert P_means[-1] > P_means[0] + 1e-3


# ---- 6. Threshold contour widens with kdep / PEB time -----------------

def test_threshold_contour_widens_with_kdep():
    H0 = _gaussian_H0()
    areas = []
    for kdep in [0.1, 0.5, 1.0, 5.0]:
        _, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=kdep,
            t_end_s=60.0, dx_nm=1.0,
        )
        areas.append(thresholded_area(P_t, P_threshold=0.5))
    # Non-decreasing
    for k in range(len(areas) - 1):
        assert areas[k] <= areas[k + 1]
    assert areas[-1] > areas[0]


def test_threshold_contour_widens_with_time():
    H0 = _gaussian_H0()
    areas = []
    for t_end in [15.0, 30.0, 60.0, 120.0]:
        _, P_t = evolve_acid_loss_deprotection_fd(
            H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
            t_end_s=t_end, dx_nm=1.0,
        )
        areas.append(thresholded_area(P_t, P_threshold=0.5))
    for k in range(len(areas) - 1):
        assert areas[k] <= areas[k + 1]
    assert areas[-1] > areas[0]


# ---- mechanical / API tests -------------------------------------------

def test_step_function_consistent():
    """Single-step helper agrees with one iteration of the full evolver."""
    torch.manual_seed(0)
    H = torch.rand(16, 16) * 0.2
    P = torch.zeros_like(H)
    DH, kloss, kdep = 0.5, 0.005, 0.1
    dx = 1.0
    dt_max = (dx * dx) / (4.0 * DH)
    dt = 0.1 * dt_max
    direct_H, direct_P = step_acid_loss_deprotection_fd(
        H, P, DH, kloss, kdep, dt, dx,
    )
    # The full evolver re-initializes P=0; pass H as the starting field.
    via_solver_H, via_solver_P = evolve_acid_loss_deprotection_fd(
        H, DH, kloss, kdep, t_end_s=dt, dx_nm=dx, n_steps=1,
    )
    assert torch.allclose(direct_H, via_solver_H)
    assert torch.allclose(direct_P, via_solver_P)


def test_zero_time_identity():
    H0 = _gaussian_H0()
    H_t, P_t = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=0.0, dx_nm=1.0,
    )
    assert torch.equal(H_t, H0)
    assert torch.allclose(P_t, torch.zeros_like(H0))


def test_analytic_P_from_H_integral_static_H():
    """If H is constant in time at value H_const, the closed form gives
    P(t) = 1 - exp(-k_dep * H_const * t). Check on a grid of constants."""
    H_const = torch.full((4, 4), 0.1)
    t = 10.0
    H_integral = H_const * t  # exact integral for static H
    P = deprotected_fraction_from_H_integral(H_integral, kdep_s_inv=0.5)
    expected = 1.0 - math.exp(-0.5 * 0.1 * t)
    assert torch.allclose(P, torch.full_like(P, expected), atol=1e-6)


def test_thresholded_area_monotone():
    P = torch.linspace(0.0, 1.0, 25).reshape(5, 5)
    a_low = thresholded_area(P, P_threshold=0.2)
    a_high = thresholded_area(P, P_threshold=0.8)
    assert a_low > a_high


def test_invalid_args():
    H0 = _gaussian_H0()
    for kw in ("DH_nm2_s", "kloss_s_inv", "kdep_s_inv", "t_end_s"):
        kwargs = dict(DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
                      t_end_s=10.0, dx_nm=1.0)
        kwargs[kw] = -0.1
        try:
            evolve_acid_loss_deprotection_fd(H0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} negative should raise")

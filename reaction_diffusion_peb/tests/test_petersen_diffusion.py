"""Phase 11 tests — Petersen nonlinear-diffusion FD evolver.

The crucial contract is: ``alpha = 0`` reduces the Petersen evolver
to the Phase-8 integrated FD evolver bit-for-bit (because the
variable-coefficient operator collapses to ``D * laplacian_5pt`` when
``D`` is constant). All other tests check incremental physics-level
contracts on top of that."""

from __future__ import annotations

import math

import pytest
import torch

from reaction_diffusion_peb.src.diffusion_fd import laplacian_5pt
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    evolve_full_reaction_diffusion_fd_at_T,
)
from reaction_diffusion_peb.src.petersen_diffusion import (
    divergence_diffusion_5pt,
    evolve_petersen_full_fd_at_T,
    evolve_petersen_full_fd_at_T_with_budget,
    petersen_DH_field,
    stability_report_petersen,
    step_petersen_full_fd,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


GRID = 32
DX = 1.0
T_END = 4.0
DH0 = 0.8
DQ = 0.08
KQ = 1.0
KDEP = 0.5
KLOSS = 0.005
Q0 = 0.1
T_REF = 100.0
EA = 100.0


@pytest.fixture(scope="module")
def H0():
    I = gaussian_spot(GRID, sigma_px=4.0)
    return acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2)


# ---- variable-coefficient operator -------------------------------------

def test_divergence_constant_D_matches_laplacian_5pt():
    H = torch.randn(GRID, GRID)
    D_const = torch.full_like(H, 0.7)
    out_var = divergence_diffusion_5pt(H, D_const, dx_nm=DX)
    out_const = 0.7 * laplacian_5pt(H, dx_nm=DX)
    assert torch.allclose(out_var, out_const, atol=1e-6)


def test_divergence_zero_D_returns_zero():
    H = torch.randn(GRID, GRID)
    D_zero = torch.zeros_like(H)
    out = divergence_diffusion_5pt(H, D_zero, dx_nm=DX)
    assert torch.allclose(out, torch.zeros_like(H))


def test_divergence_constant_field_returns_zero():
    """∇·(D ∇H) where H is spatially constant must be zero, regardless
    of D."""
    H = torch.full((GRID, GRID), 1.7)
    D = torch.rand(GRID, GRID) + 0.1
    out = divergence_diffusion_5pt(H, D, dx_nm=DX)
    assert torch.allclose(out, torch.zeros_like(H), atol=1e-6)


def test_divergence_shape_mismatch_rejected():
    H = torch.randn(GRID, GRID)
    D = torch.randn(GRID + 1, GRID)
    with pytest.raises(ValueError):
        divergence_diffusion_5pt(H, D, dx_nm=DX)


# ---- petersen_DH_field --------------------------------------------------

def test_petersen_DH_field_alpha_zero_returns_DH0():
    P = torch.rand(GRID, GRID)
    DH = petersen_DH_field(P, DH0_nm2_s=DH0, alpha=0.0)
    assert torch.allclose(DH, torch.full_like(DH, DH0))


def test_petersen_DH_field_increases_with_P():
    P_low = torch.zeros(GRID, GRID)
    P_high = torch.ones(GRID, GRID)
    DH_low = petersen_DH_field(P_low, DH0_nm2_s=DH0, alpha=2.0)
    DH_high = petersen_DH_field(P_high, DH0_nm2_s=DH0, alpha=2.0)
    assert torch.allclose(DH_low, torch.full_like(DH_low, DH0))
    assert torch.allclose(DH_high, torch.full_like(DH_high, DH0 * math.exp(2.0)))


def test_petersen_DH_field_rejects_negative_DH0():
    with pytest.raises(ValueError):
        petersen_DH_field(torch.zeros(8, 8), DH0_nm2_s=-1.0, alpha=1.0)


# ---- evolver — limit cases ---------------------------------------------

def test_zero_time_returns_initial_conditions(H0):
    H, Q, P = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=2.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=0.0, dx_nm=DX,
    )
    assert torch.allclose(H, H0)
    assert torch.allclose(Q, torch.full_like(H0, Q0))
    assert torch.allclose(P, torch.zeros_like(H0))


def test_alpha_zero_matches_phase8_evolver(H0):
    """Petersen with α = 0 must be bit-identical to Phase 8."""
    Hp, Qp, Pp = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=0.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    Hf, Qf, Pf = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    # Both evolvers pick their own dt grid — but with α = 0 the
    # stability bounds match and the sequence of explicit-Euler steps
    # is identical to floating-point precision.
    assert torch.allclose(Hp, Hf, atol=1e-6)
    assert torch.allclose(Qp, Qf, atol=1e-6)
    assert torch.allclose(Pp, Pf, atol=1e-6)


def test_alpha_positive_speeds_up_acid_diffusion(H0):
    """Once any P develops, larger α → larger D_H → more diffusion →
    smaller H peak at t_end."""
    _, _, _ = evolve_petersen_full_fd_at_T(  # warm-up; ensures fixture loaded
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=0.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_a0, _, _ = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=0.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_aHi, _, _ = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=3.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    assert float(H_aHi.max().item()) <= float(H_a0.max().item()) + 1e-7


def test_negative_alpha_rejected(H0):
    with pytest.raises(ValueError):
        evolve_petersen_full_fd_at_T(
            H0, Q0_mol_dm3=Q0,
            DH0_nm2_s=DH0, petersen_alpha=-0.1, DQ_nm2_s=DQ,
            kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
            temperature_c=T_REF, temperature_ref_c=T_REF,
            activation_energy_kj_mol=EA,
            t_end_s=T_END, dx_nm=DX,
        )


# ---- physical contracts ------------------------------------------------

def test_fields_stay_non_negative_and_bounded(H0):
    H, Q, P = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=2.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    assert float(H.min().item()) >= 0.0
    assert float(Q.min().item()) >= 0.0
    assert 0.0 <= float(P.min().item())
    assert float(P.max().item()) <= 1.0


def test_mass_budget_closes_with_petersen(H0):
    """Variable-coefficient flux-form discretization is conservative —
    H budget must close to ~float-precision even with α > 0."""
    _, _, _, hist = evolve_petersen_full_fd_at_T_with_budget(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=2.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    last = hist[-1]
    assert last.H_budget_relative_error < 1e-5
    assert last.Q_budget_relative_error < 1e-5


def test_mass_budget_no_kq_no_kloss_no_kdep_pure_diffusion(H0):
    """Petersen acid diffusion alone (no reactions) must conserve the
    total acid mass exactly under periodic BCs even when D is
    spatially varying."""
    H, _, _ = evolve_petersen_full_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=2.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=0.0, kdep_ref_s_inv=0.0,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    m0 = float(H0.sum().item()) * DX * DX
    m1 = float(H.sum().item()) * DX * DX
    assert math.isclose(m0, m1, rel_tol=1e-5)


# ---- stability report --------------------------------------------------

def test_stability_dt_diff_shrinks_with_alpha(H0):
    rep_low = stability_report_petersen(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=0.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        dx_nm=DX,
    )
    rep_hi = stability_report_petersen(
        H0, Q0_mol_dm3=Q0,
        DH0_nm2_s=DH0, petersen_alpha=3.0, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        dx_nm=DX,
    )
    assert rep_hi["DH_max"] > rep_low["DH_max"]
    assert rep_hi["dt_diff"] < rep_low["dt_diff"]


def test_step_function_one_step_zero_alpha_matches_phase8(H0):
    """Single explicit-Euler step at α = 0 must be identical to the
    Phase-7 step (which Phase 8 wraps)."""
    Q = torch.full_like(H0, Q0)
    P = torch.zeros_like(H0)
    dt = 0.1
    Hp, Qp, Pp = step_petersen_full_fd(
        H0, Q, P,
        DH0_nm2_s=DH0, petersen_alpha=0.0, DQ_nm2_s=DQ,
        kq_s_inv=KQ, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        dt_s=dt, dx_nm=DX,
    )
    # Direct Phase-7-like step using laplacian_5pt:
    HQ = H0 * Q
    H_phase7 = (H0 + dt * (DH0 * laplacian_5pt(H0, DX)
                           - KQ * HQ - KLOSS * H0)).clamp(min=0.0)
    Q_phase7 = (Q + dt * (DQ * laplacian_5pt(Q, DX) - KQ * HQ)).clamp(min=0.0)
    P_phase7 = (P + dt * (KDEP * H0 * (1.0 - P))).clamp(min=0.0, max=1.0)
    assert torch.allclose(Hp, H_phase7, atol=1e-6)
    assert torch.allclose(Qp, Q_phase7, atol=1e-6)
    assert torch.allclose(Pp, P_phase7, atol=1e-6)

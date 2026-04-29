"""Phase 8 tests — the integrated FD evolver must reproduce every
earlier phase as a limit case."""

from __future__ import annotations

import math

import pytest
import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
)
from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    apply_arrhenius_to_full_rates,
    evolve_full_reaction_diffusion_fd_at_T,
    evolve_full_reaction_diffusion_fd_at_T_with_budget,
    stability_report_at_T,
)
from reaction_diffusion_peb.src.arrhenius import (
    arrhenius_factor,
    evolve_acid_loss_deprotection_fd_at_T,
)
from reaction_diffusion_peb.src.quencher_reaction import evolve_quencher_fd
from reaction_diffusion_peb.src.reaction_diffusion import diffuse_acid_loss_fd
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


GRID = 32
DX = 1.0
T_END = 4.0
DH = 0.8
DQ = 0.08
Q0 = 0.1
KDEP = 0.5
KLOSS = 0.005
KQ = 1.0
T_REF = 100.0
EA = 100.0


@pytest.fixture(scope="module")
def H0():
    I = gaussian_spot(GRID, sigma_px=4.0)
    return acid_generation(I, dose=1.0, eta=1.0, Hmax=0.2)


# ---- Arrhenius rate scaling --------------------------------------------

def test_apply_arrhenius_T_eq_Tref_returns_input_rates():
    kdep, kloss, kq = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS, kq_ref_s_inv=KQ,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
    )
    assert math.isclose(kdep, KDEP)
    assert math.isclose(kloss, KLOSS)
    assert math.isclose(kq, KQ)


def test_apply_arrhenius_zero_Ea_returns_input_rates():
    kdep, kloss, kq = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS, kq_ref_s_inv=KQ,
        temperature_c=120.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=0.0,
    )
    assert math.isclose(kdep, KDEP)
    assert math.isclose(kloss, KLOSS)
    assert math.isclose(kq, KQ)


def test_apply_arrhenius_hot_T_speeds_up_all_rates():
    f = arrhenius_factor(120.0, T_REF, EA)
    assert f > 1.0
    kdep, kloss, kq = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS, kq_ref_s_inv=KQ,
        temperature_c=120.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
    )
    assert math.isclose(kdep, KDEP * f, rel_tol=1e-9)
    assert math.isclose(kloss, KLOSS * f, rel_tol=1e-9)
    assert math.isclose(kq, KQ * f, rel_tol=1e-9)


def test_apply_arrhenius_negative_inputs_rejected():
    with pytest.raises(ValueError):
        apply_arrhenius_to_full_rates(
            kdep_ref_s_inv=-1.0, kloss_ref_s_inv=KLOSS, kq_ref_s_inv=KQ,
            temperature_c=T_REF, temperature_ref_c=T_REF,
            activation_energy_kj_mol=EA,
        )
    with pytest.raises(ValueError):
        apply_arrhenius_to_full_rates(
            kdep_ref_s_inv=KDEP, kloss_ref_s_inv=-1.0, kq_ref_s_inv=KQ,
            temperature_c=T_REF, temperature_ref_c=T_REF,
            activation_energy_kj_mol=EA,
        )
    with pytest.raises(ValueError):
        apply_arrhenius_to_full_rates(
            kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS, kq_ref_s_inv=-1.0,
            temperature_c=T_REF, temperature_ref_c=T_REF,
            activation_energy_kj_mol=EA,
        )


# ---- limit-case reductions to earlier phases ---------------------------

def test_zero_time_returns_initial_conditions(H0):
    H, Q, P = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=0.0, dx_nm=DX,
    )
    assert torch.allclose(H, H0)
    assert torch.allclose(Q, torch.full_like(H0, Q0))
    assert torch.allclose(P, torch.zeros_like(H0))


def test_T_equals_Tref_matches_phase7_quencher(H0):
    """At T = T_ref the Arrhenius factor is exactly 1, so Phase 8 must
    be bit-identical to Phase 7 with the same reference rates."""
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_p7, Q_p7, P_p7 = evolve_quencher_fd(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_s_inv=KQ, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        t_end_s=T_END, dx_nm=DX,
    )
    assert torch.allclose(H_full, H_p7, atol=1e-10)
    assert torch.allclose(Q_full, Q_p7, atol=1e-10)
    assert torch.allclose(P_full, P_p7, atol=1e-10)


def test_zero_Ea_matches_phase7_quencher_at_any_T(H0):
    """Ea = 0 also drops the temperature dependence."""
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=125.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=0.0,
        t_end_s=T_END, dx_nm=DX,
    )
    H_p7, Q_p7, P_p7 = evolve_quencher_fd(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_s_inv=KQ, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        t_end_s=T_END, dx_nm=DX,
    )
    assert torch.allclose(H_full, H_p7, atol=1e-10)
    assert torch.allclose(Q_full, Q_p7, atol=1e-10)
    assert torch.allclose(P_full, P_p7, atol=1e-10)


def test_kq_zero_matches_phase6_arrhenius_HP_evolver(H0):
    """With kq = 0 the (H, Q, P) system decouples — Q stays uniform and
    constant, and (H, P) reduce to the Phase-6 Arrhenius evolver."""
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=110.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_p6, P_p6 = evolve_acid_loss_deprotection_fd_at_T(
        H0,
        DH_nm2_s=DH,
        kdep_ref_s_inv=KDEP, kloss_ref_s_inv=KLOSS,
        temperature_c=110.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    # Tolerance is small but not 0 because the two evolvers may pick
    # slightly different dt grids for the same stability bound.
    assert torch.allclose(H_full, H_p6, atol=1e-5)
    assert torch.allclose(P_full, P_p6, atol=1e-5)
    # Q stays exactly uniform-and-constant because the only sink is k_q.
    assert torch.allclose(Q_full, torch.full_like(H0, Q0), atol=1e-10)


def test_kq_zero_kdep_zero_matches_phase4_acid_loss(H0):
    """kq = kdep = 0 -> only the diffusion + linear-loss equation."""
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=0.0,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_p4 = diffuse_acid_loss_fd(
        H0, DH_nm2_s=DH, kloss_s_inv=KLOSS,
        t_end_s=T_END, dx_nm=DX,
    )
    assert torch.allclose(H_full, H_p4, atol=1e-5)
    assert torch.allclose(P_full, torch.zeros_like(H0), atol=1e-10)


def test_kq_zero_kdep_zero_kloss_zero_matches_phase2_diffusion(H0):
    """All reactions off -> pure heat equation."""
    H_full, Q_full, P_full = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=0.0, kdep_ref_s_inv=0.0,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    H_p2 = diffuse_fd(H0, DH_nm2_s=DH, t_end_s=T_END, dx_nm=DX)
    assert torch.allclose(H_full, H_p2, atol=1e-5)
    # Q diffuses but stays uniform (no spatial gradient on a constant)
    assert torch.allclose(Q_full, torch.full_like(H0, Q0), atol=1e-10)
    assert torch.allclose(P_full, torch.zeros_like(H0), atol=1e-10)


def test_zero_kq_zero_kloss_keeps_total_acid(H0):
    """With kq = kloss = 0 the H equation is purely diffusive and
    periodic BCs preserve total acid mass."""
    H_full, _, _ = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=0.0, kloss_ref_s_inv=0.0, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    m0 = float(H0.sum().item()) * DX * DX
    m1 = float(H_full.sum().item()) * DX * DX
    assert math.isclose(m0, m1, rel_tol=1e-5)


# ---- temperature monotonicity ------------------------------------------

def test_higher_T_consumes_more_quencher(H0):
    """At T > T_ref the kq Arrhenius factor is > 1, so the quencher
    Q gets consumed faster -> Q_min smaller, Q_mean smaller."""
    _, Q_ref, _ = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    _, Q_hot, _ = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=120.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    assert float(Q_hot.min().item()) < float(Q_ref.min().item())
    assert float(Q_hot.mean().item()) < float(Q_ref.mean().item())


def test_higher_T_increases_P(H0):
    """Same Arrhenius logic on kdep -> hotter T must give larger
    P_max and larger threshold area."""
    _, _, P_ref = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    _, _, P_hot = evolve_full_reaction_diffusion_fd_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=120.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    assert float(P_hot.max().item()) > float(P_ref.max().item())
    assert float(P_hot.mean().item()) > float(P_ref.mean().item())


# ---- mass-budget identity carries over ---------------------------------

def test_with_budget_identities_close_at_Tref(H0):
    _, _, _, hist = evolve_full_reaction_diffusion_fd_at_T_with_budget(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    last = hist[-1]
    # Both budgets must close to ~float32 precision.
    assert last.H_budget_relative_error < 1e-5
    assert last.Q_budget_relative_error < 1e-5


def test_with_budget_identities_close_at_hot_T(H0):
    _, _, _, hist = evolve_full_reaction_diffusion_fd_at_T_with_budget(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=120.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        t_end_s=T_END, dx_nm=DX,
    )
    last = hist[-1]
    assert last.H_budget_relative_error < 1e-5
    assert last.Q_budget_relative_error < 1e-5


# ---- stability report --------------------------------------------------

def test_stability_report_returns_four_dt_bounds(H0):
    rep = stability_report_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        dx_nm=DX,
    )
    for key in ("dt_diff", "dt_loss", "dt_dep", "dt_kq", "dt_max", "stiff_term"):
        assert key in rep
    assert rep["dt_max"] == min(rep["dt_diff"], rep["dt_loss"],
                                rep["dt_dep"], rep["dt_kq"])
    assert rep["stiff_term"] in {"dt_diff", "dt_loss", "dt_dep", "dt_kq"}


def test_stability_dt_kq_shrinks_with_T(H0):
    """At hot T the Arrhenius factor multiplies kq, kdep, kloss so the
    reaction-bound dt limits get smaller while diffusion stays put."""
    cold = stability_report_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=T_REF, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        dx_nm=DX,
    )
    hot = stability_report_at_T(
        H0, Q0_mol_dm3=Q0,
        DH_nm2_s=DH, DQ_nm2_s=DQ,
        kq_ref_s_inv=KQ, kloss_ref_s_inv=KLOSS, kdep_ref_s_inv=KDEP,
        temperature_c=125.0, temperature_ref_c=T_REF,
        activation_energy_kj_mol=EA,
        dx_nm=DX,
    )
    assert hot["dt_kq"] < cold["dt_kq"]
    assert hot["dt_dep"] < cold["dt_dep"]
    assert hot["dt_loss"] < cold["dt_loss"]
    assert math.isclose(hot["dt_diff"], cold["dt_diff"])

"""Tests for the Phase-7 acid-quencher reaction FD evolver."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.quencher_reaction import (
    QuencherBudgetSnapshot,
    evolve_quencher_fd,
    evolve_quencher_fd_with_budget,
    history_to_dicts,
    stability_report,
    step_quencher_fd,
    total_mass_dx2,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


def _gaussian_H0(grid_size: int = 64, sigma_px: float = 6.0,
                Hmax: float = 0.2) -> torch.Tensor:
    return acid_generation(
        gaussian_spot(grid_size, sigma_px=sigma_px),
        dose=1.0, eta=1.0, Hmax=Hmax,
    )


# ---- structural / API ---------------------------------------------------

def test_zero_time_identity():
    H0 = _gaussian_H0()
    H, Q, P = evolve_quencher_fd(
        H0, Q0_mol_dm3=0.1, DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=10.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=0.0, dx_nm=1.0,
    )
    assert torch.equal(H, H0)
    assert torch.allclose(Q, torch.full_like(H0, 0.1))
    assert torch.allclose(P, torch.zeros_like(H0))


def test_step_function_consistent_with_evolver():
    """A single-step call agrees with one iteration of the full evolver."""
    torch.manual_seed(0)
    H = torch.rand(16, 16) * 0.2
    Q = torch.full_like(H, 0.1)
    P = torch.zeros_like(H)
    DH, DQ, kq, kloss, kdep = 0.5, 0.05, 5.0, 0.005, 0.1
    dx = 1.0
    dt_max = (dx ** 2) / (4.0 * DH)
    dt = 0.1 * dt_max  # well below CFL
    direct_H, direct_Q, direct_P = step_quencher_fd(
        H, Q, P, DH, DQ, kq, kloss, kdep, dt, dx,
    )
    via_solver_H, via_solver_Q, via_solver_P = evolve_quencher_fd(
        H, Q0_mol_dm3=0.1,
        DH_nm2_s=DH, DQ_nm2_s=DQ, kq_s_inv=kq,
        kloss_s_inv=kloss, kdep_s_inv=kdep,
        t_end_s=dt, dx_nm=dx, n_steps=1,
    )
    assert torch.allclose(direct_H, via_solver_H)
    assert torch.allclose(direct_Q, via_solver_Q)
    assert torch.allclose(direct_P, via_solver_P)


def test_stability_report_picks_correct_stiff_term():
    H0 = _gaussian_H0()
    # Safe regime: dt_diff is the binding constraint.
    parts = stability_report(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=10.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        dx_nm=1.0,
    )
    assert parts["stiff_term"] == "dt_diff"
    # Stiff regime: dt_kq becomes the binding constraint.
    parts = stability_report(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=1000.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        dx_nm=1.0,
    )
    assert parts["stiff_term"] == "dt_kq"


# ---- physical contracts -------------------------------------------------

def test_kq_zero_keeps_Q_uniform_constant():
    """With ``k_q = 0`` and DQ > 0, Q stays at the initial uniform Q_0
    everywhere (diffusion of a constant field is zero)."""
    H0 = _gaussian_H0()
    Q0_value = 0.123
    _, Q, _ = evolve_quencher_fd(
        H0, Q0_mol_dm3=Q0_value,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=0.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert torch.allclose(Q, torch.full_like(H0, Q0_value), atol=1e-6)


def test_Q0_zero_matches_phase5():
    """If there is no quencher in the system, the (H, P) result should
    equal the Phase-5 evolver's result bit-for-bit."""
    H0 = _gaussian_H0()
    H_q, _, P_q = evolve_quencher_fd(
        H0, Q0_mol_dm3=0.0,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=10.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=30.0, dx_nm=1.0,
    )
    H_p5, P_p5 = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=30.0, dx_nm=1.0,
    )
    assert torch.allclose(H_q, H_p5, atol=1e-6)
    assert torch.allclose(P_q, P_p5, atol=1e-6)


def test_increasing_kq_consumes_more_H():
    """At equal Q_0 and other rates, larger ``k_q`` removes more H."""
    H0 = _gaussian_H0()
    masses_H = []
    for kq in (0.0, 1.0, 5.0, 10.0):
        H_t, _, _ = evolve_quencher_fd(
            H0, Q0_mol_dm3=0.1,
            DH_nm2_s=0.8, DQ_nm2_s=0.08,
            kq_s_inv=kq, kloss_s_inv=0.005, kdep_s_inv=0.5,
            t_end_s=60.0, dx_nm=1.0,
        )
        masses_H.append(total_mass_dx2(H_t, dx_nm=1.0))
    # Strictly decreasing in kq
    for k in range(len(masses_H) - 1):
        assert masses_H[k] > masses_H[k + 1] - 1e-9


def test_increasing_kq_consumes_more_Q():
    """At equal H_0 and other rates, larger ``k_q`` removes more Q."""
    H0 = _gaussian_H0()
    masses_Q = []
    for kq in (0.0, 1.0, 5.0, 10.0):
        _, Q_t, _ = evolve_quencher_fd(
            H0, Q0_mol_dm3=0.1,
            DH_nm2_s=0.8, DQ_nm2_s=0.08,
            kq_s_inv=kq, kloss_s_inv=0.005, kdep_s_inv=0.5,
            t_end_s=60.0, dx_nm=1.0,
        )
        masses_Q.append(total_mass_dx2(Q_t, dx_nm=1.0))
    for k in range(len(masses_Q) - 1):
        assert masses_Q[k] >= masses_Q[k + 1] - 1e-9


def test_Q_consumed_near_H_peak_not_at_corner():
    """Q drops below Q_0 where H is large; far from the spot Q stays
    near Q_0."""
    H0 = _gaussian_H0()
    Q0_value = 0.1
    _, Q_t, _ = evolve_quencher_fd(
        H0, Q0_mol_dm3=Q0_value,
        DH_nm2_s=0.0, DQ_nm2_s=0.0,        # no diffusion -> reaction only
        kq_s_inv=10.0, kloss_s_inv=0.0, kdep_s_inv=0.0,
        t_end_s=60.0, dx_nm=1.0,
    )
    n = H0.shape[-1]
    Q_center = Q_t[n // 2, n // 2].item()
    Q_corner = Q_t[0, 0].item()
    assert Q_center < Q0_value - 1e-3
    assert abs(Q_corner - Q0_value) < 1e-6


def test_H_Q_P_stay_non_negative():
    """All three fields stay >= 0 throughout the evolution."""
    H0 = _gaussian_H0()
    H_t, Q_t, P_t = evolve_quencher_fd(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=10.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0,
    )
    assert (H_t >= 0).all().item()
    assert (Q_t >= 0).all().item()
    assert (P_t >= 0).all().item()
    assert (P_t <= 1).all().item()


def test_stiff_kq_remains_finite_and_stable():
    """At ``k_q = 1000`` the explicit Euler step is dt_kq-bounded and
    the evolution must complete with finite, non-negative H, Q, P."""
    H0 = _gaussian_H0(grid_size=32, sigma_px=4.0)
    H_t, Q_t, P_t = evolve_quencher_fd(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=1000.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0,    # short evolution to keep test fast
    )
    for name, T in (("H", H_t), ("Q", Q_t), ("P", P_t)):
        assert torch.isfinite(T).all().item(), f"{name} contains non-finite values"
    assert (H_t >= 0).all().item()
    assert (Q_t >= 0).all().item()
    assert (P_t >= 0).all().item() and (P_t <= 1).all().item()


# ---- mass-budget identities ---------------------------------------------

def test_mass_budgets_hold_at_safe_kq():
    """Both M_H_budget and M_Q_budget close to float32 round-off in the
    safe regime."""
    H0 = _gaussian_H0()
    _, _, _, history = evolve_quencher_fd_with_budget(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=5.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0, n_log_points=11,
    )
    final = history[-1]
    assert final.H_budget_relative_error < 1e-5
    assert final.Q_budget_relative_error < 1e-5


def test_mass_budgets_hold_at_stiff_kq():
    """Same identity holds at kq=300 — the stiff CFL bound just makes
    each step smaller; the FD scheme's exact mass balance is unchanged.
    """
    H0 = _gaussian_H0(grid_size=32, sigma_px=4.0)
    _, _, _, history = evolve_quencher_fd_with_budget(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=300.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=6,
    )
    final = history[-1]
    assert final.H_budget_relative_error < 1e-4
    assert final.Q_budget_relative_error < 1e-4


def test_kq_zero_neutralization_integral_zero():
    H0 = _gaussian_H0()
    _, _, _, history = evolve_quencher_fd_with_budget(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=0.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0, n_log_points=5,
    )
    final = history[-1]
    assert final.quencher_neutralization_integral == 0.0
    # And the H budget reduces to the Phase-4 acid-loss form:
    assert final.H_budget_relative_error < 1e-5
    # Q is unchanged (kq=0 + diffusion of constant)
    initial_Q = history[0].mass_Q
    assert abs(final.mass_Q - initial_Q) / initial_Q < 1e-6


def test_history_length_and_dict_conversion():
    H0 = _gaussian_H0(grid_size=32)
    _, _, _, history = evolve_quencher_fd_with_budget(
        H0, Q0_mol_dm3=0.1,
        DH_nm2_s=0.8, DQ_nm2_s=0.08,
        kq_s_inv=5.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=7,
    )
    assert len(history) == 7
    dicts = history_to_dicts(history)
    assert len(dicts) == 7
    for d in dicts:
        assert {"t_s", "mass_H", "mass_Q",
                "acid_loss_integral", "quencher_neutralization_integral",
                "H_budget", "Q_budget",
                "H_budget_relative_error",
                "Q_budget_relative_error"} <= set(d.keys())


def test_invalid_args_raise():
    H0 = _gaussian_H0(grid_size=32)
    for kw in ("DH_nm2_s", "DQ_nm2_s", "kq_s_inv", "kloss_s_inv",
               "kdep_s_inv", "t_end_s"):
        kwargs = dict(DH_nm2_s=0.8, DQ_nm2_s=0.08, kq_s_inv=5.0,
                      kloss_s_inv=0.005, kdep_s_inv=0.5,
                      t_end_s=10.0, dx_nm=1.0)
        kwargs[kw] = -0.1
        try:
            evolve_quencher_fd(H0, Q0_mol_dm3=0.1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} negative should raise")
    # Q0 negative
    try:
        evolve_quencher_fd(
            H0, Q0_mol_dm3=-0.1,
            DH_nm2_s=0.8, DQ_nm2_s=0.08,
            kq_s_inv=5.0, kloss_s_inv=0.005, kdep_s_inv=0.5,
            t_end_s=10.0, dx_nm=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Q0 negative should raise")

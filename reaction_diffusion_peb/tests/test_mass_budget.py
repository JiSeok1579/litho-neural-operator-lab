"""Tests for the pre-Phase-7 mass-budget diagnostic."""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.mass_budget import (
    MassBudgetSnapshot,
    evolve_acid_loss_deprotection_fd_with_budget,
    evolve_acid_loss_deprotection_fd_with_budget_at_T,
    history_to_dicts,
    total_mass_dx2,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot


def _gaussian_H0(grid_size: int = 64, sigma_px: float = 6.0,
                Hmax: float = 0.2) -> torch.Tensor:
    return acid_generation(
        gaussian_spot(grid_size, sigma_px=sigma_px),
        dose=1.0, eta=1.0, Hmax=Hmax,
    )


# ---- structural / API ----------------------------------------------------

def test_history_starts_at_t_zero_with_zero_loss():
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=5,
    )
    assert history[0].t_s == 0.0
    assert history[0].acid_loss_integral == 0.0
    assert history[0].mass_budget_relative_error == 0.0


def test_history_ends_at_t_end():
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=5,
    )
    assert abs(history[-1].t_s - 10.0) < 1e-6


def test_history_log_points_count():
    H0 = _gaussian_H0()
    _, _, h5 = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=5,
    )
    _, _, h11 = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=10.0, dx_nm=1.0, n_log_points=11,
    )
    assert len(h5) == 5
    assert len(h11) == 11


def test_history_dict_conversion():
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=5.0, dx_nm=1.0, n_log_points=4,
    )
    dicts = history_to_dicts(history)
    assert len(dicts) == 4
    for d in dicts:
        assert {"t_s", "mass_H", "acid_loss_integral",
                "mass_budget", "mass_budget_relative_error"} <= set(d.keys())


# ---- mass-conservation contracts ----------------------------------------

def test_kloss_zero_conserves_total_mass():
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.0, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0, n_log_points=11,
    )
    m0 = history[0].mass_H
    for snap in history:
        # No loss term -> mass_H is preserved at every snapshot.
        assert abs(snap.mass_H - m0) / m0 < 1e-6
        # And the budget reduces to mass_H exactly.
        assert snap.acid_loss_integral == 0.0
        assert snap.mass_budget_relative_error < 1e-6


def test_kloss_positive_mass_decreases_but_budget_holds():
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=60.0, dx_nm=1.0, n_log_points=11,
    )
    final = history[-1]
    initial = history[0]
    # mass_H decreases over time (acid is being lost)
    assert final.mass_H < initial.mass_H
    # But the budget is conserved to floating-point precision
    assert final.mass_budget_relative_error < 1e-6


def test_high_T_arrhenius_budget_still_holds():
    """At 125 °C with Ea=100 kJ/mol the effective k_loss is ~0.038 1/s
    (vs 0.005 at T_ref). The budget identity must still hold to machine
    precision because it is a pure consequence of the FD scheme."""
    H0 = _gaussian_H0()
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget_at_T(
        H0, DH_nm2_s=0.8,
        kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=125.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=60.0, dx_nm=1.0, n_log_points=11,
    )
    # ~92% of acid is consumed at 125 °C, but budget identity holds.
    final = history[-1]
    initial = history[0]
    assert final.mass_H / initial.mass_H < 0.5   # strong decay
    assert final.mass_budget_relative_error < 1e-6


def test_budget_identity_pixelwise_kloss_zero_DH_zero():
    """Pure decay with D=0, k_loss=k tests the per-step identity:
    mass_H(n+1) + accum_loss(n+1) == mass_H_initial. With D=0 the field
    just decays exponentially per pixel, so we have an analytic check."""
    H0 = _gaussian_H0()
    kloss = 0.05
    _, _, history = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.0, kloss_s_inv=kloss, kdep_s_inv=0.0,
        t_end_s=10.0, dx_nm=1.0, n_log_points=11,
    )
    # At each snapshot, mass_H + accum_loss should equal the initial mass.
    initial = history[0].mass_H
    for snap in history:
        budget = snap.mass_H + snap.acid_loss_integral
        assert abs(budget - initial) / initial < 1e-6


def test_arrhenius_t_ref_matches_no_arrhenius_path():
    """At T = T_ref the Arrhenius wrapper must reproduce the plain
    evolver bit-for-bit (factor = 1)."""
    H0 = _gaussian_H0()
    _, _, h_arr = evolve_acid_loss_deprotection_fd_with_budget_at_T(
        H0, DH_nm2_s=0.8,
        kdep_ref_s_inv=0.5, kloss_ref_s_inv=0.005,
        temperature_c=100.0, temperature_ref_c=100.0,
        activation_energy_kj_mol=100.0,
        t_end_s=20.0, dx_nm=1.0, n_log_points=5,
    )
    _, _, h_plain = evolve_acid_loss_deprotection_fd_with_budget(
        H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
        t_end_s=20.0, dx_nm=1.0, n_log_points=5,
    )
    for a, b in zip(h_arr, h_plain):
        assert abs(a.t_s - b.t_s) < 1e-9
        assert abs(a.mass_H - b.mass_H) / max(b.mass_H, 1e-12) < 1e-9
        assert abs(a.mass_budget_relative_error
                   - b.mass_budget_relative_error) < 1e-9


# ---- helper sanity ------------------------------------------------------

def test_total_mass_dx2_matches_sum_times_area():
    H = torch.full((4, 4), 0.5)
    expected = 16 * 0.5 * (2.0 * 2.0)  # 16 pixels at 0.5 with dx=2 -> dx^2=4
    assert abs(total_mass_dx2(H, dx_nm=2.0) - expected) < 1e-9


def test_invalid_args_raise():
    H0 = _gaussian_H0()
    for kw in ("DH_nm2_s", "kloss_s_inv", "kdep_s_inv", "t_end_s"):
        kwargs = dict(DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
                      t_end_s=10.0, dx_nm=1.0)
        kwargs[kw] = -0.1
        try:
            evolve_acid_loss_deprotection_fd_with_budget(H0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{kw} negative should raise")
    # n_log_points too small
    try:
        evolve_acid_loss_deprotection_fd_with_budget(
            H0, DH_nm2_s=0.8, kloss_s_inv=0.005, kdep_s_inv=0.5,
            t_end_s=10.0, dx_nm=1.0, n_log_points=1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("n_log_points=1 should raise")

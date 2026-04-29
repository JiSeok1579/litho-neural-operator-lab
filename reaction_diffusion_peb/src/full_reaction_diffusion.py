"""Phase-8 full reaction-diffusion model.

Combines every term added so far in the submodule:

    dH/dt = D_H * laplacian(H) - k_q(T) * H * Q - k_loss(T) * H
    dQ/dt = D_Q * laplacian(Q) - k_q(T) * H * Q
    dP/dt = k_dep(T) * H * (1 - P)

where ``k_q``, ``k_loss`` and ``k_dep`` are the reference rates scaled
by the Arrhenius factor

    factor(T) = exp(-Ea/R * (1/T_K - 1/T_ref_K))

Initial conditions:

    H(x, y, 0) = H_0(x, y)
    Q(x, y, 0) = Q_0                       (uniform scalar)
    P(x, y, 0) = 0

This file does not introduce any new physics — it is a thin wrapper
that applies the Phase-6 Arrhenius scaling to the Phase-7 quencher
evolver. The point of Phase 8 is verifying the integrated model
reduces to each earlier phase when the corresponding term is disabled:

    kq = 0                         -> Phase 6 (Arrhenius (H, P))
    kq = 0, Ea = 0 or T = T_ref    -> Phase 5 (deprotection)
    kq = 0, kdep = 0               -> Phase 4 (acid loss)
    kq = 0, kdep = 0, kloss = 0    -> Phase 2 (diffusion only)
    Ea = 0 or T = T_ref            -> Phase 7 (quencher, no T)
"""

from __future__ import annotations

import torch

from reaction_diffusion_peb.src.arrhenius import arrhenius_factor
from reaction_diffusion_peb.src.quencher_reaction import (
    QuencherBudgetSnapshot,
    evolve_quencher_fd,
    evolve_quencher_fd_with_budget,
    stability_report,
)


def apply_arrhenius_to_full_rates(
    kdep_ref_s_inv: float,
    kloss_ref_s_inv: float,
    kq_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
) -> tuple[float, float, float]:
    """Return ``(k_dep(T), k_loss(T), k_q(T))`` after Arrhenius scaling.

    All three reaction rates are multiplied by the same factor — the
    Phase-6 single-Ea assumption carried forward to Phase 8. A future
    extension could give each rate its own activation energy.
    """
    if kdep_ref_s_inv < 0:
        raise ValueError("kdep_ref_s_inv must be non-negative")
    if kloss_ref_s_inv < 0:
        raise ValueError("kloss_ref_s_inv must be non-negative")
    if kq_ref_s_inv < 0:
        raise ValueError("kq_ref_s_inv must be non-negative")
    factor = arrhenius_factor(
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return (
        kdep_ref_s_inv * factor,
        kloss_ref_s_inv * factor,
        kq_ref_s_inv * factor,
    )


def evolve_full_reaction_diffusion_fd_at_T(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_ref_s_inv: float,
    kloss_ref_s_inv: float,
    kdep_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Arrhenius to ``kq, kloss, kdep`` and call the Phase-7 evolver.

    Returns ``(H_final, Q_final, P_final)``.
    """
    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return evolve_quencher_fd(
        H0,
        Q0_mol_dm3=Q0_mol_dm3,
        DH_nm2_s=DH_nm2_s, DQ_nm2_s=DQ_nm2_s,
        kq_s_inv=kq_eff, kloss_s_inv=kloss_eff, kdep_s_inv=kdep_eff,
        t_end_s=t_end_s, dx_nm=dx_nm,
        n_steps=n_steps, cfl_safety=cfl_safety,
    )


def evolve_full_reaction_diffusion_fd_at_T_with_budget(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_ref_s_inv: float,
    kloss_ref_s_inv: float,
    kdep_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
    n_log_points: int = 21,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[QuencherBudgetSnapshot]]:
    """Same as :func:`evolve_full_reaction_diffusion_fd_at_T` but also
    returns the per-time-snapshot mass-budget history for both H and Q.
    """
    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return evolve_quencher_fd_with_budget(
        H0,
        Q0_mol_dm3=Q0_mol_dm3,
        DH_nm2_s=DH_nm2_s, DQ_nm2_s=DQ_nm2_s,
        kq_s_inv=kq_eff, kloss_s_inv=kloss_eff, kdep_s_inv=kdep_eff,
        t_end_s=t_end_s, dx_nm=dx_nm,
        n_steps=n_steps, cfl_safety=cfl_safety,
        n_log_points=n_log_points,
    )


def stability_report_at_T(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_ref_s_inv: float,
    kloss_ref_s_inv: float,
    kdep_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    dx_nm: float,
) -> dict:
    """Return the four stability time-step bounds with the
    Arrhenius-corrected rates plugged in. Useful for telling whether
    a hot-temperature run will be diffusion-bound or reaction-bound.
    """
    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return stability_report(
        H0, Q0_mol_dm3=Q0_mol_dm3,
        DH_nm2_s=DH_nm2_s, DQ_nm2_s=DQ_nm2_s,
        kq_s_inv=kq_eff, kloss_s_inv=kloss_eff, kdep_s_inv=kdep_eff,
        dx_nm=dx_nm,
    )

"""Phase 11 — Petersen nonlinear diffusion + variable-coefficient FD.

Replaces the constant ``D_H`` in the Phase-8 integrated evolver with the
Petersen form

    D_H(x, y, t) = D_H0 * exp(alpha * P(x, y, t))

Diffusion of the acid field then becomes

    dH/dt = ∇·(D_H(P) ∇H) - k_q H Q - k_loss H

while ``Q`` keeps its constant-coefficient diffusion (the plan only
introduces nonlinearity through the deprotection-coupled acid mobility).

Naming caveat (from the plan): the original "DH = DH0 exp(α D)" form
is ambiguous because ``D`` could mean dose, diffusion coefficient, or
deprotected fraction. This module uses ``P`` (deprotected fraction)
as the coupling variable.

The variable-coefficient 5-point operator uses arithmetic half-cell
averages for the face diffusivity:

    [∇·(D ∇H)]_{i,j} ≈ (1 / dx^2) * [
        D_{i+1/2, j} (H_{i+1, j} - H_{i,   j})
      - D_{i-1/2, j} (H_{i,   j} - H_{i-1, j})
      + D_{i, j+1/2} (H_{i, j+1} - H_{i, j  })
      - D_{i, j-1/2} (H_{i, j  } - H_{i, j-1})
    ]

with periodic boundary conditions via ``torch.roll``. When ``D`` is
spatially constant, this collapses to ``D * laplacian_5pt(H)`` so the
``alpha = 0`` case is bit-identical to the Phase-2 / Phase-8
constant-coefficient evolver.

Stability — the diffusion CFL bound becomes

    dt <= dx^2 / (4 * D_H_max),
    D_H_max = D_H0 * exp(alpha) at the worst case (P -> 1).

We use this conservative bound rather than the per-step ``D_max`` so
the step size does not change as ``P`` evolves.
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.arrhenius import arrhenius_factor
from reaction_diffusion_peb.src.diffusion_fd import laplacian_5pt
from reaction_diffusion_peb.src.full_reaction_diffusion import (
    apply_arrhenius_to_full_rates,
)
from reaction_diffusion_peb.src.quencher_reaction import (
    QuencherBudgetSnapshot,
    total_mass_dx2,
)


# --------------------------------------------------------------------------
# variable-coefficient diffusion operator
# --------------------------------------------------------------------------

def divergence_diffusion_5pt(
    H: torch.Tensor, D: torch.Tensor, dx_nm: float,
) -> torch.Tensor:
    """``∇·(D ∇H)`` discretized with arithmetic half-cell averages.

    ``H`` and ``D`` must broadcast to the same shape. Periodic
    boundaries are handled by ``torch.roll``. When ``D`` is constant
    this returns ``D * laplacian_5pt(H, dx_nm)`` exactly.
    """
    if D.shape != H.shape and D.numel() != 1:
        raise ValueError("D must be a scalar or have the same shape as H")
    inv_dx2 = 1.0 / (dx_nm * dx_nm)
    # Face diffusivities (arithmetic mean of cell-center values).
    D_xp = 0.5 * (D + torch.roll(D, shifts=-1, dims=-1))
    D_xm = 0.5 * (D + torch.roll(D, shifts=+1, dims=-1))
    D_yp = 0.5 * (D + torch.roll(D, shifts=-1, dims=-2))
    D_ym = 0.5 * (D + torch.roll(D, shifts=+1, dims=-2))
    # Field differences across each face.
    H_xp = torch.roll(H, shifts=-1, dims=-1) - H
    H_xm = H - torch.roll(H, shifts=+1, dims=-1)
    H_yp = torch.roll(H, shifts=-1, dims=-2) - H
    H_ym = H - torch.roll(H, shifts=+1, dims=-2)
    return inv_dx2 * (
        D_xp * H_xp - D_xm * H_xm + D_yp * H_yp - D_ym * H_ym
    )


def petersen_DH_field(P: torch.Tensor, DH0_nm2_s: float,
                      alpha: float) -> torch.Tensor:
    """Petersen acid-mobility field ``D_H(P) = D_H0 * exp(alpha * P)``."""
    if DH0_nm2_s < 0:
        raise ValueError("DH0_nm2_s must be non-negative")
    return DH0_nm2_s * torch.exp(alpha * P)


# --------------------------------------------------------------------------
# Petersen + quencher FD step
# --------------------------------------------------------------------------

def step_petersen_full_fd(
    H: torch.Tensor,
    Q: torch.Tensor,
    P: torch.Tensor,
    DH0_nm2_s: float,
    petersen_alpha: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    dt_s: float,
    dx_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One explicit Euler step of the Petersen-coupled (H, Q, P) system.

    The acid diffusion uses the ``D_H(P)`` field; the quencher
    diffusion stays constant-coefficient at ``DQ_nm2_s``.
    """
    D_H = petersen_DH_field(P, DH0_nm2_s, petersen_alpha)
    HQ = H * Q
    div_DH_grad_H = divergence_diffusion_5pt(H, D_H, dx_nm)
    lap_Q = laplacian_5pt(Q, dx_nm)
    H_new = H + dt_s * (div_DH_grad_H - kq_s_inv * HQ - kloss_s_inv * H)
    Q_new = Q + dt_s * (DQ_nm2_s * lap_Q - kq_s_inv * HQ)
    P_new = P + dt_s * (kdep_s_inv * H * (1.0 - P))
    return (
        H_new.clamp(min=0.0),
        Q_new.clamp(min=0.0),
        P_new.clamp(min=0.0, max=1.0),
    )


# --------------------------------------------------------------------------
# stability bound
# --------------------------------------------------------------------------

def _compute_dt_max_petersen(
    DH0_nm2_s: float, petersen_alpha: float,
    DQ_nm2_s: float,
    kq_s_inv: float, kloss_s_inv: float, kdep_s_inv: float,
    H0_max: float, Q0_max: float, dx_nm: float,
) -> tuple[float, dict]:
    """Conservative stability bound — uses ``D_H_max = D_H0 * exp(alpha)``
    so the step does not need to shrink as ``P`` rises."""
    D_H_max = DH0_nm2_s * math.exp(max(petersen_alpha, 0.0))
    Dmax = max(D_H_max, DQ_nm2_s)
    dt_diff = (dx_nm ** 2) / (4.0 * Dmax) if Dmax > 0 else float("inf")
    dt_loss = (1.0 / kloss_s_inv) if kloss_s_inv > 0 else float("inf")
    dt_dep = (1.0 / (kdep_s_inv * max(H0_max, 1e-12))) if kdep_s_inv > 0 else float("inf")
    dt_kq = (1.0 / (kq_s_inv * max(H0_max, Q0_max, 1e-12))) if kq_s_inv > 0 else float("inf")
    dt_max = min(dt_diff, dt_loss, dt_dep, dt_kq)
    parts = {
        "dt_diff": dt_diff,
        "dt_loss": dt_loss,
        "dt_dep": dt_dep,
        "dt_kq": dt_kq,
        "dt_max": dt_max,
        "DH_max": D_H_max,
        "stiff_term": min(
            ("dt_diff", dt_diff), ("dt_loss", dt_loss),
            ("dt_dep", dt_dep), ("dt_kq", dt_kq),
            key=lambda kv: kv[1],
        )[0],
    }
    return dt_max, parts


# --------------------------------------------------------------------------
# evolver
# --------------------------------------------------------------------------

def evolve_petersen_full_fd_at_T(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH0_nm2_s: float,
    petersen_alpha: float,
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
    """Phase-8 integrated FD evolver with Petersen nonlinear diffusion.

    Returns ``(H_final, Q_final, P_final)``. When ``petersen_alpha = 0``
    the result is bit-identical to
    :func:`evolve_full_reaction_diffusion_fd_at_T` from Phase 8.
    """
    if petersen_alpha < 0:
        raise ValueError("petersen_alpha must be non-negative")
    for name, val in (("DH0_nm2_s", DH0_nm2_s), ("DQ_nm2_s", DQ_nm2_s),
                      ("kq_ref_s_inv", kq_ref_s_inv),
                      ("kloss_ref_s_inv", kloss_ref_s_inv),
                      ("kdep_ref_s_inv", kdep_ref_s_inv),
                      ("t_end_s", t_end_s)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
    if Q0_mol_dm3 < 0:
        raise ValueError("Q0_mol_dm3 must be non-negative")

    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )

    Q0 = torch.full_like(H0, float(Q0_mol_dm3))
    P0 = torch.zeros_like(H0)
    if t_end_s == 0:
        return H0.clone(), Q0, P0

    H0_max = float(H0.abs().max().item())
    dt_max, _ = _compute_dt_max_petersen(
        DH0_nm2_s=DH0_nm2_s, petersen_alpha=petersen_alpha,
        DQ_nm2_s=DQ_nm2_s,
        kq_s_inv=kq_eff, kloss_s_inv=kloss_eff, kdep_s_inv=kdep_eff,
        H0_max=H0_max, Q0_max=Q0_mol_dm3, dx_nm=dx_nm,
    )
    if n_steps is None:
        dt = cfl_safety * dt_max
        n_steps = max(1, int(math.ceil(t_end_s / dt)))
        dt = t_end_s / n_steps
    else:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        dt = t_end_s / n_steps
        if dt > dt_max + 1e-12:
            raise ValueError(
                f"FD step dt={dt:g} s violates stability limit {dt_max:g} s"
            )

    H, Q, P = H0.clone(), Q0, P0
    for _ in range(n_steps):
        H, Q, P = step_petersen_full_fd(
            H, Q, P, DH0_nm2_s, petersen_alpha, DQ_nm2_s,
            kq_eff, kloss_eff, kdep_eff, dt, dx_nm,
        )
    return H, Q, P


def evolve_petersen_full_fd_at_T_with_budget(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH0_nm2_s: float,
    petersen_alpha: float,
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
    """Same as :func:`evolve_petersen_full_fd_at_T` but additionally
    returns the per-time-snapshot mass-budget history.

    The variable-coefficient diffusion conserves total ``H`` for
    constant-coefficient periodic BCs (the discretization is
    flux-form), so the H mass budget closes to float-precision exactly
    like Phase 8 even with ``alpha > 0``.
    """
    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    if t_end_s < 0 or DH0_nm2_s < 0 or DQ_nm2_s < 0:
        raise ValueError("rates / D / time must be non-negative")
    if petersen_alpha < 0:
        raise ValueError("petersen_alpha must be non-negative")
    if n_log_points < 2:
        raise ValueError("n_log_points must be >= 2 (start + end at minimum)")

    Q0 = torch.full_like(H0, float(Q0_mol_dm3))
    mass_H_initial = total_mass_dx2(H0, dx_nm)
    mass_Q_initial = total_mass_dx2(Q0, dx_nm)
    H0_max = float(H0.abs().max().item())

    if t_end_s == 0:
        snap = QuencherBudgetSnapshot(
            t_s=0.0,
            mass_H=mass_H_initial, mass_Q=mass_Q_initial,
            acid_loss_integral=0.0,
            quencher_neutralization_integral=0.0,
            H_budget=mass_H_initial, Q_budget=mass_Q_initial,
            H_budget_relative_error=0.0, Q_budget_relative_error=0.0,
        )
        return H0.clone(), Q0, torch.zeros_like(H0), [snap]

    dt_max, _ = _compute_dt_max_petersen(
        DH0_nm2_s, petersen_alpha, DQ_nm2_s,
        kq_eff, kloss_eff, kdep_eff,
        H0_max, Q0_mol_dm3, dx_nm,
    )
    if n_steps is None:
        dt = cfl_safety * dt_max
        n_steps = max(1, int(math.ceil(t_end_s / dt)))
        dt = t_end_s / n_steps
    else:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        dt = t_end_s / n_steps
        if dt > dt_max + 1e-12:
            raise ValueError(
                f"FD step dt={dt:g} s violates stability limit {dt_max:g} s"
            )

    log_idx = sorted(set(
        [0, n_steps]
        + [int(round(k * n_steps / (n_log_points - 1)))
           for k in range(1, n_log_points - 1)]
    ))

    H = H0.clone()
    Q = Q0.clone()
    P = torch.zeros_like(H0)
    accum_loss = 0.0
    accum_neutr = 0.0
    history: list[QuencherBudgetSnapshot] = []
    history.append(QuencherBudgetSnapshot(
        t_s=0.0,
        mass_H=mass_H_initial, mass_Q=mass_Q_initial,
        acid_loss_integral=0.0,
        quencher_neutralization_integral=0.0,
        H_budget=mass_H_initial, Q_budget=mass_Q_initial,
        H_budget_relative_error=0.0, Q_budget_relative_error=0.0,
    ))
    for n in range(n_steps):
        accum_loss += dt * kloss_eff * total_mass_dx2(H, dx_nm)
        accum_neutr += dt * kq_eff * float((H * Q).sum().item()) * (dx_nm * dx_nm)
        H, Q, P = step_petersen_full_fd(
            H, Q, P, DH0_nm2_s, petersen_alpha, DQ_nm2_s,
            kq_eff, kloss_eff, kdep_eff, dt, dx_nm,
        )
        if (n + 1) in log_idx:
            t_now = (n + 1) * dt
            mH = total_mass_dx2(H, dx_nm)
            mQ = total_mass_dx2(Q, dx_nm)
            H_budget = mH + accum_loss + accum_neutr
            Q_budget = mQ + accum_neutr
            H_rel = abs(H_budget - mass_H_initial) / max(mass_H_initial, 1e-12)
            Q_rel = abs(Q_budget - mass_Q_initial) / max(mass_Q_initial, 1e-12)
            history.append(QuencherBudgetSnapshot(
                t_s=t_now,
                mass_H=mH, mass_Q=mQ,
                acid_loss_integral=accum_loss,
                quencher_neutralization_integral=accum_neutr,
                H_budget=H_budget, Q_budget=Q_budget,
                H_budget_relative_error=H_rel,
                Q_budget_relative_error=Q_rel,
            ))
    return H, Q, P, history


def stability_report_petersen(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH0_nm2_s: float,
    petersen_alpha: float,
    DQ_nm2_s: float,
    kq_ref_s_inv: float,
    kloss_ref_s_inv: float,
    kdep_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    dx_nm: float,
) -> dict:
    """Return the four stability-step bounds for the Petersen evolver,
    with the Arrhenius-corrected rates plugged in. Useful for telling
    when the ``D_H_max = D_H0 * exp(alpha)`` term starts to dominate."""
    H0_max = float(H0.abs().max().item())
    kdep_eff, kloss_eff, kq_eff = apply_arrhenius_to_full_rates(
        kdep_ref_s_inv=kdep_ref_s_inv,
        kloss_ref_s_inv=kloss_ref_s_inv,
        kq_ref_s_inv=kq_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    _, parts = _compute_dt_max_petersen(
        DH0_nm2_s, petersen_alpha, DQ_nm2_s,
        kq_eff, kloss_eff, kdep_eff,
        H0_max, Q0_mol_dm3, dx_nm,
    )
    return parts

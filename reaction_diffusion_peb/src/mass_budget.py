"""Mass-budget / loss-budget diagnostic for the (H, P) FD evolver.

For the Phase 5 / Phase 6 system

    dH/dt = D_H * laplacian(H) - k_loss * H
    dP/dt = k_dep * H * (1 - P)

P does not consume H (no quencher yet), so the only acid sink is the
``k_loss`` term. This gives a closed budget identity:

    M_budget(t) = total_mass(H(t))
                  + integral_0^t  k_loss * total_mass(H(tau)) dtau
                  ≈ total_mass(H_0)                        (closed system)

For an explicit-Euler FD step ``H_{n+1} = H_n + dt * (D laplacian(H_n)
- k_loss H_n)`` with periodic boundary conditions, this identity is
satisfied to machine precision:

    sum(H_{n+1}) - sum(H_n) = -dt * k_loss * sum(H_n)

so the running budget tracks ``mass_initial`` exactly. Any drift from
``mass_initial`` flags a bug in the evolver — that is the diagnostic.

This module also extends to the Arrhenius case by treating ``k_loss``
as the **effective** rate ``k_loss_ref * arrhenius_factor(T)``. The
caller is responsible for passing the effective rate in.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from reaction_diffusion_peb.src.deprotection import (
    step_acid_loss_deprotection_fd,
)


def total_mass_dx2(H: torch.Tensor, dx_nm: float) -> float:
    """``sum(H) * dx^2`` — total acid in (mol/dm^3) * nm^2."""
    return float(H.sum().item()) * (dx_nm * dx_nm)


@dataclass
class MassBudgetSnapshot:
    """One row of the mass-budget history."""

    t_s: float
    mass_H: float
    acid_loss_integral: float
    mass_budget: float
    mass_budget_relative_error: float


def evolve_acid_loss_deprotection_fd_with_budget(
    H0: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
    n_log_points: int = 21,
) -> tuple[torch.Tensor, torch.Tensor, list[MassBudgetSnapshot]]:
    """Same evolver as Phase 5's ``evolve_acid_loss_deprotection_fd`` but
    additionally returns a per-time-snapshot mass-budget history.

    Args:
        ... same as the Phase-5 evolver ...
        n_log_points: number of (t, mass) snapshots to retain (the
            initial ``t = 0`` is always one of them; the final
            ``t = t_end`` is also forced to be one of them).

    Returns:
        ``(H_final, P_final, history)`` where history is a list of
        :class:`MassBudgetSnapshot` evenly spaced through the run.
    """
    for name, val in (("DH_nm2_s", DH_nm2_s), ("kloss_s_inv", kloss_s_inv),
                      ("kdep_s_inv", kdep_s_inv), ("t_end_s", t_end_s)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
    if n_log_points < 2:
        raise ValueError("n_log_points must be >= 2 (start + end at minimum)")

    mass_initial = total_mass_dx2(H0, dx_nm)

    if t_end_s == 0:
        snap = MassBudgetSnapshot(
            t_s=0.0, mass_H=mass_initial,
            acid_loss_integral=0.0,
            mass_budget=mass_initial,
            mass_budget_relative_error=0.0,
        )
        return H0.clone(), torch.zeros_like(H0), [snap]

    # --- step-size selection (same logic as the Phase-5 evolver) -----
    dt_diff = (dx_nm * dx_nm) / (4.0 * DH_nm2_s) if DH_nm2_s > 0 else float("inf")
    dt_loss = (1.0 / kloss_s_inv) if kloss_s_inv > 0 else float("inf")
    H0_max = float(H0.abs().max().item())
    dt_dep = (1.0 / (kdep_s_inv * max(H0_max, 1e-12))) if kdep_s_inv > 0 else float("inf")
    dt_max = min(dt_diff, dt_loss, dt_dep)

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
                f"FD step dt={dt:g} s violates stability limit "
                f"min(dx^2/(4D), 1/k_loss, 1/(k_dep H_max)) = {dt_max:g} s"
            )

    # --- log-step selection: include 0 and n_steps explicitly --------
    log_idx = sorted(set(
        [0, n_steps]
        + [int(round(k * n_steps / (n_log_points - 1)))
           for k in range(1, n_log_points - 1)]
    ))

    H = H0.clone()
    P = torch.zeros_like(H0)
    accum_loss = 0.0

    history: list[MassBudgetSnapshot] = []
    history.append(MassBudgetSnapshot(
        t_s=0.0,
        mass_H=mass_initial,
        acid_loss_integral=0.0,
        mass_budget=mass_initial,
        mass_budget_relative_error=0.0,
    ))

    for n in range(n_steps):
        # Loss "incurred during step n" uses H_n (i.e. the current H,
        # before the step is applied). This is exactly the FD scheme's
        # mass change per step.
        accum_loss += dt * kloss_s_inv * total_mass_dx2(H, dx_nm)
        # Take the step
        H, P = step_acid_loss_deprotection_fd(
            H, P, DH_nm2_s, kloss_s_inv, kdep_s_inv, dt, dx_nm,
        )
        if (n + 1) in log_idx:
            t_now = (n + 1) * dt
            mass_now = total_mass_dx2(H, dx_nm)
            budget = mass_now + accum_loss
            rel_err = abs(budget - mass_initial) / max(mass_initial, 1e-12)
            history.append(MassBudgetSnapshot(
                t_s=t_now,
                mass_H=mass_now,
                acid_loss_integral=accum_loss,
                mass_budget=budget,
                mass_budget_relative_error=rel_err,
            ))

    return H, P, history


def evolve_acid_loss_deprotection_fd_with_budget_at_T(
    H0: torch.Tensor,
    DH_nm2_s: float,
    kdep_ref_s_inv: float,
    kloss_ref_s_inv: float,
    temperature_c: float,
    temperature_ref_c: float,
    activation_energy_kj_mol: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
    n_log_points: int = 21,
) -> tuple[torch.Tensor, torch.Tensor, list[MassBudgetSnapshot]]:
    """Arrhenius-aware wrapper.

    Applies the Arrhenius correction to ``k_loss`` and ``k_dep`` and
    then calls :func:`evolve_acid_loss_deprotection_fd_with_budget`.
    The mass-budget identity holds for the **effective** rates.
    """
    # Local import to avoid a top-level cycle.
    from reaction_diffusion_peb.src.arrhenius import apply_arrhenius_to_rates

    kdep_eff, kloss_eff = apply_arrhenius_to_rates(
        kdep_ref_s_inv, kloss_ref_s_inv,
        temperature_c=temperature_c,
        temperature_ref_c=temperature_ref_c,
        activation_energy_kj_mol=activation_energy_kj_mol,
    )
    return evolve_acid_loss_deprotection_fd_with_budget(
        H0,
        DH_nm2_s=DH_nm2_s,
        kloss_s_inv=kloss_eff,
        kdep_s_inv=kdep_eff,
        t_end_s=t_end_s,
        dx_nm=dx_nm,
        n_steps=n_steps,
        cfl_safety=cfl_safety,
        n_log_points=n_log_points,
    )


def history_to_dicts(history: list[MassBudgetSnapshot]) -> list[dict]:
    """Convert ``MassBudgetSnapshot`` list to a list of plain dicts —
    convenient for CSV writers and tests."""
    return [
        {
            "t_s": s.t_s,
            "mass_H": s.mass_H,
            "acid_loss_integral": s.acid_loss_integral,
            "mass_budget": s.mass_budget,
            "mass_budget_relative_error": s.mass_budget_relative_error,
        }
        for s in history
    ]

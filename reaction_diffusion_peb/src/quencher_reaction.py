"""Phase-7 acid-quencher reaction.

Adds a quencher field ``Q`` and a bimolecular acid-quencher
neutralization to the (H, P) system from Phase 5/6:

    dH/dt = D_H * laplacian(H) - k_q * H * Q - k_loss * H
    dQ/dt = D_Q * laplacian(Q) - k_q * H * Q
    dP/dt = k_dep * H * (1 - P)

Initial conditions:

    H(x, y, 0) = H_0(x, y)
    Q(x, y, 0) = Q_0                       (uniform scalar)
    P(x, y, 0) = 0

The system is **nonlinear** through the ``H * Q`` and ``H * (1 - P)``
products, so there is no closed-form FFT reference. FD with explicit
Euler is the truth.

Stability — explicit Euler needs

    dt <= dx^2 / (4 * max(D_H, D_Q))     (diffusion CFL)
    dt <= 1 / k_loss                     (linear loss)
    dt <= 1 / (k_dep * H_max)            (deprotection)
    dt <= 1 / (k_q   * max(H_max, Q_0))  (acid-quencher reaction)

The last bound is the **stiff** one: at ``k_q = 1000 1/s`` with
``H_max ≈ 0.2`` the reaction step has to be smaller than ~5 ms,
so a 60 s evolution needs ~12 000 explicit steps. Tracked via
``cfl_safety`` and exposed cleanly so the demo can show the safe
(``k_q in [1, 10]``) and stiff (``k_q in [100, 1000]``) regimes
separately.

Mass-budget identities (closed-system, periodic BCs):

    M_H_budget(t) = total(H)(t)
                  + integral_0^t k_loss * total(H)(τ) dτ
                  + integral_0^t k_q * sum(H * Q)(τ) * dx^2 dτ
                ≈ total(H_0)

    M_Q_budget(t) = total(Q)(t)
                  + integral_0^t k_q * sum(H * Q)(τ) * dx^2 dτ
                ≈ total(Q_0)

For the FD scheme each identity holds to machine precision because
the per-step mass change is exactly the integrand we accumulate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from reaction_diffusion_peb.src.diffusion_fd import laplacian_5pt


def step_quencher_fd(
    H: torch.Tensor,
    Q: torch.Tensor,
    P: torch.Tensor,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    dt_s: float,
    dx_nm: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One explicit Euler step of the (H, Q, P) system.

    H, Q, P are clamped to non-negative (and P to ``<= 1``) after the
    update — explicit Euler can dip slightly outside the physical band
    near the stability limit, and the analytic solution is bounded.
    """
    HQ = H * Q
    lap_H = laplacian_5pt(H, dx_nm)
    lap_Q = laplacian_5pt(Q, dx_nm)
    H_new = H + dt_s * (DH_nm2_s * lap_H - kq_s_inv * HQ - kloss_s_inv * H)
    Q_new = Q + dt_s * (DQ_nm2_s * lap_Q - kq_s_inv * HQ)
    P_new = P + dt_s * (kdep_s_inv * H * (1.0 - P))
    H_new = H_new.clamp(min=0.0)
    Q_new = Q_new.clamp(min=0.0)
    P_new = P_new.clamp(min=0.0, max=1.0)
    return H_new, Q_new, P_new


def _compute_dt_max(
    DH_nm2_s: float, DQ_nm2_s: float,
    kq_s_inv: float, kloss_s_inv: float, kdep_s_inv: float,
    H0_max: float, Q0_max: float, dx_nm: float,
) -> tuple[float, dict]:
    """Compute the explicit-Euler stability bound and report the
    individual term limits for diagnostics."""
    Dmax = max(DH_nm2_s, DQ_nm2_s)
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
        "stiff_term": min(
            ("dt_diff", dt_diff), ("dt_loss", dt_loss),
            ("dt_dep", dt_dep), ("dt_kq", dt_kq),
            key=lambda kv: kv[1],
        )[0],
    }
    return dt_max, parts


def evolve_quencher_fd(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evolve (H, Q, P) for ``t_end_s`` seconds. Returns the three final
    fields. ``Q`` starts uniform at ``Q0_mol_dm3``; ``P`` starts at 0.
    """
    for name, val in (("DH_nm2_s", DH_nm2_s), ("DQ_nm2_s", DQ_nm2_s),
                      ("kq_s_inv", kq_s_inv), ("kloss_s_inv", kloss_s_inv),
                      ("kdep_s_inv", kdep_s_inv), ("t_end_s", t_end_s)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
    if Q0_mol_dm3 < 0:
        raise ValueError("Q0_mol_dm3 must be non-negative")

    Q0 = torch.full_like(H0, float(Q0_mol_dm3))
    P0 = torch.zeros_like(H0)
    if t_end_s == 0:
        return H0.clone(), Q0, P0

    H0_max = float(H0.abs().max().item())
    dt_max, _ = _compute_dt_max(
        DH_nm2_s=DH_nm2_s, DQ_nm2_s=DQ_nm2_s,
        kq_s_inv=kq_s_inv, kloss_s_inv=kloss_s_inv, kdep_s_inv=kdep_s_inv,
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
        H, Q, P = step_quencher_fd(
            H, Q, P, DH_nm2_s, DQ_nm2_s, kq_s_inv, kloss_s_inv,
            kdep_s_inv, dt, dx_nm,
        )
    return H, Q, P


# ---- mass-budget tracking ----------------------------------------------

@dataclass
class QuencherBudgetSnapshot:
    """One row of the quencher mass-budget history."""

    t_s: float
    mass_H: float
    mass_Q: float
    acid_loss_integral: float        # ∫ k_loss * total(H) dτ
    quencher_neutralization_integral: float   # ∫ k_q * sum(H*Q) * dx^2 dτ
    H_budget: float
    Q_budget: float
    H_budget_relative_error: float
    Q_budget_relative_error: float


def total_mass_dx2(H: torch.Tensor, dx_nm: float) -> float:
    return float(H.sum().item()) * (dx_nm * dx_nm)


def evolve_quencher_fd_with_budget(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
    n_log_points: int = 21,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[QuencherBudgetSnapshot]]:
    """Same as :func:`evolve_quencher_fd` but additionally returns a
    per-time-snapshot mass-budget history for both H and Q sinks."""
    for name, val in (("DH_nm2_s", DH_nm2_s), ("DQ_nm2_s", DQ_nm2_s),
                      ("kq_s_inv", kq_s_inv), ("kloss_s_inv", kloss_s_inv),
                      ("kdep_s_inv", kdep_s_inv), ("t_end_s", t_end_s)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
    if Q0_mol_dm3 < 0:
        raise ValueError("Q0_mol_dm3 must be non-negative")
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

    dt_max, _ = _compute_dt_max(
        DH_nm2_s, DQ_nm2_s, kq_s_inv, kloss_s_inv, kdep_s_inv,
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
        # Sinks during step n use H_n, Q_n (state BEFORE the step is applied).
        accum_loss += dt * kloss_s_inv * total_mass_dx2(H, dx_nm)
        accum_neutr += dt * kq_s_inv * float((H * Q).sum().item()) * (dx_nm * dx_nm)
        # Take the step
        H, Q, P = step_quencher_fd(
            H, Q, P, DH_nm2_s, DQ_nm2_s,
            kq_s_inv, kloss_s_inv, kdep_s_inv, dt, dx_nm,
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


def history_to_dicts(
    history: list[QuencherBudgetSnapshot],
) -> list[dict]:
    """Convert ``QuencherBudgetSnapshot`` list to a list of plain dicts."""
    return [
        {
            "t_s": s.t_s,
            "mass_H": s.mass_H,
            "mass_Q": s.mass_Q,
            "acid_loss_integral": s.acid_loss_integral,
            "quencher_neutralization_integral": s.quencher_neutralization_integral,
            "H_budget": s.H_budget,
            "Q_budget": s.Q_budget,
            "H_budget_relative_error": s.H_budget_relative_error,
            "Q_budget_relative_error": s.Q_budget_relative_error,
        }
        for s in history
    ]


def stability_report(
    H0: torch.Tensor,
    Q0_mol_dm3: float,
    DH_nm2_s: float,
    DQ_nm2_s: float,
    kq_s_inv: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    dx_nm: float,
) -> dict:
    """Return the four stability time-step bounds for diagnostic display.

    Helpful for telling the safe (``kq <= 10``) and stiff (``kq >= 100``)
    regimes apart in the demo printouts.
    """
    H0_max = float(H0.abs().max().item())
    _, parts = _compute_dt_max(
        DH_nm2_s, DQ_nm2_s, kq_s_inv, kloss_s_inv, kdep_s_inv,
        H0_max, Q0_mol_dm3, dx_nm,
    )
    return parts

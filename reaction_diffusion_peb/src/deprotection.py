"""Deprotection (`P` field) for the PEB submodule.

This phase keeps every other piece of physics intact (no quencher, no
Arrhenius, no Petersen, no stochastic) and adds only the deprotected-
fraction equation:

    H equation : dH/dt = D_H * laplacian(H) - k_loss * H
    P equation : dP/dt = k_dep * H * (1 - P)

Initial conditions:

    H(x, y, 0) = H_0(x, y)
    P(x, y, 0) = 0

The H equation is unchanged from Phase 4 and is independent of P (P
does not appear in the H equation). The P equation is a first-order,
**linear in P** ODE with time-varying coefficient ``k_dep * H(x, y, t)``,
so per pixel it has the closed form

    P(x, y, t) = 1 - exp(-k_dep * integral_0^t H(x, y, tau) dtau)

We use that closed form for tests (and as an extra cross-check), but
the production demos run an explicit-Euler FD update on both ``H``
and ``P`` together.
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.diffusion_fd import laplacian_5pt


def step_acid_loss_deprotection_fd(
    H: torch.Tensor,
    P: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    dt_s: float,
    dx_nm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One explicit Euler step of the coupled (H, P) system.

    H update : H + dt * (D_H * laplacian(H) - k_loss * H)
    P update : P + dt * (k_dep * H * (1 - P))

    P is clamped to ``[0, 1]`` after the update — explicit Euler can
    drift slightly outside the physical band when ``k_dep * dt`` gets
    close to its stability bound, and the analytic solution stays in
    [0, 1] anyway.
    """
    H_new = H + dt_s * (DH_nm2_s * laplacian_5pt(H, dx_nm) - kloss_s_inv * H)
    P_new = P + dt_s * (kdep_s_inv * H * (1.0 - P))
    P_new = P_new.clamp(min=0.0, max=1.0)
    return H_new, P_new


def evolve_acid_loss_deprotection_fd(
    H0: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    kdep_s_inv: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the FD evolution for ``t_end_s`` seconds and return ``(H, P)``.

    ``P`` starts at zero. Time step is chosen to satisfy the diffusion
    CFL ``dt <= dx^2 / (4 D_H)``, the loss-stability bound
    ``dt <= 1 / k_loss``, and a coarse deprotection-stability bound
    ``dt <= 1 / (k_dep * max(H_0))``. All three limits are scaled by
    ``cfl_safety``.
    """
    for name, val in (("DH_nm2_s", DH_nm2_s), ("kloss_s_inv", kloss_s_inv),
                      ("kdep_s_inv", kdep_s_inv), ("t_end_s", t_end_s)):
        if val < 0:
            raise ValueError(f"{name} must be non-negative")
    if t_end_s == 0:
        return H0.clone(), torch.zeros_like(H0)

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

    H = H0
    P = torch.zeros_like(H0)
    for _ in range(n_steps):
        H, P = step_acid_loss_deprotection_fd(
            H, P, DH_nm2_s, kloss_s_inv, kdep_s_inv, dt, dx_nm,
        )
    return H, P


def deprotected_fraction_from_H_integral(
    H_integral: torch.Tensor,
    kdep_s_inv: float,
) -> torch.Tensor:
    """Closed-form ``P = 1 - exp(-k_dep * integral_0^t H(tau) dtau)``.

    Useful as a per-pixel analytic cross-check when ``H(x, y, tau)`` is
    available on a fine time grid (or in closed form via FFT).
    """
    if kdep_s_inv < 0:
        raise ValueError("kdep_s_inv must be non-negative")
    return 1.0 - torch.exp(-kdep_s_inv * H_integral)


def thresholded_area(P: torch.Tensor, P_threshold: float = 0.5) -> int:
    """Number of pixels whose ``P`` exceeds ``P_threshold`` — a CD-like
    proxy for the deprotection contour."""
    if not (0.0 <= P_threshold <= 1.0):
        raise ValueError("P_threshold must be in [0, 1]")
    return int((P > P_threshold).sum().item())

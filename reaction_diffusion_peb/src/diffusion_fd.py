"""Finite-difference diffusion baseline for the PEB submodule.

Solves the 2D heat equation in physical units:

    dH/dt = D_H * laplacian(H)

with periodic boundary conditions, using a 5-point stencil and
explicit Euler time stepping. Units throughout:

    H              : [mol / dm^3]
    D_H            : [nm^2 / s]
    dx (grid step) : [nm]
    t (time)       : [s]

The CFL stability bound is

    dt <= dx^2 / (4 * D_H)

`diffuse_fd` picks `dt` automatically with a configurable safety
factor and rejects user-provided step counts that would violate CFL.
"""

from __future__ import annotations

import math

import torch


def laplacian_5pt(H: torch.Tensor, dx_nm: float) -> torch.Tensor:
    """5-point Laplacian with periodic boundary conditions.

    Returns ``laplacian(H)`` in units of ``H / nm^2``.
    """
    if dx_nm <= 0:
        raise ValueError("dx_nm must be positive")
    H_xp = torch.roll(H, shifts=-1, dims=-1)
    H_xn = torch.roll(H, shifts=+1, dims=-1)
    H_yp = torch.roll(H, shifts=-1, dims=-2)
    H_yn = torch.roll(H, shifts=+1, dims=-2)
    return (H_xp + H_xn + H_yp + H_yn - 4.0 * H) / (dx_nm * dx_nm)


def step_diffusion_fd(H: torch.Tensor, DH_nm2_s: float,
                      dt_s: float, dx_nm: float) -> torch.Tensor:
    """One explicit Euler step of pure diffusion."""
    return H + dt_s * DH_nm2_s * laplacian_5pt(H, dx_nm)


def diffuse_fd(
    H0: torch.Tensor,
    DH_nm2_s: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> torch.Tensor:
    """Diffuse ``H0`` forward by ``t_end_s`` seconds.

    With ``n_steps=None`` the time step is chosen automatically to
    satisfy the CFL bound with a safety factor (``dt = cfl_safety *
    dx_nm**2 / (4 * DH_nm2_s)``). Setting ``n_steps`` explicitly
    raises ``ValueError`` if the resulting ``dt`` exceeds the CFL
    limit.
    """
    if DH_nm2_s < 0:
        raise ValueError("DH_nm2_s must be non-negative")
    if t_end_s < 0:
        raise ValueError("t_end_s must be non-negative")
    if DH_nm2_s == 0 or t_end_s == 0:
        return H0.clone()
    dt_max_s = (dx_nm * dx_nm) / (4.0 * DH_nm2_s)
    if n_steps is None:
        dt = cfl_safety * dt_max_s
        n_steps = max(1, int(math.ceil(t_end_s / dt)))
        dt = t_end_s / n_steps
    else:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        dt = t_end_s / n_steps
        if dt > dt_max_s + 1e-12:
            raise ValueError(
                f"FD step dt={dt:g} s violates CFL limit {dt_max_s:g} s "
                f"(dx={dx_nm} nm, DH={DH_nm2_s} nm^2/s)"
            )
    H = H0
    for _ in range(n_steps):
        H = step_diffusion_fd(H, DH_nm2_s, dt, dx_nm)
    return H

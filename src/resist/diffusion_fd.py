"""Finite-difference (explicit Euler) diffusion solver.

Solves the 2D heat equation

    dA/dt = D * laplacian(A)

with periodic boundary conditions, using the standard 5-point stencil:

    laplacian(A)_{i,j} = (A_{i+1,j} + A_{i-1,j} + A_{i,j+1} + A_{i,j-1}
                         - 4 * A_{i,j}) / dx**2

and explicit Euler time stepping. The CFL stability condition is

    dt <= dx**2 / (4 * D)

Periodic BCs are implemented with `torch.roll`, which keeps the operation
fully differentiable and pure-tensor (so it works on CPU and CUDA without
any kernel changes).
"""

from __future__ import annotations

import math

import torch


def laplacian_5pt(A: torch.Tensor, dx: float) -> torch.Tensor:
    """5-point Laplacian with periodic boundary conditions."""
    if dx <= 0:
        raise ValueError("dx must be positive")
    A_xp = torch.roll(A, shifts=-1, dims=-1)
    A_xn = torch.roll(A, shifts=+1, dims=-1)
    A_yp = torch.roll(A, shifts=-1, dims=-2)
    A_yn = torch.roll(A, shifts=+1, dims=-2)
    return (A_xp + A_xn + A_yp + A_yn - 4.0 * A) / (dx * dx)


def step_diffusion_fd(A: torch.Tensor, D: float, dt: float, dx: float) -> torch.Tensor:
    return A + dt * D * laplacian_5pt(A, dx)


def diffuse_fd(
    A0: torch.Tensor,
    D: float,
    t_end: float,
    dx: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> torch.Tensor:
    """Diffuse ``A0`` for ``t_end`` units of time.

    If ``n_steps`` is None, it is chosen automatically to satisfy the CFL
    condition with a safety factor (``dt = cfl_safety * dx**2 / (4 D)``).
    Setting ``n_steps`` explicitly raises ``ValueError`` if the resulting
    ``dt`` violates CFL.
    """
    if D < 0:
        raise ValueError("D must be non-negative")
    if t_end < 0:
        raise ValueError("t_end must be non-negative")
    if D == 0 or t_end == 0:
        return A0.clone()
    dt_max = dx * dx / (4.0 * D)
    if n_steps is None:
        dt = cfl_safety * dt_max
        n_steps = max(1, int(math.ceil(t_end / dt)))
        dt = t_end / n_steps
    else:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        dt = t_end / n_steps
        if dt > dt_max + 1e-12:
            raise ValueError(f"FD step dt={dt:g} violates CFL limit {dt_max:g}")
    A = A0
    for _ in range(n_steps):
        A = step_diffusion_fd(A, D, dt, dx)
    return A

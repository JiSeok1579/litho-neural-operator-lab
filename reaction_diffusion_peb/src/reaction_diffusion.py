"""Reaction-diffusion solvers for the PEB submodule.

Phase 4 introduces the acid-loss term:

    dH/dt = D_H * laplacian(H) - k_loss * H            (Phase 4)

Later phases will extend this with deprotection (P field), Arrhenius
temperature dependence, and quencher reaction. For Phase 4 the loss
term is **linear and homogeneous**, which means the FFT closed form
still works:

    H_hat(f, t) = H_hat(f, 0) * exp(-4 pi^2 D_H t |f|^2)
                              * exp(-k_loss * t)

so we keep an exact analytic reference. From Phase 5 onward the
equation becomes nonlinear and FD has to be the truth.

Units throughout:
    H              : [mol / dm^3]
    D_H            : [nm^2 / s]
    k_loss         : [1 / s]
    dx_nm          : [nm]
    t_s            : [s]
"""

from __future__ import annotations

import math

import torch

from reaction_diffusion_peb.src.diffusion_fd import laplacian_5pt
from reaction_diffusion_peb.src.fft_utils import fft2c, freq_grid_nm, ifft2c


# ---- finite difference -----------------------------------------------------

def step_acid_loss_fd(
    H: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    dt_s: float,
    dx_nm: float,
) -> torch.Tensor:
    """One explicit Euler step of ``dH/dt = D_H laplacian(H) - k_loss H``."""
    return H + dt_s * (DH_nm2_s * laplacian_5pt(H, dx_nm) - kloss_s_inv * H)


def diffuse_acid_loss_fd(
    H0: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    t_end_s: float,
    dx_nm: float,
    n_steps: int | None = None,
    cfl_safety: float = 0.5,
) -> torch.Tensor:
    """FD evolution of the diffusion + linear-loss equation.

    Time step is selected to satisfy both the diffusion CFL bound
    ``dt <= dx^2 / (4 D_H)`` and a coarse loss-stability bound
    ``dt <= 1 / k_loss``. Both limits are scaled by ``cfl_safety``.
    """
    if DH_nm2_s < 0:
        raise ValueError("DH_nm2_s must be non-negative")
    if kloss_s_inv < 0:
        raise ValueError("kloss_s_inv must be non-negative")
    if t_end_s < 0:
        raise ValueError("t_end_s must be non-negative")
    if t_end_s == 0:
        return H0.clone()
    if DH_nm2_s == 0 and kloss_s_inv == 0:
        return H0.clone()

    dt_diff = (dx_nm * dx_nm) / (4.0 * DH_nm2_s) if DH_nm2_s > 0 else float("inf")
    dt_react = (1.0 / kloss_s_inv) if kloss_s_inv > 0 else float("inf")
    dt_max = min(dt_diff, dt_react)

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
                f"min(dx^2/(4D), 1/k_loss) = {dt_max:g} s"
            )

    H = H0
    for _ in range(n_steps):
        H = step_acid_loss_fd(H, DH_nm2_s, kloss_s_inv, dt, dx_nm)
    return H


# ---- exact FFT solution ----------------------------------------------------

def diffuse_acid_loss_fft(
    H0: torch.Tensor,
    DH_nm2_s: float,
    kloss_s_inv: float,
    t_end_s: float,
    dx_nm: float,
) -> torch.Tensor:
    """Exact closed-form solution of ``dH/dt = D laplacian(H) - k_loss H``.

    The reaction term is linear and homogeneous, so the Fourier-domain
    decay simply factorizes:

        H_hat(f, t) = H_hat(f, 0) * exp(-4 pi^2 D_H t |f|^2 - k_loss t)
    """
    if DH_nm2_s < 0:
        raise ValueError("DH_nm2_s must be non-negative")
    if kloss_s_inv < 0:
        raise ValueError("kloss_s_inv must be non-negative")
    if t_end_s < 0:
        raise ValueError("t_end_s must be non-negative")
    if t_end_s == 0:
        return H0.clone()
    if DH_nm2_s == 0 and kloss_s_inv == 0:
        return H0.clone()
    grid_size = H0.shape[-1]
    fx, fy = freq_grid_nm(grid_size, dx_nm, dtype=H0.dtype, device=H0.device)
    fr2 = fx * fx + fy * fy
    decay = torch.exp(
        -4.0 * math.pi ** 2 * DH_nm2_s * t_end_s * fr2
        - kloss_s_inv * t_end_s
    )
    H_hat = fft2c(H0.to(torch.complex64))
    return ifft2c(H_hat * decay).real


# ---- diagnostic helpers ----------------------------------------------------

def total_mass(H: torch.Tensor, dx_nm: float) -> float:
    """``sum(H) * dx^2`` — total acid in mol/dm^3 * nm^2 (our units)."""
    return float(H.sum().item()) * (dx_nm * dx_nm)


def expected_mass_decay_factor(kloss_s_inv: float, t_end_s: float) -> float:
    """``exp(-k_loss * t)`` — analytic mass-decay factor."""
    return float(math.exp(-kloss_s_inv * t_end_s))

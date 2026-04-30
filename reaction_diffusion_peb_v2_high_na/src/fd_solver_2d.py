"""2D PEB solver: spectral diffusion+decay for H,Q (periodic) + explicit reaction for P.

Equations (with quencher; quencher disabled => Q≡0, kq=0):
    dH/dt = D_H ∇²H - k_loss H - kq H Q
    dQ/dt = D_Q ∇²Q          - kq H Q
    dP/dt = k_dep H (1 - P)

Operator splitting per step dt:
  1. linear diffusion+linear decay of H, Q via exact spectral step
  2. explicit reaction for the bilinear coupling and for P
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PEBResult:
    H: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    H_history: dict[float, np.ndarray]
    P_history: dict[float, np.ndarray]
    times_s: np.ndarray


def _spectral_diffusion_decay(
    field: np.ndarray,
    dx_nm: float,
    D: float,
    k_decay: float,
    dt: float,
) -> np.ndarray:
    """Exact step for ∂u/∂t = D∇²u - k_decay u with periodic BC."""
    if D == 0.0 and k_decay == 0.0:
        return field
    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx_nm) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx_nm) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    decay = np.exp(-(D * (KX ** 2 + KY ** 2) + k_decay) * dt)
    Fk = np.fft.fft2(field)
    return np.real(np.fft.ifft2(Fk * decay))


def solve_peb_2d(
    H0: np.ndarray,
    dx_nm: float,
    DH_nm2_s: float,
    kdep_s_inv: float,
    kloss_s_inv: float = 0.0,
    time_s: float = 60.0,
    dt_s: float = 0.5,
    quencher_enabled: bool = False,
    Q0: np.ndarray | float = 0.0,
    DQ_nm2_s: float = 0.0,
    kq_s_inv: float = 0.0,
    snapshots_s: tuple[float, ...] = (),
) -> PEBResult:
    H = H0.astype(np.float64, copy=True)
    P = np.zeros_like(H)
    if quencher_enabled:
        if np.isscalar(Q0):
            Q = np.full_like(H, float(Q0))
        else:
            Q = np.asarray(Q0, dtype=np.float64).copy()
    else:
        Q = np.zeros_like(H)

    n_steps = int(round(time_s / dt_s))
    if n_steps * dt_s < time_s - 1e-9:
        n_steps += 1
    dt_eff = time_s / n_steps

    snaps_set = set(round(s, 6) for s in snapshots_s)
    H_hist: dict[float, np.ndarray] = {}
    P_hist: dict[float, np.ndarray] = {}
    if 0.0 in snaps_set:
        H_hist[0.0] = H.copy()
        P_hist[0.0] = P.copy()

    times = np.zeros(n_steps + 1)
    for step in range(n_steps):
        # 1. spectral diffusion + linear loss for H, Q.
        H = _spectral_diffusion_decay(H, dx_nm, DH_nm2_s, kloss_s_inv, dt_eff)
        if quencher_enabled and (DQ_nm2_s > 0.0 or kq_s_inv != 0.0):
            Q = _spectral_diffusion_decay(Q, dx_nm, DQ_nm2_s, 0.0, dt_eff)

        # 2. explicit reaction. Use H at start of reaction substep.
        if quencher_enabled and kq_s_inv > 0.0:
            HQ = H * Q
            H = H - dt_eff * kq_s_inv * HQ
            Q = Q - dt_eff * kq_s_inv * HQ
            np.clip(H, 0.0, None, out=H)
            np.clip(Q, 0.0, None, out=Q)

        # P update.
        P = P + dt_eff * kdep_s_inv * H * (1.0 - P)
        np.clip(P, 0.0, 1.0, out=P)

        t = (step + 1) * dt_eff
        times[step + 1] = t
        t_key = round(t, 6)
        if t_key in snaps_set:
            H_hist[t_key] = H.copy()
            P_hist[t_key] = P.copy()

    if round(time_s, 6) not in H_hist:
        H_hist[round(time_s, 6)] = H.copy()
        P_hist[round(time_s, 6)] = P.copy()

    return PEBResult(
        H=H,
        Q=Q,
        P=P,
        H_history=H_hist,
        P_history=P_hist,
        times_s=times,
    )

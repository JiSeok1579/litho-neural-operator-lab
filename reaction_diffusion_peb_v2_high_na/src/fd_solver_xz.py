"""x-z PEB solver: spectral diffusion (periodic-x, Neumann-z via even mirror) +
explicit reaction. Field arrays have shape (n_z, n_x) — z is rows, x is cols.

Equations:
    dH/dt = D_H ∇²H - k_loss H - kq H Q
    dQ/dt = D_Q ∇²Q          - kq H Q
    dP/dt = k_dep H (1 - P)

Boundary conditions:
    x : periodic (FFT-native)
    z : Neumann / no-flux at z=0 and z=Lz (handled by even-mirror extension
        of the field along axis 0 before applying FFT in z).

Even-mirror extension trick:
    For a field u of length n_z along z, the periodic extension that mimics
    Neumann BCs is u_ext = [u, u[-2:0:-1]] of length 2*n_z - 2. Applying a
    plain 2D FFT to u_ext then cropping back to n_z is equivalent to using
    a discrete cosine transform in z and a Fourier transform in x.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PEBxzResult:
    H: np.ndarray
    Q: np.ndarray
    P: np.ndarray
    times_s: np.ndarray


def _even_mirror_extend_z(field: np.ndarray) -> np.ndarray:
    n_z = field.shape[0]
    if n_z <= 2:
        return field
    return np.concatenate([field, field[-2:0:-1, :]], axis=0)


def _spectral_diffusion_decay_xz(
    field: np.ndarray,
    dx_nm: float,
    dz_nm: float,
    D: float,
    k_decay: float,
    dt: float,
) -> np.ndarray:
    """Exact spectral step for ∂u/∂t = D∇²u - k_decay u with periodic-x, Neumann-z."""
    if D == 0.0 and k_decay == 0.0:
        return field
    n_z = field.shape[0]
    n_x = field.shape[1]
    ext = _even_mirror_extend_z(field)
    n_z_ext = ext.shape[0]
    kx = np.fft.fftfreq(n_x, d=dx_nm) * 2.0 * np.pi
    kz = np.fft.fftfreq(n_z_ext, d=dz_nm) * 2.0 * np.pi
    KX, KZ = np.meshgrid(kx, kz, indexing="xy")
    decay = np.exp(-(D * (KX ** 2 + KZ ** 2) + k_decay) * dt)
    Fk = np.fft.fft2(ext)
    out_ext = np.real(np.fft.ifft2(Fk * decay))
    return out_ext[:n_z, :]


def solve_peb_xz(
    H0: np.ndarray,
    dx_nm: float,
    dz_nm: float,
    DH_nm2_s: float,
    kdep_s_inv: float,
    kloss_s_inv: float = 0.0,
    time_s: float = 30.0,
    dt_s: float = 0.5,
    quencher_enabled: bool = False,
    Q0: float | np.ndarray = 0.0,
    DQ_nm2_s: float = 0.0,
    kq_s_inv: float = 0.0,
) -> PEBxzResult:
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
    times = np.zeros(n_steps + 1)

    for step in range(n_steps):
        H = _spectral_diffusion_decay_xz(H, dx_nm, dz_nm, DH_nm2_s, kloss_s_inv, dt_eff)
        if quencher_enabled and (DQ_nm2_s > 0.0 or kq_s_inv != 0.0):
            Q = _spectral_diffusion_decay_xz(Q, dx_nm, dz_nm, DQ_nm2_s, 0.0, dt_eff)

        if quencher_enabled and kq_s_inv > 0.0:
            HQ = H * Q
            H = H - dt_eff * kq_s_inv * HQ
            Q = Q - dt_eff * kq_s_inv * HQ
            np.clip(H, 0.0, None, out=H)
            np.clip(Q, 0.0, None, out=Q)

        P = P + dt_eff * kdep_s_inv * H * (1.0 - P)
        np.clip(P, 0.0, 1.0, out=P)

        times[step + 1] = (step + 1) * dt_eff

    return PEBxzResult(H=H, Q=Q, P=P, times_s=times)

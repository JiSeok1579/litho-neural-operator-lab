"""EUV secondary-electron blur as Gaussian convolution (FFT, periodic)."""
from __future__ import annotations

import numpy as np


def apply_gaussian_blur(
    field: np.ndarray,
    dx_nm: float,
    sigma_nm: float,
) -> np.ndarray:
    """Periodic Gaussian blur in 2D. sigma_nm <= 0 returns input unchanged."""
    if sigma_nm <= 0.0:
        return field.astype(np.float64, copy=True)

    ny, nx = field.shape
    kx = np.fft.fftfreq(nx, d=dx_nm) * 2.0 * np.pi
    ky = np.fft.fftfreq(ny, d=dx_nm) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    H = np.exp(-0.5 * (KX ** 2 + KY ** 2) * sigma_nm ** 2)

    Fk = np.fft.fft2(field)
    return np.real(np.fft.ifft2(Fk * H))

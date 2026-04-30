"""1D Gaussian-correlated edge roughness."""
from __future__ import annotations

import numpy as np


def gaussian_correlated_noise_1d(
    n: int,
    dy_nm: float,
    amp_nm: float,
    corr_nm: float,
    seed: int | None = None,
) -> np.ndarray:
    """Return a length-n 1D noise array with RMS amp_nm and Gaussian correlation length corr_nm."""
    if amp_nm <= 0.0 or corr_nm <= 0.0:
        return np.zeros(n, dtype=np.float64)

    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)

    # Gaussian kernel in y, periodic via FFT.
    y = (np.arange(n) - n // 2) * dy_nm
    sigma = corr_nm
    kernel = np.exp(-0.5 * (y / sigma) ** 2)
    kernel /= kernel.sum()

    Wk = np.fft.fft(white)
    Kk = np.fft.fft(np.fft.ifftshift(kernel))
    filtered = np.real(np.fft.ifft(Wk * Kk))

    rms = np.sqrt(np.mean(filtered ** 2))
    if rms < 1e-30:
        return np.zeros_like(filtered)
    return filtered * (amp_nm / rms)

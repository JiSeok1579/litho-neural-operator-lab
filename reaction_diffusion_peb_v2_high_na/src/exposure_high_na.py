"""Dill-style acid generation for High-NA EUV PEB v2."""
from __future__ import annotations

import numpy as np


def dill_acid_generation(
    intensity: np.ndarray,
    dose_norm: float,
    eta: float,
    Hmax: float,
) -> np.ndarray:
    """H0 = Hmax * (1 - exp(-eta * dose_norm * I))."""
    return Hmax * (1.0 - np.exp(-eta * dose_norm * intensity))


def normalize_dose(dose_mJ_cm2: float, reference_dose_mJ_cm2: float) -> float:
    if reference_dose_mJ_cm2 <= 0.0:
        raise ValueError("reference_dose_mJ_cm2 must be positive")
    return float(dose_mJ_cm2) / float(reference_dose_mJ_cm2)


def line_space_intensity_1d(
    domain_x_nm: float,
    grid_spacing_nm: float,
    pitch_nm: float,
    line_cd_nm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1D binary line-space intensity I(x). Returns (I, x_nm, line_centers_nm)."""
    nx = int(round(domain_x_nm / grid_spacing_nm))
    x = (np.arange(nx) + 0.5) * grid_spacing_nm
    n_lines = int(np.floor(domain_x_nm / pitch_nm))
    centers = (np.arange(n_lines) + 0.5) * pitch_nm
    half_cd = 0.5 * line_cd_nm
    I = np.zeros_like(x)
    for c in centers:
        I[(x >= c - half_cd) & (x <= c + half_cd)] = 1.0
    return I, x, centers


def gaussian_blur_1d(field: np.ndarray, dx_nm: float, sigma_nm: float) -> np.ndarray:
    """1D periodic Gaussian blur."""
    if sigma_nm <= 0.0:
        return field.astype(np.float64, copy=True)
    n = field.size
    k = np.fft.fftfreq(n, d=dx_nm) * 2.0 * np.pi
    H = np.exp(-0.5 * (k ** 2) * sigma_nm ** 2)
    return np.real(np.fft.ifft(np.fft.fft(field) * H))


def build_xz_intensity(
    I_x: np.ndarray,
    z_nm: np.ndarray,
    standing_wave_period_nm: float,
    standing_wave_amplitude: float,
    standing_wave_phase_rad: float = 0.0,
    absorption_length_nm: float | None = 30.0,
) -> np.ndarray:
    """Construct I(z, x) from a 1D intensity profile + standing-wave / absorption envelopes.

        I(x, z) = I_x(x) * [1 + A * cos(2*pi*z/period + phase)] * exp(-z/abs_len)

    The cosine modulation is omitted when amplitude == 0; the exponential
    envelope is omitted when absorption_length_nm is None or <= 0.

    Returned shape: (n_z, n_x).
    """
    sw = 1.0 + standing_wave_amplitude * np.cos(
        2.0 * np.pi * z_nm / standing_wave_period_nm + standing_wave_phase_rad
    )
    if absorption_length_nm is not None and absorption_length_nm > 0.0:
        env = np.exp(-z_nm / absorption_length_nm)
    else:
        env = np.ones_like(z_nm)
    z_factor = sw * env
    return z_factor[:, None] * I_x[None, :]

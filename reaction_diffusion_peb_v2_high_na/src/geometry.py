"""Line/space geometry with optional Gaussian-correlated edge roughness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roughness import gaussian_correlated_noise_1d


@dataclass
class LineSpaceGrid:
    x_nm: np.ndarray  # (nx,)
    y_nm: np.ndarray  # (ny,)
    dx_nm: float
    pitch_nm: float
    line_cd_nm: float
    line_centers_nm: np.ndarray  # (n_lines,)


def build_grid(
    domain_x_nm: float,
    domain_y_nm: float,
    grid_spacing_nm: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    nx = int(round(domain_x_nm / grid_spacing_nm))
    ny = int(round(domain_y_nm / grid_spacing_nm))
    x = (np.arange(nx) + 0.5) * grid_spacing_nm
    y = (np.arange(ny) + 0.5) * grid_spacing_nm
    return x, y, float(grid_spacing_nm)


def line_centers(domain_x_nm: float, pitch_nm: float) -> np.ndarray:
    """Centers of lines that fit in [0, domain_x_nm] when the first line is centered at pitch/2."""
    n = int(np.floor(domain_x_nm / pitch_nm))
    return (np.arange(n) + 0.5) * pitch_nm


def line_space_intensity(
    domain_x_nm: float,
    domain_y_nm: float,
    grid_spacing_nm: float,
    pitch_nm: float,
    line_cd_nm: float,
    edge_roughness_amp_nm: float = 0.0,
    edge_roughness_corr_nm: float = 5.0,
    edge_roughness_seed: int | None = None,
) -> tuple[np.ndarray, LineSpaceGrid]:
    """Binary intensity I(x,y) for line-space pattern.

    Convention: line region (resist exposed to "open" mask) has I=1; space has I=0.
    Edge roughness perturbs each line's left/right edges per y row.
    """
    x, y, dx = build_grid(domain_x_nm, domain_y_nm, grid_spacing_nm)
    centers = line_centers(domain_x_nm, pitch_nm)
    half_cd = 0.5 * line_cd_nm

    ny = y.size
    nx = x.size
    I = np.zeros((ny, nx), dtype=np.float64)

    for i, c in enumerate(centers):
        if edge_roughness_amp_nm > 0.0:
            seed_l = None if edge_roughness_seed is None else edge_roughness_seed + 2 * i
            seed_r = None if edge_roughness_seed is None else edge_roughness_seed + 2 * i + 1
            dl = gaussian_correlated_noise_1d(ny, dx, edge_roughness_amp_nm, edge_roughness_corr_nm, seed=seed_l)
            dr = gaussian_correlated_noise_1d(ny, dx, edge_roughness_amp_nm, edge_roughness_corr_nm, seed=seed_r)
        else:
            dl = np.zeros(ny)
            dr = np.zeros(ny)
        left = (c - half_cd) + dl  # (ny,)
        right = (c + half_cd) + dr
        # Broadcast: x has shape (nx,), left/right have shape (ny,). Result (ny,nx).
        inside = (x[None, :] >= left[:, None]) & (x[None, :] <= right[:, None])
        I[inside] = 1.0

    grid = LineSpaceGrid(
        x_nm=x,
        y_nm=y,
        dx_nm=dx,
        pitch_nm=float(pitch_nm),
        line_cd_nm=float(line_cd_nm),
        line_centers_nm=centers,
    )
    return I, grid

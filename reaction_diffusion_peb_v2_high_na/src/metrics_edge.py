"""Edge / LER / LWR / CD metrics from a 2D field at a given threshold."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeResult:
    line_centers_nm: np.ndarray  # (n_lines,)
    left_edges_nm: np.ndarray    # (n_lines, ny) NaN where no crossing found
    right_edges_nm: np.ndarray   # (n_lines, ny) NaN where no crossing found
    cd_nm: np.ndarray            # (n_lines, ny)
    ler_left_nm: np.ndarray      # (n_lines,)
    ler_right_nm: np.ndarray     # (n_lines,)
    ler_mean_nm: float
    cd_mean_nm: np.ndarray       # (n_lines,)
    lwr_nm: np.ndarray           # (n_lines,) 3*sigma(CD(y))
    lwr_mean_nm: float
    cd_overall_mean_nm: float


def _interp_crossing(x: np.ndarray, vals: np.ndarray, threshold: float, ascending: bool) -> float:
    """Find x where vals(x) crosses threshold. ascending=True => first index i with vals[i]>=threshold.

    Linear interpolation between i-1 and i. Returns NaN if no crossing.
    """
    if ascending:
        mask = vals >= threshold
        if not mask.any():
            return np.nan
        i = int(np.argmax(mask))
        if i == 0:
            return float(x[0])
        v0, v1 = vals[i - 1], vals[i]
        if v1 == v0:
            return float(x[i])
        frac = (threshold - v0) / (v1 - v0)
        return float(x[i - 1] + frac * (x[i] - x[i - 1]))
    else:
        mask = vals >= threshold
        if not mask.any():
            return np.nan
        # last index that is >= threshold
        i = len(vals) - 1 - int(np.argmax(mask[::-1]))
        if i == len(vals) - 1:
            return float(x[i])
        v0, v1 = vals[i], vals[i + 1]
        if v1 == v0:
            return float(x[i])
        frac = (threshold - v0) / (v1 - v0)
        return float(x[i] + frac * (x[i + 1] - x[i]))


def extract_edges(
    field: np.ndarray,
    x_nm: np.ndarray,
    line_centers_nm: np.ndarray,
    pitch_nm: float,
    threshold: float,
) -> EdgeResult:
    """For each line, scan its half-pitch window left/right of center and locate
    the threshold crossing per y row. field shape (ny, nx)."""
    ny, nx = field.shape
    half_p = 0.5 * pitch_nm
    n_lines = len(line_centers_nm)
    left = np.full((n_lines, ny), np.nan)
    right = np.full((n_lines, ny), np.nan)

    for li, c in enumerate(line_centers_nm):
        i_c = int(np.argmin(np.abs(x_nm - c)))
        i_left_min = int(np.searchsorted(x_nm, c - half_p, side="left"))
        i_right_max = int(np.searchsorted(x_nm, c + half_p, side="right")) - 1
        i_left_min = max(i_left_min, 0)
        i_right_max = min(i_right_max, nx - 1)

        for j in range(ny):
            row = field[j]
            # left edge: ascending crossing in [i_left_min, i_c]
            seg_l = row[i_left_min:i_c + 1]
            x_l = x_nm[i_left_min:i_c + 1]
            left[li, j] = _interp_crossing(x_l, seg_l, threshold, ascending=True)
            # right edge: descending crossing in [i_c, i_right_max]
            seg_r = row[i_c:i_right_max + 1]
            x_r = x_nm[i_c:i_right_max + 1]
            right[li, j] = _interp_crossing(x_r, seg_r, threshold, ascending=False)

    cd = right - left
    ler_left = np.array([3.0 * np.nanstd(left[li]) for li in range(n_lines)])
    ler_right = np.array([3.0 * np.nanstd(right[li]) for li in range(n_lines)])
    cd_mean = np.array([np.nanmean(cd[li]) for li in range(n_lines)])
    lwr = np.array([3.0 * np.nanstd(cd[li]) for li in range(n_lines)])

    valid = np.isfinite(np.concatenate([ler_left, ler_right]))
    ler_mean = float(np.nanmean(np.concatenate([ler_left, ler_right])[valid])) if valid.any() else float("nan")

    return EdgeResult(
        line_centers_nm=np.asarray(line_centers_nm, dtype=np.float64),
        left_edges_nm=left,
        right_edges_nm=right,
        cd_nm=cd,
        ler_left_nm=ler_left,
        ler_right_nm=ler_right,
        ler_mean_nm=ler_mean,
        cd_mean_nm=cd_mean,
        lwr_nm=lwr,
        lwr_mean_nm=float(np.nanmean(lwr)),
        cd_overall_mean_nm=float(np.nanmean(cd_mean)),
    )


def edge_residual_psd(edge_y_nm: np.ndarray, dy_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """1D PSD of the edge-residual signal (edge minus mean), returned as positive-frequency only."""
    if not np.isfinite(edge_y_nm).all():
        edge_y_nm = np.where(np.isfinite(edge_y_nm), edge_y_nm, np.nanmean(edge_y_nm))
    resid = edge_y_nm - np.mean(edge_y_nm)
    n = resid.size
    F = np.fft.rfft(resid)
    psd = (np.abs(F) ** 2) * dy_nm / n
    freqs = np.fft.rfftfreq(n, d=dy_nm)
    return freqs, psd

"""PEB Phase-2 demo: FD vs FFT cross-check.

Run:
    python reaction_diffusion_peb/experiments/02_diffusion_baseline/compare_fd_fft.py

Runs both solvers on the same H_0 (Gaussian-spot exposure) at three
``DH`` values, computes the absolute and relative errors, and saves
side-by-side figures plus a metrics CSV.

Saves:
    outputs/figures/peb_phase2_compare_fd_fft.png
    outputs/figures/peb_phase2_compare_fd_fft_DH<x>.png    (per-DH cuts)
    outputs/logs/peb_phase2_compare_fd_fft_metrics.csv

The FFT solution is treated as ground truth (it solves the heat
equation exactly modulo float precision); FD truncation error is
measured against it.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.diffusion_fft import diffuse_fft
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_fd_vs_fft,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
ETA = 1.0
HMAX = 0.2

T_REF = 60.0
DH_VALUES = [0.3, 0.8, 1.5]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=1.0, eta=ETA, Hmax=HMAX)

    rows = [["DH_nm2_s", "t_s", "L_nm",
             "FD_peak", "FFT_peak", "max_abs_err", "L2_rel_err",
             "fd_wall_s", "fft_wall_s"]]

    for DH in DH_VALUES:
        t0 = time.time()
        H_fd = diffuse_fd(H0, DH_nm2_s=DH, t_end_s=T_REF, dx_nm=DX_NM)
        fd_wall = time.time() - t0

        t0 = time.time()
        H_fft = diffuse_fft(H0, DH_nm2_s=DH, t_end_s=T_REF, dx_nm=DX_NM)
        fft_wall = time.time() - t0

        diff = H_fd - H_fft
        max_abs_err = float(diff.abs().max().item())
        l2_norm_diff = float((diff ** 2).sum().sqrt().item())
        l2_norm_ref = float((H_fft ** 2).sum().sqrt().item())
        rel_err = l2_norm_diff / max(l2_norm_ref, 1e-12)
        L_nm = math.sqrt(2.0 * DH * T_REF)
        print(f"  DH={DH} nm^2/s, t={T_REF} s, L={L_nm:.2f} nm, "
              f"max|FD - FFT|={max_abs_err:.3e}, rel L2={rel_err:.3e}, "
              f"FD {fd_wall:.2f} s, FFT {fft_wall:.4f} s")

        rows.append([
            f"{DH:.2f}", f"{T_REF:.0f}", f"{L_nm:.3f}",
            f"{H_fd.max().item():.4f}",
            f"{H_fft.max().item():.4f}",
            f"{max_abs_err:.3e}",
            f"{rel_err:.3e}",
            f"{fd_wall:.3f}",
            f"{fft_wall:.5f}",
        ])

        fig = show_fd_vs_fft(
            H0, H_fd, H_fft, dx_nm=DX_NM, t_end_s=T_REF, Hmax=HMAX,
            suptitle=(
                f"PEB phase 2: FD vs FFT   DH={DH} nm^2/s   t={T_REF} s   "
                f"L={L_nm:.2f} nm   max|FD-FFT|={max_abs_err:.3e}"
            ),
        )
        out = save_figure(fig, OUT_FIG / f"peb_phase2_compare_fd_fft_DH{DH}.png")
        print(f"    wrote {out}")

    # Combined figure: just use the reference DH=0.8 for the headline plot.
    DH = 0.8
    H_fd = diffuse_fd(H0, DH_nm2_s=DH, t_end_s=T_REF, dx_nm=DX_NM)
    H_fft = diffuse_fft(H0, DH_nm2_s=DH, t_end_s=T_REF, dx_nm=DX_NM)
    fig = show_fd_vs_fft(
        H0, H_fd, H_fft, dx_nm=DX_NM, t_end_s=T_REF, Hmax=HMAX,
        suptitle=(
            f"PEB phase 2 reference: FD vs FFT   DH={DH} nm^2/s   t={T_REF} s"
        ),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase2_compare_fd_fft.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase2_compare_fd_fft_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

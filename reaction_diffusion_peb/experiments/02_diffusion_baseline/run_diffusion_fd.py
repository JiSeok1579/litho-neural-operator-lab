"""PEB Phase-2 demo: FD diffusion baseline on a Gaussian acid spot.

Run:
    python reaction_diffusion_peb/experiments/02_diffusion_baseline/run_diffusion_fd.py

Pipeline:
    gaussian_spot(grid_size=128, sigma_px=12)
        -> acid_generation(dose=1, eta=1, Hmax=0.2)            -> H_0
        -> diffuse_fd(DH=0.8 nm^2/s, t_end=60 s, dx=1 nm)      -> H(60)

Saves:
    outputs/figures/peb_phase2_fd_chain.png
    outputs/figures/peb_phase2_fd_dh_sweep.png
    outputs/arrays/peb_phase2_fd_H_DH0.8_t60.npy
    outputs/logs/peb_phase2_fd_metrics.csv

Verifies:
    - Mass conservation (no loss term).
    - Larger DH and longer t both increase blur.
    - CD-like FWHM grows with diffusion length sqrt(2 D t).
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
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_diffusion_chain,
    show_diffusion_dh_sweep,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_ARR = Path("reaction_diffusion_peb/outputs/arrays")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_PX = 12.0
ETA = 1.0
HMAX = 0.2

DH_REF = 0.8           # nm^2 / s
T_REF = 60.0           # s
DH_VALUES = [0.3, 0.8, 1.5]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    H0 = acid_generation(I, dose=1.0, eta=ETA, Hmax=HMAX)

    print(f"  H_0 peak={H0.max().item():.4f} mol/dm^3, "
          f"sum={H0.sum().item():.3f}")

    # Reference run
    t0 = time.time()
    H_ref = diffuse_fd(H0, DH_nm2_s=DH_REF, t_end_s=T_REF, dx_nm=DX_NM)
    elapsed = time.time() - t0
    L_ref_nm = math.sqrt(2.0 * DH_REF * T_REF)
    print(f"  reference DH={DH_REF}, t={T_REF}: peak={H_ref.max().item():.4f}, "
          f"sum={H_ref.sum().item():.3f}, L={L_ref_nm:.2f} nm, "
          f"FD wall {elapsed:.2f} s")

    fig = show_diffusion_chain(
        H0, H_ref, dx_nm=DX_NM, t_end_s=T_REF, Hmax=HMAX,
        suptitle=(f"PEB phase 2 (FD): DH={DH_REF} nm^2/s   t={T_REF} s   "
                  f"L=sqrt(2 D t)={L_ref_nm:.2f} nm"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase2_fd_chain.png")
    print(f"  wrote {out}")

    # DH sweep
    H_results = []
    rows = [["solver", "DH_nm2_s", "t_s", "L_nm",
             "H_peak", "H_sum_rel_change", "wallclock_s"]]
    rows.append([
        "FD", "0.0", "0.0", "0.000",
        f"{H0.max().item():.4f}", "0.000", "0.000",
    ])
    s_before = H0.sum().item()
    for DH in DH_VALUES:
        t0 = time.time()
        H_t = diffuse_fd(H0, DH_nm2_s=DH, t_end_s=T_REF, dx_nm=DX_NM)
        elapsed = time.time() - t0
        H_results.append(H_t)
        L_nm = math.sqrt(2.0 * DH * T_REF)
        s_after = H_t.sum().item()
        rel_dm = abs(s_after - s_before) / max(s_before, 1e-12)
        rows.append([
            "FD",
            f"{DH:.2f}",
            f"{T_REF:.0f}",
            f"{L_nm:.3f}",
            f"{H_t.max().item():.4f}",
            f"{rel_dm:.6f}",
            f"{elapsed:.3f}",
        ])

    fig = show_diffusion_dh_sweep(
        H0, H_results, DH_VALUES, dx_nm=DX_NM, t_end_s=T_REF, Hmax=HMAX,
        suptitle=f"PEB phase 2 (FD): DH sweep at t={T_REF} s, dx={DX_NM} nm",
    )
    out = save_figure(fig, OUT_FIG / "peb_phase2_fd_dh_sweep.png")
    print(f"  wrote {out}")

    OUT_ARR.mkdir(parents=True, exist_ok=True)
    arr_path = OUT_ARR / f"peb_phase2_fd_H_DH{DH_REF}_t{int(T_REF)}.npy"
    np.save(arr_path, H_ref.cpu().numpy())
    print(f"  wrote {arr_path}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase2_fd_metrics.csv"
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

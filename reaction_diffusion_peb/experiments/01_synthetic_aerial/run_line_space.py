"""PEB Phase-1 demo: line-space exposure with edge smoothing.

Run:
    python reaction_diffusion_peb/experiments/01_synthetic_aerial/run_line_space.py

Saves:
    outputs/figures/peb_phase1_line_space_chain.png
    outputs/figures/peb_phase1_line_space_dose_sweep.png
    outputs/arrays/peb_phase1_line_space_H0_dose1.0.npy
    outputs/logs/peb_phase1_line_space_metrics.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import line_space
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_aerial_and_acid,
    show_dose_sweep,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_ARR = Path("reaction_diffusion_peb/outputs/arrays")

GRID_SIZE = 128
PITCH_PX = 32.0
DUTY = 0.5
CONTRAST = 1.0
SMOOTH_PX = 1.5    # gentle edges so the IC is not Nyquist-sharp
ETA = 1.0
HMAX = 0.2
DOSES = [0.5, 1.0, 1.5, 2.0]


def main() -> None:
    I = line_space(
        GRID_SIZE, pitch_px=PITCH_PX, duty=DUTY, contrast=CONTRAST,
        smooth_px=SMOOTH_PX,
    )
    print(f"  aerial peak={I.max().item():.4f}, min={I.min().item():.4f}, "
          f"mean={I.mean().item():.4f}")

    H0_list = []
    rows = [["dose", "H0_peak", "H0_min", "H0_mean", "H0_le_Hmax"]]
    for dose in DOSES:
        H0 = acid_generation(I, dose=dose, eta=ETA, Hmax=HMAX)
        H0_list.append(H0)
        rows.append([
            f"{dose:.2f}",
            f"{H0.max().item():.6f}",
            f"{H0.min().item():.6f}",
            f"{H0.mean().item():.6f}",
            "yes" if (H0 <= HMAX + 1e-9).all().item() else "NO",
        ])

    fig = show_aerial_and_acid(
        I, H0_list[1], Hmax=HMAX,
        suptitle=(f"PEB phase 1: line-space   pitch={PITCH_PX:g} px   "
                  f"duty={DUTY}   contrast={CONTRAST}   smooth={SMOOTH_PX} px   "
                  f"dose={DOSES[1]}   Hmax={HMAX}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase1_line_space_chain.png")
    print(f"  wrote {out}")

    fig = show_dose_sweep(
        I, H0_list, DOSES, Hmax=HMAX,
        suptitle=(f"PEB phase 1: line-space dose sweep   "
                  f"pitch={PITCH_PX:g} px   duty={DUTY}   smooth={SMOOTH_PX} px"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase1_line_space_dose_sweep.png")
    print(f"  wrote {out}")

    OUT_ARR.mkdir(parents=True, exist_ok=True)
    arr_path = OUT_ARR / "peb_phase1_line_space_H0_dose1.0.npy"
    np.save(arr_path, H0_list[1].cpu().numpy())
    print(f"  wrote {arr_path}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase1_line_space_metrics.csv"
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

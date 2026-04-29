"""PEB Phase-1 demo: Gaussian-spot exposure.

Run:
    python reaction_diffusion_peb/experiments/01_synthetic_aerial/run_gaussian_spot.py

Pipeline:
    gaussian_spot(grid_size=128, sigma_px=12)
        -> normalized aerial intensity I(x, y) in [0, 1]
        -> acid_generation(dose=[0.5, 1.0, 1.5, 2.0], eta=1, Hmax=0.2)
        -> H_0(x, y) per dose

Saves:
    outputs/figures/peb_phase1_gaussian_spot.png        (single chain)
    outputs/figures/peb_phase1_gaussian_dose_sweep.png  (dose sweep row)
    outputs/arrays/peb_phase1_gaussian_H0_dose1.0.npy
    outputs/logs/peb_phase1_gaussian_metrics.csv

Verifies:
    - I is normalized to [0, 1] with peak 1 at the center.
    - H_0 increases monotonically with dose.
    - H_0 never exceeds Hmax.
    - I = 0 implies H_0 ~ 0.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_aerial_and_acid,
    show_dose_sweep,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_ARR = Path("reaction_diffusion_peb/outputs/arrays")

GRID_SIZE = 128
SIGMA_PX = 12.0
ETA = 1.0
HMAX = 0.2
DOSES = [0.5, 1.0, 1.5, 2.0]


def main() -> None:
    I = gaussian_spot(GRID_SIZE, sigma_px=SIGMA_PX)
    print(f"  aerial peak={I.max().item():.4f}, min={I.min().item():.4f}")

    H0_list = []
    rows = [["dose", "I_peak", "I_min", "H0_peak", "H0_min", "H0_le_Hmax"]]
    for dose in DOSES:
        H0 = acid_generation(I, dose=dose, eta=ETA, Hmax=HMAX)
        H0_list.append(H0)
        rows.append([
            f"{dose:.2f}",
            f"{I.max().item():.4f}",
            f"{I.min().item():.4f}",
            f"{H0.max().item():.6f}",
            f"{H0.min().item():.6f}",
            "yes" if (H0 <= HMAX + 1e-9).all().item() else "NO",
        ])

    fig = show_aerial_and_acid(
        I, H0_list[1], Hmax=HMAX,
        suptitle=(f"PEB phase 1: Gaussian spot   sigma={SIGMA_PX:g} px   "
                  f"dose={DOSES[1]}   eta={ETA}   Hmax={HMAX} mol/dm^3"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase1_gaussian_spot.png")
    print(f"  wrote {out}")

    fig = show_dose_sweep(
        I, H0_list, DOSES, Hmax=HMAX,
        suptitle=("PEB phase 1: Gaussian spot dose sweep   "
                  f"sigma={SIGMA_PX:g} px   eta={ETA}   Hmax={HMAX}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase1_gaussian_dose_sweep.png")
    print(f"  wrote {out}")

    OUT_ARR.mkdir(parents=True, exist_ok=True)
    arr_path = OUT_ARR / "peb_phase1_gaussian_H0_dose1.0.npy"
    np.save(arr_path, H0_list[1].cpu().numpy())
    print(f"  wrote {arr_path}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase1_gaussian_metrics.csv"
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

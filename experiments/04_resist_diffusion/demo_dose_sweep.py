"""Phase 5 demo 5-A: dose sweep on a vertical line-space.

Run:
    python experiments/04_resist_diffusion/demo_dose_sweep.py

Pipeline:

    mask -> coherent_aerial_image (NA=0.6) -> acid_from_aerial(dose)
        -> diffuse_fft_by_length(L=0.10 lambda)
        -> soft_threshold(A_th=0.40, beta=25)

For each dose in {0.5, 1.0, 1.5, 2.0}, the demo records:
- aerial peak intensity (constant across doses by construction)
- max acid concentration (saturating in dose)
- thresholded area (CD-like, at the row through y=0)
- area fraction in the resist (proxy for printed line width)

Saves under ``outputs/figures/``:
- phase5_dose_sweep.png — 4 rows of (diffused acid, resist) at each dose
- phase5_dose_chain_dose1.5.png — single-dose chain figure (aerial -> acid
  -> diffused acid -> resist) for context

Saves under ``outputs/logs/``:
- phase5_dose_sweep_metrics.csv

Physical takeaway:
- Dose increases acid concentration smoothly toward saturation A=1, so
  the resist contour widens with dose. CD-like width grows monotonically.
- After a fixed diffusion length, edges blur the same amount in absolute
  terms — but high-dose features still print because the bulk acid stays
  far above the threshold even after diffusion.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_resist_chain,
    show_resist_sweep,
)
from src.mask import patterns
from src.mask.transmission import binary_transmission
from src.optics.coherent_imaging import coherent_aerial_image
from src.optics.pupil import circular_pupil
from src.resist.diffusion_fft import diffuse_fft_by_length
from src.resist.exposure import acid_from_aerial
from src.resist.threshold import (
    measure_cd_horizontal,
    soft_threshold,
    thresholded_area,
)

OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
N = 256
EXTENT = 10.0
WAVELENGTH = 1.0
NA = 0.6
PITCH = 2.0
ETA = 1.0
DIFFUSION_LENGTH = 0.10
A_TH = 0.40
BETA = 25.0
DOSES = [0.5, 1.0, 1.5, 2.0]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    mask = patterns.line_space(grid, pitch=PITCH, duty=0.5, orientation="vertical")
    t = binary_transmission(mask)
    pupil = circular_pupil(grid, NA=NA, wavelength=WAVELENGTH)
    aerial = coherent_aerial_image(t, pupil, normalize=True)

    rows = [["dose", "aerial_peak", "acid_max", "acid_diffused_max",
             "resist_max", "cd_lambda", "thresholded_pixels"]]
    sweep_rows = []
    for dose in DOSES:
        acid0 = acid_from_aerial(aerial, dose=dose, eta=ETA)
        acid_diff = diffuse_fft_by_length(acid0, grid, diffusion_length=DIFFUSION_LENGTH)
        resist = soft_threshold(acid_diff, A_th=A_TH, beta=BETA)

        cd_lambda = measure_cd_horizontal(resist, threshold=0.5, dx=grid.dx)
        rows.append([
            f"{dose:.1f}",
            f"{aerial.max().item():.4f}",
            f"{acid0.max().item():.4f}",
            f"{acid_diff.max().item():.4f}",
            f"{resist.max().item():.4f}",
            f"{cd_lambda:.4f}",
            thresholded_area(resist, threshold=0.5),
        ])
        sweep_rows.append((f"dose={dose:.1f}", acid_diff, resist))

    fig = show_resist_sweep(
        sweep_rows, extent=grid.extent,
        suptitle=(f"phase 5 dose sweep   line-space pitch={PITCH} lambda   "
                  f"NA={NA}   L_diff={DIFFUSION_LENGTH}   A_th={A_TH}"),
    )
    out = save_figure(fig, OUT_FIG / "phase5_dose_sweep.png")
    print(f"  wrote {out}")

    # One-shot chain figure at dose 1.5 for context.
    dose_ref = 1.5
    acid0_ref = acid_from_aerial(aerial, dose=dose_ref, eta=ETA)
    acid_diff_ref = diffuse_fft_by_length(acid0_ref, grid, diffusion_length=DIFFUSION_LENGTH)
    resist_ref = soft_threshold(acid_diff_ref, A_th=A_TH, beta=BETA)
    fig = show_resist_chain(
        aerial, acid0_ref, acid_diff_ref, resist_ref,
        extent=grid.extent, threshold=A_TH,
        suptitle=(f"phase 5 chain   dose={dose_ref}   line-space pitch={PITCH} lambda   "
                  f"NA={NA}   L_diff={DIFFUSION_LENGTH}   A_th={A_TH}"),
    )
    out = save_figure(fig, OUT_FIG / "phase5_dose_chain_dose1.5.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase5_dose_sweep_metrics.csv"
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

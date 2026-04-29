"""Phase 5 demo 5-B: diffusion-length sweep on a vertical line-space.

Run:
    python experiments/04_resist_diffusion/demo_diffusion_length.py

Pipeline as in demo_dose_sweep, but with dose fixed and diffusion length
swept across {0.0, 0.05, 0.10, 0.20, 0.40} lambda.

Captures the canonical "aerial vs resist" effect from study plan §5.10:
sub-threshold leakage in the aerial vanishes from the resist after
diffusion, because diffusion blurs both signal and leakage but the
thresholding then cuts the now-low-amplitude leakage off entirely.

Saves:
- outputs/figures/phase5_diffusion_length_sweep.png
- outputs/logs/phase5_diffusion_length_metrics.csv
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import save_figure, show_resist_sweep
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
DOSE = 1.5
A_TH = 0.40
BETA = 25.0
DIFFUSION_LENGTHS = [0.0, 0.05, 0.10, 0.20, 0.40]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    mask = patterns.line_space(grid, pitch=PITCH, duty=0.5, orientation="vertical")
    t = binary_transmission(mask)
    pupil = circular_pupil(grid, NA=NA, wavelength=WAVELENGTH)
    aerial = coherent_aerial_image(t, pupil, normalize=True)
    acid0 = acid_from_aerial(aerial, dose=DOSE, eta=ETA)

    rows = [["L_diff_lambda", "acid_max", "acid_min", "acid_diffused_contrast",
             "resist_max", "cd_lambda", "thresholded_pixels"]]
    sweep_rows = []
    for L in DIFFUSION_LENGTHS:
        acid_diff = diffuse_fft_by_length(acid0, grid, diffusion_length=L)
        resist = soft_threshold(acid_diff, A_th=A_TH, beta=BETA)
        # Contrast of the diffused acid (as a proxy for "edge sharpness"
        # before the threshold cut).
        a_max = acid_diff.max().item()
        a_min = acid_diff.min().item()
        contrast = (a_max - a_min) / (a_max + a_min + 1e-12)
        cd_lambda = measure_cd_horizontal(resist, threshold=0.5, dx=grid.dx)
        rows.append([
            f"{L:.2f}",
            f"{a_max:.4f}",
            f"{a_min:.4f}",
            f"{contrast:.4f}",
            f"{resist.max().item():.4f}",
            f"{cd_lambda:.4f}",
            thresholded_area(resist, threshold=0.5),
        ])
        sweep_rows.append((f"L_diff={L:.2f}", acid_diff, resist))

    fig = show_resist_sweep(
        sweep_rows, extent=grid.extent,
        suptitle=(f"phase 5 diffusion length sweep   line-space pitch={PITCH} lambda   "
                  f"NA={NA}   dose={DOSE}   A_th={A_TH}"),
    )
    out = save_figure(fig, OUT_FIG / "phase5_diffusion_length_sweep.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase5_diffusion_length_metrics.csv"
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

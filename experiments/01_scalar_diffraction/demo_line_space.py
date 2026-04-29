"""Phase 1 demo: line-space gratings -> diffraction spectrum.

Run:
    python experiments/01_scalar_diffraction/demo_line_space.py

Sweeps pitch over a small set of values, plots mask + |T| + log|T| +
phase(T), and saves one figure per pitch into ``outputs/figures/``.

Physical takeaway:
- Smaller pitch increases the spacing between diffraction orders along the
  axis perpendicular to the lines (orders sit at fx = k / pitch).
- Sharp on/off edges spread spectral energy into many orders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import save_figure, show_field_pair
from src.mask.patterns import line_space
from src.mask.transmission import binary_transmission
from src.optics.scalar_diffraction import diffraction_spectrum

OUT_DIR = Path("outputs/figures")
N = 256
EXTENT = 1.0
PITCHES = [0.40, 0.20, 0.10, 0.05]
DUTY = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    torch.manual_seed(0)
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    for pitch in PITCHES:
        mask = line_space(grid, pitch=pitch, duty=DUTY, orientation="vertical")
        spec = diffraction_spectrum(binary_transmission(mask))
        fig = show_field_pair(
            mask, spec, extent=grid.extent, df=grid.df,
            title=f"line-space  pitch={pitch}  duty={DUTY}",
        )
        out = save_figure(fig, OUT_DIR / f"phase1_line_space_pitch_{pitch:.2f}.png")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

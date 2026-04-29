"""Phase 1 demo: isolated contact hole -> 2D sinc-like spectrum.

Run:
    python experiments/01_scalar_diffraction/demo_contact_hole.py

For a circular hole, the diffraction spectrum follows an Airy / Bessel-J1
pattern (the 2D analogue of a sinc). Smaller hole radius spreads the
spectrum more widely. We also demonstrate that an attenuated phase-shift
mask preserves the same |T| but flips the phase outside the open region.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import save_figure, show_field_pair
from src.mask.patterns import contact_hole
from src.mask.transmission import attenuated_phase_shift, binary_transmission
from src.optics.scalar_diffraction import diffraction_spectrum

OUT_DIR = Path("outputs/figures")
N = 256
EXTENT = 1.0
RADII = [0.20, 0.10, 0.05]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    torch.manual_seed(0)
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)

    for r in RADII:
        mask = contact_hole(grid, radius=r)
        spec_b = diffraction_spectrum(binary_transmission(mask))
        fig = show_field_pair(
            mask, spec_b, extent=grid.extent, df=grid.df,
            title=f"contact hole  binary mask  r={r}",
        )
        out = save_figure(fig, OUT_DIR / f"phase1_contact_hole_r_{r:.2f}.png")
        print(f"  wrote {out}")

    # Compare binary vs Att-PSM at one radius
    r = 0.10
    mask = contact_hole(grid, radius=r)
    spec_b = diffraction_spectrum(binary_transmission(mask))
    spec_p = diffraction_spectrum(
        attenuated_phase_shift(mask, attenuation=0.06, phase_rad=math.pi)
    )
    fig_b = show_field_pair(mask, spec_b, extent=grid.extent, df=grid.df,
                            title=f"binary  r={r}")
    fig_p = show_field_pair(mask, spec_p, extent=grid.extent, df=grid.df,
                            title=f"Att-PSM 6%  pi  r={r}")
    save_figure(fig_b, OUT_DIR / f"phase1_contact_hole_binary_r_{r:.2f}.png")
    save_figure(fig_p, OUT_DIR / f"phase1_contact_hole_attpsm_r_{r:.2f}.png")
    print("  wrote binary vs Att-PSM comparison")


if __name__ == "__main__":
    main()

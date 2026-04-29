"""Phase 2 demo: coherent aerial imaging with a circular pupil (NA sweep).

Run:
    python experiments/01_scalar_diffraction/demo_coherent_aerial.py

Produces under ``outputs/figures/``:

- phase2_pupil_NA_sweep.png — circular pupils for the NA values used below
- phase2_pipeline_line_space.png — full mask -> |T| -> P -> |T*P| -> aerial
- phase2_aerial_line_space.png — mask + aerial across NAs
- phase2_aerial_contact_hole.png — same, for an isolated contact hole
- phase2_aerial_isolated_line.png — same, for an isolated line (side-lobe demo)

Physical takeaways:
- Lower NA cuts off high spatial frequencies -> aerial edges blur.
- For a periodic line-space, when NA falls below the fundamental frequency
  (= 1 / pitch in cycles per wavelength), the aerial collapses to a flat DC.
- An isolated contact hole shows the classic Airy / Bessel-J1 ringing whose
  ring spacing scales inversely with NA.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import torch

from src.common.fft_utils import fft2c
from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_aerial_sweep,
    show_pipeline,
)
from src.mask import patterns
from src.mask.transmission import binary_transmission
from src.optics.coherent_imaging import coherent_aerial_image
from src.optics.pupil import circular_pupil

OUT_DIR = Path("outputs/figures")
N = 256
EXTENT = 20.0      # grid spans 20 wavelengths on a side
WAVELENGTH = 1.0   # everything in units of lambda
NAS = [0.2, 0.4, 0.6, 0.8]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pupil_sweep_fig(grid: Grid2D, NAs: list[float]):
    fig, axes = plt.subplots(1, len(NAs), figsize=(4 * len(NAs), 4))
    f_max = (grid.n / 2) * grid.df
    for ax, NA in zip(axes, NAs):
        pupil = circular_pupil(grid, NA=NA, wavelength=WAVELENGTH).cpu().numpy()
        ax.imshow(pupil, cmap="gray",
                  extent=(-f_max, f_max, -f_max, f_max), origin="lower",
                  vmin=0.0, vmax=1.0)
        ax.set_title(f"NA={NA}")
        ax.set_xlabel("fx"); ax.set_ylabel("fy")
    fig.suptitle(f"circular pupil   wavelength={WAVELENGTH}")
    fig.tight_layout()
    return fig


def main() -> None:
    torch.manual_seed(0)
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)

    # 1) Pupil sweep — show the NA-cutoff disks themselves
    fig = _pupil_sweep_fig(grid, NAS)
    out = save_figure(fig, OUT_DIR / "phase2_pupil_NA_sweep.png")
    print(f"  wrote {out}")

    # 2) End-to-end pipeline (mask -> |T| -> P -> |T*P| -> aerial)
    #    Use a line-space at pitch 4 lambda so the fundamental at 0.25 cycles/
    #    lambda is well inside NA=0.6, while the 3rd order at 0.75 sits right
    #    at the edge — ideal for showing how the pupil truncates orders.
    pitch = 4.0
    mask_ls = patterns.line_space(grid, pitch=pitch, duty=0.5, orientation="vertical")
    t_ls = binary_transmission(mask_ls)
    pupil_ref = circular_pupil(grid, NA=0.6, wavelength=WAVELENGTH)
    spectrum_ls = fft2c(t_ls)
    filtered_ls = spectrum_ls * pupil_ref
    aerial_ls = coherent_aerial_image(t_ls, pupil_ref, normalize=True)
    fig = show_pipeline(
        mask_ls, spectrum_ls, pupil_ref, filtered_ls, aerial_ls,
        extent=grid.extent, df=grid.df, freq_zoom=32,
        suptitle=f"pipeline   line-space pitch={pitch} lambda   NA=0.6",
    )
    out = save_figure(fig, OUT_DIR / "phase2_pipeline_line_space.png")
    print(f"  wrote {out}")

    # 3) Aerial sweeps over NA — three target patterns
    targets = [
        ("line_space",
         patterns.line_space(grid, pitch=4.0, duty=0.5, orientation="vertical"),
         "line-space  pitch=4 lambda  duty=0.5"),
        ("contact_hole",
         patterns.contact_hole(grid, radius=0.5),
         "contact hole  r=0.5 lambda"),
        ("isolated_line",
         patterns.isolated_line(grid, width=0.5, orientation="vertical"),
         "isolated line  width=0.5 lambda"),
    ]
    for name, mask, suptitle in targets:
        t = binary_transmission(mask)
        aerials = [
            coherent_aerial_image(t, circular_pupil(grid, NA=NA, wavelength=WAVELENGTH),
                                  normalize=True)
            for NA in NAS
        ]
        fig = show_aerial_sweep(
            mask, aerials, NAS, extent=grid.extent,
            suptitle=f"{suptitle}   wavelength={WAVELENGTH}",
        )
        out = save_figure(fig, OUT_DIR / f"phase2_aerial_{name}.png")
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()

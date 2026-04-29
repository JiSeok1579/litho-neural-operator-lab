"""Phase 4 demo: partial-coherent imaging across source shapes.

Run:
    python experiments/03_partial_coherence/demo_source_shapes.py

Two mask targets are imaged through five source shapes (coherent,
annular, dipole-x, dipole-y, quadrupole). All sources use the same
NA=0.4 / wavelength=1 setting; only the angular distribution of the
illumination changes.

Saved figures (in ``outputs/figures/``):

- phase4_sources.png — the source shapes themselves
- phase4_aerial_line_space.png — vertical line-space (pitch 1.5 lambda)
  imaged under each source
- phase4_aerial_contact_hole.png — contact hole (r 0.5 lambda) imaged
  under each source

Saved metrics (in ``outputs/logs/``):

- phase4_metrics.csv — image contrast, peak intensity, integrated
  leakage in the off-target ring for every (mask, source) pair

Physical takeaways verified by the run:
- Vertical line-space at pitch 1.5 lambda has its fundamental at
  fx=0.667 cycles/lambda. NA=0.4 cannot reach this on-axis. Dipole-x
  shifts the pupil center to ~+/-0.28 cycles/lambda, allowing the right
  pole to capture the +1 order. The contrast jumps an order of
  magnitude versus coherent / dipole-y.
- Contact hole shows mild side-lobe / contrast trades between coherent
  and annular illumination — annular flattens the central peak and
  spreads energy out (lower peak, lower side-lobe contrast).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.metrics import (
    image_contrast,
    integrated_leakage,
    peak_intensity_in_region,
)
from src.common.visualization import (
    save_figure,
    show_partial_coherence_sweep,
    show_source,
)
from src.mask import patterns
from src.mask.transmission import binary_transmission
from src.optics.partial_coherence import partial_coherent_aerial_image
from src.optics.source import (
    annular_source,
    coherent_source,
    dipole_source,
    quadrupole_source,
)

OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
N = 256
EXTENT = 20.0
WAVELENGTH = 1.0
NA = 0.4
N_SIGMA = 31
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_sources():
    return [
        ("coherent",   coherent_source(N_SIGMA)),
        ("annular",    annular_source(N_SIGMA, sigma_inner=0.40, sigma_outer=0.80)),
        ("dipole-x",   dipole_source(N_SIGMA, sigma_center=0.70, sigma_radius=0.20, axis="x")),
        ("dipole-y",   dipole_source(N_SIGMA, sigma_center=0.70, sigma_radius=0.20, axis="y")),
        ("quadrupole", quadrupole_source(N_SIGMA, sigma_center=0.60, sigma_radius=0.20)),
    ]


def _ring_around_center(grid: Grid2D, inner: float, outer: float) -> torch.Tensor:
    """Annular off-target region for leakage measurement around the contact hole."""
    inner_disk = patterns.contact_hole(grid, radius=outer)
    inner_hole = patterns.contact_hole(grid, radius=inner)
    return (inner_disk - inner_hole).clamp(min=0.0)


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    sources = _build_sources()

    # 1) Save the source shapes themselves as one combined figure.
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(sources), figsize=(4 * len(sources), 4))
    sigma_ext = (-1.0, 1.0, -1.0, 1.0)
    for ax, (name, s) in zip(axes, sources):
        ax.imshow(s.cpu().numpy(), cmap="gray", extent=sigma_ext, origin="lower",
                  vmin=0.0, vmax=max(float(s.max()), 1e-6))
        ax.set_title(f"{name}  (npix={int(s.sum().item())})")
        ax.set_xlabel("sigma_x"); ax.set_ylabel("sigma_y")
        ax.set_aspect("equal")
    fig.suptitle(f"phase 4 source shapes  NA={NA}")
    fig.tight_layout()
    out = save_figure(fig, OUT_FIG / "phase4_sources.png")
    print(f"  wrote {out}")

    # 2) Image two target masks through every source.
    targets = [
        ("line_space",
         patterns.line_space(grid, pitch=1.5, duty=0.5, orientation="vertical"),
         f"vertical line-space  pitch=1.5 lambda  NA={NA}"),
        ("contact_hole",
         patterns.contact_hole(grid, radius=0.5),
         f"contact hole  r=0.5 lambda  NA={NA}"),
    ]

    rows = [["mask", "source", "n_src_points", "peak_in_target",
             "contrast_full_field", "leakage_ring_3to6_lambda"]]
    for tag, mask, suptitle in targets:
        t = binary_transmission(mask)
        target_region = mask.to(torch.float32)
        leakage_region = _ring_around_center(grid, inner=3.0, outer=6.0).to(DEVICE)

        srcs_here, names_here, aerials = [], [], []
        for name, src in sources:
            src = src.to(DEVICE)
            aerial = partial_coherent_aerial_image(
                t, grid, src, NA=NA, wavelength=WAVELENGTH, normalize=True
            )
            aerials.append(aerial)
            srcs_here.append(src)
            names_here.append(name)
            with torch.no_grad():
                rows.append([
                    tag,
                    name,
                    int(src.sum().item()),
                    f"{peak_intensity_in_region(aerial, target_region).item():.4f}",
                    f"{image_contrast(aerial).item():.4f}",
                    f"{integrated_leakage(aerial, leakage_region).item():.4f}",
                ])

        fig = show_partial_coherence_sweep(
            mask, srcs_here, aerials, names_here,
            extent=grid.extent, sigma_max=1.0, suptitle=suptitle,
        )
        out = save_figure(fig, OUT_FIG / f"phase4_aerial_{tag}.png")
        print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log_path = OUT_LOG / "phase4_metrics.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_path}")

    # Pretty-print the metrics table to stdout.
    print()
    print("metrics:")
    col_widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, col_widths)))


if __name__ == "__main__":
    main()

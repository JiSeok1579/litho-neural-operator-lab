"""Phase 3 demo: target spot with explicit forbidden side-lobe regions.

Run:
    python experiments/02_inverse_aerial/demo_forbidden_region.py

Same target as ``demo_target_spot`` (central disk of radius 1.0 lambda),
but the forbidden region is now a few small disks placed where naive
diffraction tends to deposit side-lobe energy. The neutral background
(neither target nor forbidden) is left unconstrained, so the optimizer
can park stray energy there if it must.

Saves under ``outputs/figures/``:

- phase3_forbidden_region_before_after.png
- phase3_forbidden_region_loss.png

Physical takeaway: explicit forbidden zones change the trade-off — the
background loss focuses gradient pressure on a few high-leakage spots,
which usually makes the target a bit weaker but suppresses the worst
side-lobes more aggressively than the "everything-is-forbidden" version.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_inverse_result,
    show_loss_history,
)
from src.inverse.optimize_mask import (
    LossWeights,
    OptimizationConfig,
    optimize_mask,
)
from src.mask import patterns

OUT_DIR = Path("outputs/figures")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_regions(grid: Grid2D):
    """Target = central disk; forbidden = four off-axis disks at +/- 3 lambda."""
    target = patterns.contact_hole(grid, radius=1.0, center=(0.0, 0.0))

    forbidden = torch.zeros((grid.n, grid.n), dtype=grid.dtype, device=grid.device)
    for cx, cy in [(3.0, 0.0), (-3.0, 0.0), (0.0, 3.0), (0.0, -3.0)]:
        forbidden = torch.maximum(forbidden, patterns.contact_hole(grid, radius=0.8, center=(cx, cy)))

    # Target and forbidden must not overlap (they don't here, but be safe).
    forbidden = forbidden * (1.0 - target)
    return target.to(DEVICE), forbidden.to(DEVICE)


def main() -> None:
    grid = Grid2D(n=128, extent=10.0, device=DEVICE)
    target_region, forbidden_region = _build_regions(grid)

    cfg = OptimizationConfig(
        n_iters=800,
        lr=5e-2,
        NA=0.4,
        wavelength=1.0,
        target_value=1.0,
        # Bump the background weight: the forbidden region is now small in
        # area, so an unweighted MSE underweights it relative to target.
        weights=LossWeights(target=1.0, background=4.0, tv=1e-3, binarization=0.0),
        alpha_schedule=((0, 1.0), (300, 4.0), (600, 12.0)),
        log_every=10,
        seed=0,
    )

    res = optimize_mask(grid, target_region, forbidden_region, config=cfg)

    init_mask = torch.full_like(res.mask, 0.5)
    fig = show_inverse_result(
        initial_mask=init_mask,
        final_mask=res.mask,
        initial_aerial=res.aerial_initial,
        final_aerial=res.aerial_final,
        target_region=res.target_region,
        forbidden_region=res.forbidden_region,
        extent=grid.extent,
        target_value=cfg.target_value,
        suptitle=(
            f"forbidden regions (4 side-lobe disks)   NA={cfg.NA}   "
            f"alpha={cfg.alpha_schedule[-1][1]:g} (annealed)"
        ),
    )
    out = save_figure(fig, OUT_DIR / "phase3_forbidden_region_before_after.png")
    print(f"  wrote {out}")

    fig = show_loss_history(res.history,
                            suptitle="forbidden regions — loss components and region intensities")
    out = save_figure(fig, OUT_DIR / "phase3_forbidden_region_loss.png")
    print(f"  wrote {out}")

    h0, h_last = res.history[0], res.history[-1]
    print(
        "  summary: target loss "
        f"{h0['loss_target']:.4f} -> {h_last['loss_target']:.4f}, "
        f"bg loss {h0['loss_background']:.4f} -> {h_last['loss_background']:.4f}, "
        f"mean(I) target {h0['mean_intensity_target']:.3f} -> "
        f"{h_last['mean_intensity_target']:.3f}, "
        f"mean(I) forbidden {h0['mean_intensity_background']:.3f} -> "
        f"{h_last['mean_intensity_background']:.3f}"
    )


if __name__ == "__main__":
    main()

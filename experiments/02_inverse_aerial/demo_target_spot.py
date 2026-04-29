"""Phase 3 demo: optimize a mask to make a small bright target spot.

Run:
    python experiments/02_inverse_aerial/demo_target_spot.py

Setup matches study plan §3.6 (experiment 3-A) plus the binarization
annealing schedule from §3.8:

- target: a central disk of radius 1.0 lambda
- forbidden: everywhere outside the target (no neutral buffer)
- NA = 0.4, wavelength = 1.0
- Adam, lr = 5e-2
- 800 iterations
- alpha schedule (0, 1) -> (300, 4) -> (600, 12)

Saves under ``outputs/figures/``:

- phase3_target_spot_before_after.png — initial vs final mask, aerial,
  cross-section, and region overlay.
- phase3_target_spot_loss.png — per-component loss + per-region mean
  intensity vs iteration.

Physical takeaway:
- A 1.0 lambda target is sub-Rayleigh at NA=0.4 (Rayleigh ~ 1.22 lambda /
  NA = 3.05 lambda), so the optimizer can never make a perfectly tight
  spot. It instead finds a mask whose diffraction concentrates as much
  energy as possible inside the target while suppressing the background.
- Binarization annealing degrades the (continuous) optimum but produces a
  manufacturable {0, 1} mask.
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


def main() -> None:
    grid = Grid2D(n=128, extent=10.0, device=DEVICE)

    target_region = patterns.contact_hole(grid, radius=1.0).to(DEVICE)
    forbidden_region = (1.0 - target_region).to(DEVICE)

    cfg = OptimizationConfig(
        n_iters=800,
        lr=5e-2,
        NA=0.4,
        wavelength=1.0,
        target_value=1.0,
        weights=LossWeights(target=1.0, background=1.0, tv=1e-3, binarization=0.0),
        alpha_schedule=((0, 1.0), (300, 4.0), (600, 12.0)),
        log_every=10,
        seed=0,
    )

    res = optimize_mask(grid, target_region, forbidden_region, config=cfg)

    init_mask = torch.full_like(res.mask, 0.5)  # theta=0 -> sigmoid(0)=0.5
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
            f"target spot   NA={cfg.NA}   target r=1 lambda   "
            f"alpha={cfg.alpha_schedule[-1][1]:g} (annealed)"
        ),
    )
    out = save_figure(fig, OUT_DIR / "phase3_target_spot_before_after.png")
    print(f"  wrote {out}")

    fig = show_loss_history(res.history,
                            suptitle="target spot — loss components and region intensities")
    out = save_figure(fig, OUT_DIR / "phase3_target_spot_loss.png")
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

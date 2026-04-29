"""Phase 9 demo: closed-loop surrogate-assisted inverse mask design.

Run:
    python experiments/07_closed_loop_inverse/optimize_with_fno.py

Three optimizations are run on the same target / forbidden / NA setting,
differing only in which 3D-mask correction sits between the thin-mask
spectrum and the pupil:

    A) physics-only        (no correction; T_3d = T_thin)
    B) true correction     (oracle: T_3d = T_thin * correction_operator(theta))
    C) FNO surrogate       (T_3d = T_thin + FNO_pred(mask, T_thin, theta))

Then comes the headline check: take case (C)'s optimized mask and
evaluate it under the *true* correction (B). If the FNO surrogate has
been "fooled" by the optimizer, this re-evaluation will look much worse
than what the optimizer believed during case (C).

Saves under ``outputs/figures/``:
- phase9_closed_loop_comparison.png   — A / B / C / C-validated rows
- phase9_loss_history.png             — loss + region-intensity curves

Saves under ``outputs/logs/``:
- phase9_metrics.csv                   — per-case mean target / forbidden
                                         intensities and loss endpoints

Physical takeaway (verified by the metrics): the FNO surrogate's
optimizer can find a mask that the *predicted* aerial says is great,
but when re-imaged through the true correction the leakage / target
fidelity has degraded. That is why study plan §9.6 demands the
re-validation step every time a surrogate is used inside an
optimization loop.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from src.closed_loop.surrogate_optimizer import (
    SurrogateOptimizationConfig,
    evaluate_mask_under_correction,
    fno_correction_fn,
    identity_correction_fn,
    optimize_mask_with_correction,
    true_correction_fn,
)
from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_closed_loop_comparison,
    show_loss_history,
)
from src.inverse.optimize_mask import LossWeights
from src.mask import patterns
from src.neural_operator.datasets import CorrectionDataset
from src.neural_operator.fno2d import FNO2d
from src.neural_operator.synthetic_3d_correction_data import CorrectionParams

OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
CKPT = Path("outputs/checkpoints/fno_correction.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Imaging / target setup (matches Phase-3 demos)
N = 128
EXTENT = 10.0
NA = 0.4
WAVELENGTH = 1.0
TARGET_RADIUS = 1.0   # lambda
TARGET_VALUE = 1.0

# Strong correction so cases (A) / (B) / (C) clearly differ.
TRUE_PARAMS = CorrectionParams(
    gamma=0.20, alpha=0.30, beta=-0.20,
    delta=0.10, s=0.30, c=2.0,
)

OPT_CFG = SurrogateOptimizationConfig(
    n_iters=600,
    lr=5e-2,
    NA=NA,
    wavelength=WAVELENGTH,
    target_value=TARGET_VALUE,
    weights=LossWeights(target=1.0, background=1.0, tv=1e-3, binarization=0.0),
    alpha_schedule=((0, 1.0), (200, 4.0), (450, 12.0)),
    log_every=10,
    seed=0,
)


def _load_fno() -> FNO2d:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"FNO checkpoint not found at {CKPT}. Run "
            "experiments/06_fno_correction/train_fno_correction.py first."
        )
    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    cfg = blob["config"]
    fno = FNO2d(
        in_channels=cfg["in_channels"], out_channels=cfg["out_channels"],
        hidden=cfg["hidden"], modes_x=cfg["modes_x"], modes_y=cfg["modes_y"],
        n_layers=cfg["n_layers"],
    ).to(DEVICE)
    fno.load_state_dict(blob["state_dict"])
    fno.eval()
    print(f"  loaded FNO: {fno.num_parameters():,} params")
    print(f"  test spectrum MSE  : {blob['train_result']['final_test_spectrum_mse']:.3e}")
    print(f"  test complex rel err: {blob['train_result']['final_test_complex_rel_err']:.3f}")
    return fno


def _setup_regions(grid: Grid2D):
    target = patterns.contact_hole(grid, radius=TARGET_RADIUS).to(DEVICE)
    forbidden = (1.0 - target).to(DEVICE)
    return target, forbidden


def _stats(label: str, mean_target: float, mean_bg: float,
           target_loss: float, bg_loss: float) -> dict:
    return {
        "case": label,
        "mean_intensity_target": f"{mean_target:.4f}",
        "mean_intensity_forbidden": f"{mean_bg:.4f}",
        "loss_target": f"{target_loss:.4f}",
        "loss_background": f"{bg_loss:.4f}",
    }


def main() -> None:
    grid = Grid2D(n=N, extent=EXTENT, device=DEVICE)
    target, forbidden = _setup_regions(grid)

    print("Phase 9 setup:")
    print(f"  grid={grid.n}x{grid.n}, extent={grid.extent} lambda, NA={NA}")
    print(f"  target = central disk r={TARGET_RADIUS} lambda")
    print(f"  forbidden = complement")
    print(f"  true correction theta = {TRUE_PARAMS}")

    fno = _load_fno()

    # --- (A) physics-only ---------------------------------------------------
    print()
    print("(A) optimize physics-only (no correction)")
    cfn_A = identity_correction_fn()
    res_A = optimize_mask_with_correction(grid, target, forbidden, cfn_A, OPT_CFG)
    last_A = res_A.history[-1]
    print(f"    mean(I) target {last_A['mean_intensity_target']:.3f}, "
          f"forbidden {last_A['mean_intensity_background']:.3f}")

    # --- (B) true correction ------------------------------------------------
    print()
    print("(B) optimize with true correction (oracle)")
    cfn_B = true_correction_fn(grid, TRUE_PARAMS)
    res_B = optimize_mask_with_correction(grid, target, forbidden, cfn_B, OPT_CFG)
    last_B = res_B.history[-1]
    print(f"    mean(I) target {last_B['mean_intensity_target']:.3f}, "
          f"forbidden {last_B['mean_intensity_background']:.3f}")

    # --- (C) FNO surrogate --------------------------------------------------
    print()
    print("(C) optimize with FNO surrogate correction")
    cfn_C = fno_correction_fn(fno, TRUE_PARAMS, device=DEVICE)
    res_C = optimize_mask_with_correction(grid, target, forbidden, cfn_C, OPT_CFG)
    last_C = res_C.history[-1]
    print(f"    mean(I) target {last_C['mean_intensity_target']:.3f}, "
          f"forbidden {last_C['mean_intensity_background']:.3f}  "
          "(as predicted by FNO)")

    # --- (C-validated) re-evaluate (C)'s mask under TRUE correction --------
    print()
    print("(C-validated) re-image (C)'s mask under true correction")
    aerial_C_under_true = evaluate_mask_under_correction(
        grid, res_C.mask, true_correction_fn(grid, TRUE_PARAMS),
        NA=NA, wavelength=WAVELENGTH,
    )
    # Compute the same diagnostic numbers under truth.
    with torch.no_grad():
        from src.inverse.losses import (background_loss as _bg,
                                         target_loss as _tg,
                                         mean_intensity_in_region as _mi)
        true_target_loss = _tg(aerial_C_under_true, target, TARGET_VALUE).item()
        true_bg_loss = _bg(aerial_C_under_true, forbidden).item()
        true_mean_target = _mi(aerial_C_under_true, target).item()
        true_mean_bg = _mi(aerial_C_under_true, forbidden).item()
    print(f"    mean(I) target {true_mean_target:.3f}, "
          f"forbidden {true_mean_bg:.3f}  (under true correction)")

    # --- comparison figure --------------------------------------------------
    rows = [
        {
            "label": "A physics-only",
            "mask": res_A.mask,
            "aerial": res_A.aerial_final,
            "target_region": target, "forbidden_region": forbidden,
        },
        {
            "label": "B true correction",
            "mask": res_B.mask,
            "aerial": res_B.aerial_final,
            "target_region": target, "forbidden_region": forbidden,
        },
        {
            "label": "C FNO (predicted)",
            "mask": res_C.mask,
            "aerial": res_C.aerial_final,
            "target_region": target, "forbidden_region": forbidden,
        },
        {
            "label": "C validated under true",
            "mask": res_C.mask,
            "aerial": aerial_C_under_true,
            "target_region": target, "forbidden_region": forbidden,
            "aerial_alt": res_C.aerial_final,
            "aerial_alt_label": "FNO predicted",
        },
    ]
    fig = show_closed_loop_comparison(
        rows, extent=grid.extent, target_value=TARGET_VALUE,
        suptitle=(
            f"phase 9 closed-loop comparison   NA={NA}   target r={TARGET_RADIUS} lambda\n"
            f"true theta = {TRUE_PARAMS}"
        ),
    )
    out = save_figure(fig, OUT_FIG / "phase9_closed_loop_comparison.png")
    print(f"  wrote {out}")

    # Loss history figures (A vs B vs C)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for label, res, color in (("A physics", res_A, "C0"),
                               ("B true",     res_B, "C2"),
                               ("C FNO",      res_C, "C3")):
        its = [h["iter"] for h in res.history]
        axes[0].plot(its, [h["loss_total"] for h in res.history],
                     label=f"{label} total loss", color=color)
        axes[1].plot(its, [h["mean_intensity_target"] for h in res.history],
                     label=f"{label} mean(I) target", color=color)
        axes[1].plot(its, [h["mean_intensity_background"] for h in res.history],
                     color=color, linestyle="--")
    axes[0].set_yscale("log"); axes[0].set_title("total loss")
    axes[0].set_xlabel("iter"); axes[0].set_ylabel("loss (log)"); axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].set_title("mean intensity (solid: target, dashed: forbidden)")
    axes[1].set_xlabel("iter"); axes[1].set_ylabel("intensity"); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle("phase 9 loss + region intensities")
    fig.tight_layout()
    out = save_figure(fig, OUT_FIG / "phase9_loss_history.png")
    print(f"  wrote {out}")

    # Metrics CSV
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase9_metrics.csv"
    rows_csv = [["case", "mean_intensity_target", "mean_intensity_forbidden",
                 "loss_target", "loss_background"]]
    rows_csv.append([
        "A physics-only",
        f"{last_A['mean_intensity_target']:.4f}",
        f"{last_A['mean_intensity_background']:.4f}",
        f"{last_A['loss_target']:.4f}",
        f"{last_A['loss_background']:.4f}",
    ])
    rows_csv.append([
        "B true correction",
        f"{last_B['mean_intensity_target']:.4f}",
        f"{last_B['mean_intensity_background']:.4f}",
        f"{last_B['loss_target']:.4f}",
        f"{last_B['loss_background']:.4f}",
    ])
    rows_csv.append([
        "C FNO (predicted)",
        f"{last_C['mean_intensity_target']:.4f}",
        f"{last_C['mean_intensity_background']:.4f}",
        f"{last_C['loss_target']:.4f}",
        f"{last_C['loss_background']:.4f}",
    ])
    rows_csv.append([
        "C validated under true correction",
        f"{true_mean_target:.4f}",
        f"{true_mean_bg:.4f}",
        f"{true_target_loss:.4f}",
        f"{true_bg_loss:.4f}",
    ])
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows_csv)
    print(f"  wrote {log}")

    print()
    print("metrics:")
    widths = [max(len(str(r[c])) for r in rows_csv) for c in range(len(rows_csv[0]))]
    for r in rows_csv:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

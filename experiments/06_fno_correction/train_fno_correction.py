"""Phase 8 demo: train a 2D FNO surrogate for the synthetic correction.

Run:
    python experiments/06_fno_correction/train_fno_correction.py

Pipeline:

1. Load the Phase-7 paired NPZ archives.
2. Build a 2D FNO (in 9 channels = mask + T_thin re/im + 6 theta;
   out 2 channels = delta_T re/im).
3. Train with Adam + step LR for ``n_epochs``.
4. Evaluate on the test split — spectrum MSE, complex relative error,
   and downstream aerial-intensity MSE (the latter checks that the
   spectrum prediction also makes the imaging output close).
5. Save the best checkpoint, the metrics CSV, the loss curve figure,
   and a 3-row prediction-vs-truth figure.

Saves:
- outputs/checkpoints/fno_correction.pt
- outputs/figures/phase8_fno_training.png
- outputs/figures/phase8_fno_predictions.png
- outputs/logs/phase8_metrics.csv
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.utils.data import DataLoader

from src.common.grid import Grid2D
from src.common.visualization import (
    save_figure,
    show_fno_predictions,
    show_fno_training,
)
from src.neural_operator.datasets import CorrectionDataset
from src.neural_operator.fno2d import FNO2d
from src.neural_operator.synthetic_3d_correction_data import (
    CorrectionParams,
    THETA_NAMES,
)
from src.neural_operator.train_fno import (
    evaluate_fno,
    train_fno_correction,
)

OUT_FIG = Path("outputs/figures")
OUT_LOG = Path("outputs/logs")
OUT_CKPT = Path("outputs/checkpoints")

TRAIN_PATH = Path("outputs/datasets/synthetic_3d_correction_train.npz")
TEST_PATH = Path("outputs/datasets/synthetic_3d_correction_test.npz")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture
HIDDEN = 32
MODES_X = 16
MODES_Y = 16
N_LAYERS = 4

# Training
N_EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 8
LR_DECAY_STEP = 35
LR_DECAY_GAMMA = 0.5
WEIGHT_AERIAL = 0.0
WEIGHT_DECAY = 1e-4   # AdamW regularization to fight overfitting on 800 train samples


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Phase-7 datasets not found. Run "
            "experiments/06_fno_correction/generate_synthetic_dataset.py first."
        )

    print(f"loading datasets from {TRAIN_PATH.name} / {TEST_PATH.name}")
    train_ds = CorrectionDataset(TRAIN_PATH, target="delta_T", device=DEVICE)
    test_ds = CorrectionDataset(TEST_PATH, target="delta_T", device=DEVICE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    grid = Grid2D(n=train_ds.grid_n, extent=train_ds.grid_extent)
    print(f"  grid {grid.n}x{grid.n}, extent {grid.extent} lambda")
    print(f"  train n={len(train_ds)}, test n={len(test_ds)}")

    model = FNO2d(
        in_channels=CorrectionDataset.INPUT_CHANNELS_PER_SAMPLE,
        out_channels=CorrectionDataset.OUTPUT_CHANNELS,
        hidden=HIDDEN, modes_x=MODES_X, modes_y=MODES_Y, n_layers=N_LAYERS,
    ).to(DEVICE)
    print(f"  FNO has {model.num_parameters():,} parameters")

    print(f"training {N_EPOCHS} epochs, lr={LR} (step decay every {LR_DECAY_STEP}), "
          f"weight_decay={WEIGHT_DECAY}")
    res = train_fno_correction(
        model, train_loader, test_loader,
        n_epochs=N_EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY,
        weight_aerial=WEIGHT_AERIAL,
        lr_decay_step=LR_DECAY_STEP, lr_decay_gamma=LR_DECAY_GAMMA,
        device=DEVICE, log_every=2,
    )
    print(f"  trained in {res.train_time_sec:.1f} s")
    print(f"  final lr {res.final_lr:.2e}")

    # Final evaluation
    final_mse, final_rel, final_aerial = evaluate_fno(
        model, test_loader, device=DEVICE, weight_aerial=WEIGHT_AERIAL,
    )
    print(f"  test spectrum MSE   : {final_mse:.6e}")
    print(f"  test complex rel err: {final_rel:.6e}")
    print(f"  test aerial MSE     : {final_aerial:.6e}")

    # Identity baseline (predict delta_T = 0): test spectrum MSE is just
    # the MS of the truth. Computing it from the dataset arrays directly:
    with torch.no_grad():
        T_3d_re = test_ds.T_3d_real
        T_3d_im = test_ds.T_3d_imag
        T_thin_re = test_ds.T_thin_real
        T_thin_im = test_ds.T_thin_imag
        delta_re = T_3d_re - T_thin_re
        delta_im = T_3d_im - T_thin_im
        baseline_mse = float(((delta_re ** 2 + delta_im ** 2) / 2.0).mean().item())
    print(f"  baseline (pred=0)   : {baseline_mse:.6e}  (improvement: "
          f"{baseline_mse / max(final_mse, 1e-30):.1f}x)")

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / "fno_correction.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "hidden": HIDDEN, "modes_x": MODES_X, "modes_y": MODES_Y,
            "n_layers": N_LAYERS,
            "in_channels": CorrectionDataset.INPUT_CHANNELS_PER_SAMPLE,
            "out_channels": CorrectionDataset.OUTPUT_CHANNELS,
        },
        "train_result": {
            "train_time_sec": res.train_time_sec,
            "final_test_spectrum_mse": final_mse,
            "final_test_complex_rel_err": final_rel,
            "final_test_aerial_mse": final_aerial,
            "baseline_spectrum_mse": baseline_mse,
        },
    }, ckpt_path)
    print(f"  saved {ckpt_path}")

    # Training history figure
    fig = show_fno_training(
        res.epochs, res.train_losses, res.test_losses,
        test_complex_rel_err=res.test_complex_rel_err,
        suptitle=f"phase 8 FNO training   final test MSE {final_mse:.3e}",
    )
    out = save_figure(fig, OUT_FIG / "phase8_fno_training.png")
    print(f"  wrote {out}")

    # Predictions on a few test samples
    model.eval()
    samples = []
    n_show = 3
    indices = [0, len(test_ds) // 2, len(test_ds) - 1][:n_show]
    with torch.no_grad():
        for idx in indices:
            x, y_true = test_ds[idx]
            x_b = x.unsqueeze(0).to(DEVICE)
            y_pred = model(x_b)[0].cpu()
            T_thin = torch.complex(test_ds.T_thin_real[idx].cpu(),
                                   test_ds.T_thin_imag[idx].cpu())
            dt_true = torch.complex(y_true[0].cpu(), y_true[1].cpu())
            dt_pred = torch.complex(y_pred[0], y_pred[1])
            params = CorrectionParams.from_array(test_ds.theta[idx].cpu().numpy())
            params_str = (
                f"gamma={params.gamma:+.2f} alpha={params.alpha:+.2f} "
                f"s={params.s:+.2f}"
            )
            samples.append({
                "delta_T_true": dt_true,
                "delta_T_pred": dt_pred,
                "T_thin": T_thin,
                "params_str": params_str,
            })
    fig = show_fno_predictions(
        samples, df=grid.df, freq_zoom=24,
        suptitle=f"phase 8 FNO predictions   test rel err {final_rel:.3f}",
    )
    out = save_figure(fig, OUT_FIG / "phase8_fno_predictions.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "phase8_metrics.csv"
    rows = [
        ["metric", "value"],
        ["n_train", len(train_ds)],
        ["n_test", len(test_ds)],
        ["grid_n", grid.n],
        ["fno_params", model.num_parameters()],
        ["epochs", N_EPOCHS],
        ["batch_size", BATCH_SIZE],
        ["train_time_sec", f"{res.train_time_sec:.2f}"],
        ["final_test_spectrum_mse", f"{final_mse:.6e}"],
        ["final_test_complex_rel_err", f"{final_rel:.6e}"],
        ["final_test_aerial_mse", f"{final_aerial:.6e}"],
        ["baseline_spectrum_mse_pred_0", f"{baseline_mse:.6e}"],
        ["improvement_over_baseline", f"{baseline_mse / max(final_mse, 1e-30):.2f}x"],
    ]
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")


if __name__ == "__main__":
    main()

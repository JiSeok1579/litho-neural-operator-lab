"""PEB Phase-10 demo: train a 2D FNO surrogate on the safe Phase-9 dataset.

Run:
    python reaction_diffusion_peb/experiments/10_operator_learning_optional/train_fno.py

Loads ``outputs/datasets/peb_phase9_safe_dataset.npz``, packs it into
the FNO's (B, C_in, G, G) input tensor (H0 plus 9 broadcast scalar
parameter channels), trains an FNO2d to predict (H_final, P_final),
and saves the checkpoint plus a loss-history CSV.

Phase 10 is the optional follow-up to Phase 9 — there is no
"correct" surrogate result to gate on, but the demo is set up so the
train/val/test L2 relative errors must (a) decrease through training
and (b) stay finite.

DeepONet is intentionally not implemented here. The plan lists
``DeepONet and / or FNO``; FNO matches the regular-grid 128 x 128
inputs more naturally and matches the architecture the main repo
already uses for its surrogate-learning phase.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import torch
import torch.nn as nn

from reaction_diffusion_peb.src.dataset_builder import load_dataset
from reaction_diffusion_peb.src.fno_surrogate import (
    OUTPUT_FIELD_NAMES,
    build_fno_for_dataset,
    fit_channel_stats,
    make_input_tensor,
    make_output_tensor,
    manual_seed_everything,
    per_channel_relative_l2,
    thresholded_iou,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_CKPT = Path("reaction_diffusion_peb/outputs/checkpoints")

SAFE_PATH = OUT_DATA / "peb_phase9_safe_dataset.npz"

WIDTH = 32
N_BLOCKS = 4
MODES = 16

EPOCHS = 300
BATCH_SIZE = 8
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 20260429


def _select(idx, *tensors):
    return tuple(t[idx] for t in tensors)


def main() -> None:
    if not SAFE_PATH.exists():
        raise FileNotFoundError(
            f"missing safe dataset {SAFE_PATH}; "
            "run experiments/09_dataset_generation/generate_fd_dataset.py first"
        )

    arrays, meta = load_dataset(SAFE_PATH)
    splits = meta["splits"]
    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]
    G = arrays["I"].shape[-1]
    print(f"  loaded {SAFE_PATH.name}: n={arrays['I'].shape[0]} grid={G}x{G}")
    print(f"  splits: train={len(train_idx)} val={len(val_idx)} "
          f"test={len(test_idx)}")

    manual_seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = make_input_tensor(arrays)        # (n, C_in, G, G)
    Y = make_output_tensor(arrays)       # (n, 2, G, G)

    Xtr = X[train_idx]; Ytr = Y[train_idx]
    Xva = X[val_idx];   Yva = Y[val_idx]
    Xte = X[test_idx];  Yte = Y[test_idx]

    in_stats = fit_channel_stats(Xtr)
    out_stats = fit_channel_stats(Ytr)

    Xtr_n = in_stats.normalize(Xtr).to(device)
    Ytr_n = out_stats.normalize(Ytr).to(device)
    Xva_n = in_stats.normalize(Xva).to(device)
    Yva_n = out_stats.normalize(Yva).to(device)
    Xte_n = in_stats.normalize(Xte).to(device)
    # We keep Yte in real units — the per-channel rel-L2 is computed in
    # real units so the number is comparable with the FD outputs.
    Yte_real = Yte.to(device)
    Xva_real = Xva.to(device)
    Yva_real = Yva.to(device)

    model = build_fno_for_dataset(
        width=WIDTH, n_blocks=N_BLOCKS,
        modes_x=MODES, modes_y=MODES,
    ).to(device)
    n_params = model.num_parameters()
    print(f"  FNO2d width={WIDTH} blocks={N_BLOCKS} modes={MODES}  "
          f"params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS, eta_min=1e-5,
    )
    loss_fn = nn.MSELoss()

    rng = np.random.default_rng(SEED)
    n_train = Xtr_n.shape[0]
    history_rows = [["epoch", "lr", "train_loss",
                     "val_relL2_H", "val_relL2_P"]]

    t0 = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = Xtr_n[idx]
            yb = Ytr_n[idx]
            optim.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        epoch_loss /= max(1, n_batches)
        sched.step()

        if (epoch + 1) % max(1, EPOCHS // 30) == 0 or epoch == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred_val_n = model(Xva_n)
                pred_val = out_stats.denormalize(pred_val_n)
                rel = per_channel_relative_l2(pred_val, Yva_real)
            history_rows.append([
                f"{epoch + 1}", f"{sched.get_last_lr()[0]:.4g}",
                f"{epoch_loss:.4e}",
                f"{rel[0].item():.4e}", f"{rel[1].item():.4e}",
            ])
            elapsed = time.perf_counter() - t0
            print(f"  ep {epoch + 1:4d}/{EPOCHS}  loss={epoch_loss:.3e}  "
                  f"val rel-L2 H={rel[0].item():.3e}  "
                  f"P={rel[1].item():.3e}  ({elapsed:.1f}s)")
    train_wall = time.perf_counter() - t0

    # Final test-set evaluation in real units.
    model.eval()
    with torch.no_grad():
        pred_test_n = model(Xte_n)
        pred_test = out_stats.denormalize(pred_test_n)
        test_rel = per_channel_relative_l2(pred_test, Yte_real)
        test_iou = thresholded_iou(pred_test[:, 1:2], Yte_real[:, 1:2])

    print()
    print(f"  training wall-clock: {train_wall:.1f} s")
    for c, name in enumerate(OUTPUT_FIELD_NAMES):
        print(f"  test rel-L2 [{name}]: {test_rel[c].item():.4e}")
    print(f"  test threshold-IoU (P > 0.5): "
          f"mean={test_iou.mean().item():.4f}  "
          f"min={test_iou.min().item():.4f}  "
          f"max={test_iou.max().item():.4f}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase10_fno_training_history.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(history_rows)
    print(f"  wrote {log}")

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / "peb_phase10_fno.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "width": WIDTH, "n_blocks": N_BLOCKS,
                "modes_x": MODES, "modes_y": MODES,
            },
            "in_mean": in_stats.mean.cpu(),
            "in_std": in_stats.std.cpu(),
            "out_mean": out_stats.mean.cpu(),
            "out_std": out_stats.std.cpu(),
            "test_relL2": test_rel.detach().cpu().tolist(),
            "test_iou_mean": float(test_iou.mean().item()),
            "train_wall_s": train_wall,
            "epochs": EPOCHS,
            "seed": SEED,
            "n_params": n_params,
        },
        ckpt_path,
    )
    print(f"  wrote {ckpt_path}")

    summary = {
        "n_params": n_params,
        "epochs": EPOCHS,
        "train_wall_s": train_wall,
        "test_relL2": {
            name: float(test_rel[c].item())
            for c, name in enumerate(OUTPUT_FIELD_NAMES)
        },
        "test_iou": {
            "mean": float(test_iou.mean().item()),
            "min": float(test_iou.min().item()),
            "max": float(test_iou.max().item()),
        },
    }
    sum_path = OUT_LOG / "peb_phase10_fno_training_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {sum_path}")


if __name__ == "__main__":
    main()

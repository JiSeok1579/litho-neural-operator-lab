"""PEB Phase-10 improvement: ablation across data size and target-head choice.

Run:
    python reaction_diffusion_peb/experiments/10_operator_learning_optional/run_phase10_ablation.py

Trains 4 FNO surrogates that span the two factors the Phase-10
post-mortem identified as candidate failure causes:

  - **data size**: 64-sample safe archive (original Phase-9) vs
                   1024-sample safe archive (new Phase-10 prerequisite).
  - **target head**: 2-output (H_final, P_final) vs
                     3-output (H, P, R-logit) where R is supervised
                     by BCE-with-logits.

Each model is trained on its own ``train`` split, validated on its
``val`` split, and finally evaluated on:

  - the matching safe ``test`` split  (in-distribution)
  - the full Phase-9 stiff archive    (OOD; kq trained on [0.5, 5],
                                       evaluated on [100, 1000])

Reported metrics per evaluation slice:

  rel_L2_H, rel_L2_P,           — per-channel L2 relative error
  IoU_from_P,                   — IoU of (predP > 0.5) vs (P_truth > 0.5)
  IoU_from_R_logit,             — IoU of (sigmoid(R_logit) > 0.5) vs R
                                  (None for the 2-output variants)
  area_err_mean / area_err_max, — pixel-count error |A_pred - A_truth|
  p_max_err_mean / p_max_err_max — peak-value error |P_max_pred - P_max_truth|

The acceptance bar from the user is intentionally low and exploratory:
move ``rel_L2_P`` below the original 0.24, get any positive IoU, and
shrink the area error.
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
    OUTPUT_FIELD_NAMES_WITH_R,
    area_error,
    build_fno_for_dataset,
    fit_channel_stats,
    make_input_tensor,
    make_output_tensor,
    make_output_tensor_with_R,
    manual_seed_everything,
    mask_iou_from_logits,
    p_max_error,
    per_channel_relative_l2,
    thresholded_iou,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_CKPT = Path("reaction_diffusion_peb/outputs/checkpoints")

SAFE_SMALL = OUT_DATA / "peb_phase9_safe_dataset.npz"
SAFE_LARGE = OUT_DATA / "peb_phase9_safe_large_dataset.npz"
STIFF = OUT_DATA / "peb_phase9_stiff_dataset.npz"

WIDTH = 32
N_BLOCKS = 4
MODES = 16
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
SEED = 20260430
LAMBDA_R = 1.0


# --------------------------------------------------------------------------
# variant definitions
# --------------------------------------------------------------------------

VARIANTS = [
    {"label": "small_2out", "data": "small", "with_R": False},
    {"label": "small_3out", "data": "small", "with_R": True},
    {"label": "large_2out", "data": "large", "with_R": False},
    {"label": "large_3out", "data": "large", "with_R": True},
]


# --------------------------------------------------------------------------
# train / evaluate one variant
# --------------------------------------------------------------------------

def _build_target_tensor(arrays, with_R: bool) -> torch.Tensor:
    return make_output_tensor_with_R(arrays) if with_R else make_output_tensor(arrays)


def _split_views(X, Y, idx):
    return X[idx], Y[idx]


def _evaluate(
    model, in_stats, out_stats_HP, X, Y, with_R: bool, device,
):
    """Run the model on (X, Y) and return a dict of evaluation metrics
    in real (denormalized) units."""
    Xn = in_stats.normalize(X).to(device)
    Y_dev = Y.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_raw = model(Xn)
        # First two channels are predicted in normalized space.
        pred_HP = out_stats_HP.denormalize(pred_raw[:, :2])
    wall = time.perf_counter() - t0

    # Per-channel rel-L2 against ground-truth H, P.
    rel = per_channel_relative_l2(pred_HP, Y_dev[:, :2])
    rel_L2_H = float(rel[0].item())
    rel_L2_P = float(rel[1].item())

    # IoU computed from P (post-thresholded) — same metric as v1.
    iou_from_P = thresholded_iou(pred_HP[:, 1:2], Y_dev[:, 1:2], threshold=0.5)
    iou_from_P_mean = float(iou_from_P.mean().item())

    # Area / P_max errors (always defined).
    a_err = area_error(pred_HP[:, 1:2], Y_dev[:, 1:2], threshold=0.5)
    p_err = p_max_error(pred_HP[:, 1:2], Y_dev[:, 1:2])

    iou_from_R = None
    if with_R:
        # 3rd channel of pred_raw is R-logit (un-normalized).
        # Ground-truth R lives in Y_dev[:, 2:3] (binary).
        iou_R = mask_iou_from_logits(pred_raw[:, 2:3], Y_dev[:, 2:3])
        iou_from_R = float(iou_R.mean().item())

    return {
        "n_samples": int(X.shape[0]),
        "rel_L2_H": rel_L2_H,
        "rel_L2_P": rel_L2_P,
        "iou_from_P": iou_from_P_mean,
        "iou_from_R_logit": iou_from_R,
        "area_err_mean": float(a_err.mean().item()),
        "area_err_max": float(a_err.max().item()),
        "p_max_err_mean": float(p_err.mean().item()),
        "p_max_err_max": float(p_err.max().item()),
        "wall_s": float(wall),
    }


def train_one_variant(
    label: str, data_path: Path, with_R: bool, device: torch.device,
) -> dict:
    arrays, meta = load_dataset(data_path)
    splits = meta["splits"]
    n_total = arrays["I"].shape[0]
    print(f"\n[{label}] data={data_path.name}  n_total={n_total}  "
          f"with_R={with_R}")

    manual_seed_everything(SEED)

    X = make_input_tensor(arrays)
    Y = _build_target_tensor(arrays, with_R=with_R)

    Xtr, Ytr = _split_views(X, Y, splits["train"])
    Xva, Yva = _split_views(X, Y, splits["val"])
    Xte, Yte = _split_views(X, Y, splits["test"])

    in_stats = fit_channel_stats(Xtr)
    out_stats_HP = fit_channel_stats(Ytr[:, :2])

    Xtr_n = in_stats.normalize(Xtr).to(device)
    Ytr_HP_n = out_stats_HP.normalize(Ytr[:, :2]).to(device)
    Ytr_R = Ytr[:, 2:3].to(device) if with_R else None

    Xva_dev = Xva.to(device)
    Yva_dev = Yva.to(device)

    model = build_fno_for_dataset(
        width=WIDTH, n_blocks=N_BLOCKS,
        modes_x=MODES, modes_y=MODES,
        with_R_head=with_R,
    ).to(device)
    n_params = model.num_parameters()
    print(f"  FNO2d width={WIDTH} blocks={N_BLOCKS} modes={MODES}  "
          f"params={n_params:,}")

    optim = torch.optim.AdamW(model.parameters(), lr=LR,
                              weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=EPOCHS, eta_min=1e-5,
    )
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    rng = np.random.default_rng(SEED)
    n_train = Xtr_n.shape[0]

    t0 = time.perf_counter()
    log_every = max(1, EPOCHS // 10)
    for epoch in range(EPOCHS):
        model.train()
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = Xtr_n[idx]
            yb_HP = Ytr_HP_n[idx]
            optim.zero_grad()
            pred = model(xb)
            loss = mse(pred[:, :2], yb_HP)
            if with_R:
                logit = pred[:, 2:3]
                yb_R = Ytr_R[idx]
                loss = loss + LAMBDA_R * bce(logit, yb_R)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        epoch_loss /= max(1, n_batches)
        sched.step()
        if (epoch + 1) % log_every == 0 or epoch == 0 or epoch == EPOCHS - 1:
            elapsed = time.perf_counter() - t0
            val_metrics = _evaluate(
                model, in_stats, out_stats_HP, Xva, Yva, with_R, device,
            )
            print(f"  ep {epoch + 1:4d}/{EPOCHS}  loss={epoch_loss:.3e}  "
                  f"val rel-L2 P={val_metrics['rel_L2_P']:.3e}  "
                  f"iou_P={val_metrics['iou_from_P']:.3f}"
                  + (f"  iou_R={val_metrics['iou_from_R_logit']:.3f}"
                     if with_R else "")
                  + f"  ({elapsed:.1f}s)")
    train_wall = time.perf_counter() - t0

    # final evaluation
    safe_test = _evaluate(model, in_stats, out_stats_HP, Xte, Yte, with_R, device)

    arrays_stiff, _ = load_dataset(STIFF)
    Xstiff = make_input_tensor(arrays_stiff)
    Ystiff = _build_target_tensor(arrays_stiff, with_R=with_R)
    stiff_full = _evaluate(model, in_stats, out_stats_HP, Xstiff, Ystiff, with_R, device)

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / f"peb_phase10_fno_{label}.pt"
    torch.save({
        "label": label,
        "config": {
            "width": WIDTH, "n_blocks": N_BLOCKS,
            "modes_x": MODES, "modes_y": MODES,
            "with_R_head": with_R,
        },
        "state_dict": model.state_dict(),
        "in_mean": in_stats.mean.cpu(),
        "in_std": in_stats.std.cpu(),
        "out_mean_HP": out_stats_HP.mean.cpu(),
        "out_std_HP": out_stats_HP.std.cpu(),
        "n_params": n_params,
        "train_wall_s": train_wall,
        "data_path": str(data_path),
        "safe_test": safe_test,
        "stiff_full": stiff_full,
    }, ckpt_path)
    print(f"  wrote {ckpt_path}")

    return {
        "label": label,
        "data_path": str(data_path),
        "with_R": with_R,
        "n_train": int(len(splits["train"])),
        "n_params": n_params,
        "train_wall_s": train_wall,
        "safe_test": safe_test,
        "stiff_full": stiff_full,
    }


# --------------------------------------------------------------------------
# main: run all four variants
# --------------------------------------------------------------------------

def main() -> None:
    if not SAFE_SMALL.exists():
        raise FileNotFoundError(f"missing {SAFE_SMALL} — run Phase 9 first")
    if not SAFE_LARGE.exists():
        raise FileNotFoundError(
            f"missing {SAFE_LARGE} — run "
            "experiments/09_dataset_generation/generate_safe_large_dataset.py"
        )
    if not STIFF.exists():
        raise FileNotFoundError(f"missing {STIFF} — run Phase 9 first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = {"small": SAFE_SMALL, "large": SAFE_LARGE}
    results: list[dict] = []
    for v in VARIANTS:
        res = train_one_variant(
            label=v["label"], data_path=paths[v["data"]],
            with_R=v["with_R"], device=device,
        )
        results.append(res)

    # Summary CSV
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    rows = [["variant", "n_train", "with_R", "n_params", "train_wall_s",
             "split",
             "rel_L2_H", "rel_L2_P",
             "iou_from_P", "iou_from_R_logit",
             "area_err_mean", "area_err_max",
             "p_max_err_mean", "p_max_err_max"]]
    for r in results:
        for split_name, m in (("safe_test", r["safe_test"]),
                               ("stiff_full", r["stiff_full"])):
            rows.append([
                r["label"], r["n_train"], r["with_R"],
                r["n_params"], f"{r['train_wall_s']:.2f}",
                split_name,
                f"{m['rel_L2_H']:.4e}", f"{m['rel_L2_P']:.4e}",
                f"{m['iou_from_P']:.4f}",
                "" if m['iou_from_R_logit'] is None
                else f"{m['iou_from_R_logit']:.4f}",
                f"{m['area_err_mean']:.2f}", f"{m['area_err_max']:.2f}",
                f"{m['p_max_err_mean']:.4f}", f"{m['p_max_err_max']:.4f}",
            ])
    log_path = OUT_LOG / "peb_phase10_ablation_summary.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nwrote {log_path}")

    sum_path = OUT_LOG / "peb_phase10_ablation_summary.json"
    with open(sum_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {sum_path}")

    # Compact stdout table
    print("\nablation summary (safe_test):")
    print("  variant       n_train  rel-L2 P  iou_from_P  iou_from_R  area_err_mean  p_max_err_mean")
    for r in results:
        m = r["safe_test"]
        iou_R = "  -- " if m["iou_from_R_logit"] is None \
            else f"{m['iou_from_R_logit']:.3f}"
        print(f"  {r['label']:<13} {r['n_train']:>6}   "
              f"{m['rel_L2_P']:.3e}  {m['iou_from_P']:>9.3f}   {iou_R:>8}   "
              f"{m['area_err_mean']:>11.2f}    {m['p_max_err_mean']:>11.4f}")
    print("\nablation summary (stiff_full / OOD):")
    print("  variant       rel-L2 P  iou_from_P  iou_from_R  area_err_mean")
    for r in results:
        m = r["stiff_full"]
        iou_R = "  -- " if m["iou_from_R_logit"] is None \
            else f"{m['iou_from_R_logit']:.3f}"
        print(f"  {r['label']:<13} {m['rel_L2_P']:.3e}  "
              f"{m['iou_from_P']:>9.3f}   {iou_R:>8}   "
              f"{m['area_err_mean']:>11.2f}")


if __name__ == "__main__":
    main()

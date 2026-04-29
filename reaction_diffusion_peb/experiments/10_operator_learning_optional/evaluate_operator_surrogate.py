"""PEB Phase-10 demo: evaluate the trained FNO against FD truth on
both the safe and stiff Phase-9 datasets.

Run:
    python reaction_diffusion_peb/experiments/10_operator_learning_optional/evaluate_operator_surrogate.py

Loads ``outputs/checkpoints/peb_phase10_fno.pt`` and re-runs the FNO on:

  - the safe-test split  (in-distribution, training kq range)
  - the entire stiff dataset (out-of-distribution, kq in [100, 1000])

Reports per-channel L2 relative error (vs FD truth), threshold IoU on
P > 0.5, and FNO inference wall-clock (vs the FD wall-clock recorded
in Phase 9 metadata when available). Writes a metrics CSV and a
side-by-side figure for the worst-case test sample.

The point is to make the surrogate's failure modes legible — Phase 10
is the optional follow-up to Phase 9 and the outcome is informative
either way.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from reaction_diffusion_peb.src.dataset_builder import load_dataset
from reaction_diffusion_peb.src.fno_surrogate import (
    ChannelStats,
    OUTPUT_FIELD_NAMES,
    build_fno_for_dataset,
    make_input_tensor,
    make_output_tensor,
    per_channel_relative_l2,
    thresholded_iou,
)

OUT_DATA = Path("reaction_diffusion_peb/outputs/datasets")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_CKPT = Path("reaction_diffusion_peb/outputs/checkpoints")
OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")

CKPT_PATH = OUT_CKPT / "peb_phase10_fno.pt"
SAFE_PATH = OUT_DATA / "peb_phase9_safe_dataset.npz"
STIFF_PATH = OUT_DATA / "peb_phase9_stiff_dataset.npz"


def _load_model(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(
            f"missing checkpoint {CKPT_PATH}; run train_fno.py first"
        )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = build_fno_for_dataset(
        width=cfg["width"], n_blocks=cfg["n_blocks"],
        modes_x=cfg["modes_x"], modes_y=cfg["modes_y"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    in_stats = ChannelStats(mean=ckpt["in_mean"], std=ckpt["in_std"])
    out_stats = ChannelStats(mean=ckpt["out_mean"], std=ckpt["out_std"])
    return model, in_stats, out_stats, ckpt


def _evaluate(model, in_stats, out_stats,
              X: torch.Tensor, Y: torch.Tensor, device):
    Xn = in_stats.normalize(X).to(device)
    Y = Y.to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_n = model(Xn)
        pred = out_stats.denormalize(pred_n)
    wall = time.perf_counter() - t0
    rel = per_channel_relative_l2(pred, Y)
    iou = thresholded_iou(pred[:, 1:2], Y[:, 1:2])
    return pred.detach().cpu(), Y.detach().cpu(), rel.detach().cpu(), \
           iou.detach().cpu(), wall


def _plot_worst_case(pred, target, sample_idx: int, dx_nm: float,
                     suptitle: str, out_path: Path) -> Path:
    G = pred.shape[-1]
    extent = (0.0, G * dx_nm, 0.0, G * dx_nm)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    H_max = max(float(target[sample_idx, 0].max().item()),
                float(pred[sample_idx, 0].max().item()), 1e-12)

    panels_h = [
        ("FD H_final", target[sample_idx, 0], "magma", 0.0, H_max),
        ("FNO H_final", pred[sample_idx, 0], "magma", 0.0, H_max),
        ("|FNO - FD| H", (pred[sample_idx, 0] - target[sample_idx, 0]).abs(),
         "inferno", 0.0, max(H_max, 1e-12)),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(panels_h):
        im = axes[0, k].imshow(img.numpy(), cmap=cmap, extent=extent,
                               origin="lower", vmin=lo, vmax=hi)
        axes[0, k].set_title(name)
        axes[0, k].set_xlabel("x [nm]"); axes[0, k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[0, k], fraction=0.046, pad=0.04)

    panels_p = [
        ("FD P_final", target[sample_idx, 1], "Greens", 0.0, 1.0),
        ("FNO P_final", pred[sample_idx, 1], "Greens", 0.0, 1.0),
        ("|FNO - FD| P", (pred[sample_idx, 1] - target[sample_idx, 1]).abs(),
         "inferno", 0.0, 1.0),
    ]
    for k, (name, img, cmap, lo, hi) in enumerate(panels_p):
        im = axes[1, k].imshow(img.numpy(), cmap=cmap, extent=extent,
                               origin="lower", vmin=lo, vmax=hi)
        axes[1, k].set_title(name)
        axes[1, k].set_xlabel("x [nm]"); axes[1, k].set_ylabel("y [nm]")
        fig.colorbar(im, ax=axes[1, k], fraction=0.046, pad=0.04)

    fig.suptitle(suptitle)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, in_stats, out_stats, ckpt = _load_model(device)
    print(f"  loaded {CKPT_PATH}  params={ckpt['n_params']:,}")

    # Safe dataset — test split only (in-distribution).
    arrays_s, meta_s = load_dataset(SAFE_PATH)
    test_idx = meta_s["splits"]["test"]
    Xs = make_input_tensor(arrays_s)[test_idx]
    Ys = make_output_tensor(arrays_s)[test_idx]
    pred_s, Y_s, rel_s, iou_s, wall_s = _evaluate(
        model, in_stats, out_stats, Xs, Ys, device,
    )
    print(f"  safe (test split, n={len(test_idx)}):")
    for c, name in enumerate(OUTPUT_FIELD_NAMES):
        print(f"    rel-L2 [{name}]: {rel_s[c].item():.4e}")
    print(f"    threshold-IoU (P>0.5): "
          f"mean={iou_s.mean().item():.4f}  "
          f"min={iou_s.min().item():.4f}  "
          f"max={iou_s.max().item():.4f}")
    print(f"    FNO inference wall: {wall_s:.4f} s "
          f"({wall_s / max(1, len(test_idx)) * 1000:.2f} ms/sample)")

    # Stiff dataset — full archive (out-of-distribution).
    arrays_t, meta_t = load_dataset(STIFF_PATH)
    Xt = make_input_tensor(arrays_t)
    Yt = make_output_tensor(arrays_t)
    pred_t, Y_t, rel_t, iou_t, wall_t = _evaluate(
        model, in_stats, out_stats, Xt, Yt, device,
    )
    n_stiff = Xt.shape[0]
    print(f"  stiff (full archive, n={n_stiff}):")
    for c, name in enumerate(OUTPUT_FIELD_NAMES):
        print(f"    rel-L2 [{name}]: {rel_t[c].item():.4e}")
    print(f"    threshold-IoU (P>0.5): "
          f"mean={iou_t.mean().item():.4f}  "
          f"min={iou_t.min().item():.4f}  "
          f"max={iou_t.max().item():.4f}")
    print(f"    FNO inference wall: {wall_t:.4f} s "
          f"({wall_t / max(1, n_stiff) * 1000:.2f} ms/sample)")

    # Worst-case sample on safe test split.
    safe_per_sample = (pred_s - Y_s).reshape(pred_s.shape[0], -1).norm(dim=1) \
        / Y_s.reshape(Y_s.shape[0], -1).norm(dim=1).clamp(min=1e-8)
    worst_idx = int(safe_per_sample.argmax().item())
    fig_path = _plot_worst_case(
        pred_s, Y_s, sample_idx=worst_idx, dx_nm=meta_s["grid_spacing_nm"],
        suptitle=(f"Phase 10 worst-case safe-test sample "
                  f"(idx_in_split={worst_idx})  rel-L2="
                  f"{safe_per_sample[worst_idx].item():.3e}"),
        out_path=OUT_FIG / "peb_phase10_fno_worst_case_safe.png",
    )
    print(f"  wrote {fig_path}")

    # Metrics CSV.
    OUT_LOG.mkdir(parents=True, exist_ok=True)
    rows = [["dataset", "n_samples",
             "rel_L2_H_final", "rel_L2_P_final",
             "iou_mean", "iou_min", "iou_max",
             "wall_s", "ms_per_sample"]]
    rows.append([
        "safe_test", str(len(test_idx)),
        f"{rel_s[0].item():.4e}", f"{rel_s[1].item():.4e}",
        f"{iou_s.mean().item():.4f}", f"{iou_s.min().item():.4f}",
        f"{iou_s.max().item():.4f}",
        f"{wall_s:.4f}",
        f"{wall_s / max(1, len(test_idx)) * 1000:.4f}",
    ])
    rows.append([
        "stiff_full", str(n_stiff),
        f"{rel_t[0].item():.4e}", f"{rel_t[1].item():.4e}",
        f"{iou_t.mean().item():.4f}", f"{iou_t.min().item():.4f}",
        f"{iou_t.max().item():.4f}",
        f"{wall_t:.4f}",
        f"{wall_t / max(1, n_stiff) * 1000:.4f}",
    ])
    log_path = OUT_LOG / "peb_phase10_fno_evaluation.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log_path}")

    summary = {
        "safe_test": {
            "n_samples": len(test_idx),
            "rel_L2": {n: float(rel_s[c].item())
                        for c, n in enumerate(OUTPUT_FIELD_NAMES)},
            "iou_mean": float(iou_s.mean().item()),
            "wall_s": float(wall_s),
        },
        "stiff_full": {
            "n_samples": n_stiff,
            "rel_L2": {n: float(rel_t[c].item())
                        for c, n in enumerate(OUTPUT_FIELD_NAMES)},
            "iou_mean": float(iou_t.mean().item()),
            "wall_s": float(wall_t),
        },
    }
    sum_path = OUT_LOG / "peb_phase10_fno_evaluation_summary.json"
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  wrote {sum_path}")

    print()
    print("evaluation summary:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

"""Pre-Phase-7 demo: train two PINNs (with and without bound penalty)
and compare them against the FD truth.

Run:
    python reaction_diffusion_peb/experiments/pre_phase7_diagnostics/run_pinn_bound_penalty.py

Trains:
  - PINN_baseline   weight_bound = 0.0    (recovers Phase-5 PINN)
  - PINN_bounded    weight_bound = 1.0    (the new soft penalty)

Both use the same architecture, seed, IC, and number of iterations.
The only difference is the soft bound penalty on P.

Acceptance targets (per FUTURE_WORK item 1):

  - P_min >= -0.05 with the bound penalty
  - P_max <= 1.05  with the bound penalty
  - max|P_PINN - P_FD| smaller than 0.289 (the Phase-5 baseline)
  - area(P > 0.5) closer to the FD truth (1876 px)

Saves:
    outputs/figures/peb_pre_phase7_pinn_bound_compare.png
    outputs/logs/peb_pre_phase7_pinn_bound_metrics.csv
    outputs/checkpoints/peb_pre_phase7_pinn_bounded.pt
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
    thresholded_area,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import gaussian_spot_acid_callable
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.train_pinn_deprotection import (
    pinn_deprotection_to_grid,
    train_pinn_deprotection,
)
from reaction_diffusion_peb.src.visualization import save_figure

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
OUT_CKPT = Path("reaction_diffusion_peb/outputs/checkpoints")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_NM = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
KLOSS = 0.005
KDEP = 0.5
T_END = 60.0
P_THRESHOLD = 0.5

X_HALF = GRID_SIZE * DX_NM / 2.0
X_RANGE_NM = (-X_HALF, X_HALF)
T_RANGE_S = (0.0, T_END)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHT_BOUND_RUNS = [0.0, 0.001, 0.01, 0.1, 1.0]
N_ITERS = 5000


def _fresh_pinn(H0_fn) -> PINNDeprotection:
    """Reproducible identical-init PINN for fair before/after comparison."""
    return PINNDeprotection(
        DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        hard_ic=True, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        hidden=64, n_hidden_layers=4, n_fourier=16,
        fourier_scale=1.0, activation="tanh", seed=0,
    ).to(DEVICE)


def main() -> None:
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=SIGMA_NM, Hmax=HMAX, eta=ETA, dose=DOSE,
    )
    H0 = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)

    # FD truth
    H_fd, P_fd = evolve_acid_loss_deprotection_fd(
        H0, DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        t_end_s=T_END, dx_nm=DX_NM,
    )
    area_fd = thresholded_area(P_fd, P_threshold=P_THRESHOLD)
    print(f"  FD truth: P_max={P_fd.max().item():.4f}, "
          f"P_min={P_fd.min().item():.4f}, area(P>0.5)={area_fd}")

    rows = [["case", "weight_bound", "train_time_sec", "P_min", "P_max",
             f"area(P>{P_THRESHOLD})", "max_abs_err_P_vs_FD",
             "mean_abs_err_P_vs_FD", "L2_rel_err_P_vs_FD",
             "max_abs_err_H_vs_FD"]]

    pinn_results: list[tuple[float, torch.Tensor, torch.Tensor, dict]] = []

    for w_bound in WEIGHT_BOUND_RUNS:
        print(f"\n  training PINN with weight_bound={w_bound:g} ...")
        pinn = _fresh_pinn(H0_fn)
        history = train_pinn_deprotection(
            pinn, H0_callable=H0_fn,
            x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
            n_iters=N_ITERS, lr=1e-3,
            n_collocation=4096, n_ic=512, n_ic_grid_side=16,
            weight_pde_H=1.0, weight_pde_P=1.0, weight_ic=0.0,
            weight_bound=w_bound,
            lr_decay_step=2000, lr_decay_gamma=0.5,
            log_every=50, seed=0,
        )
        H_pinn, P_pinn = pinn_deprotection_to_grid(
            pinn, GRID_SIZE, DX_NM, T_END, device=DEVICE,
        )
        err_P = (P_pinn - P_fd).abs()
        err_H = (H_pinn - H_fd).abs()
        l2_rel_P = (
            ((P_pinn - P_fd) ** 2).sum().sqrt()
            / ((P_fd ** 2).sum().sqrt() + 1e-12)
        ).item()
        area_pinn = thresholded_area(P_pinn, P_threshold=P_THRESHOLD)
        print(f"    P_min={P_pinn.min().item():.4f}, "
              f"P_max={P_pinn.max().item():.4f}, "
              f"area={area_pinn}, "
              f"max|P-P_FD|={err_P.max().item():.4f}")

        rows.append([
            "baseline" if w_bound == 0.0 else f"bounded_{w_bound:g}",
            f"{w_bound:g}",
            f"{history.train_time_sec:.1f}",
            f"{P_pinn.min().item():.4f}",
            f"{P_pinn.max().item():.4f}",
            f"{area_pinn}",
            f"{err_P.max().item():.4f}",
            f"{err_P.mean().item():.4f}",
            f"{l2_rel_P:.4f}",
            f"{err_H.max().item():.4f}",
        ])
        pinn_results.append((w_bound, H_pinn, P_pinn, {
            "P_min": float(P_pinn.min().item()),
            "P_max": float(P_pinn.max().item()),
            "area": area_pinn,
            "history": history,
        }))

        if w_bound > 0:
            OUT_CKPT.mkdir(parents=True, exist_ok=True)
            ckpt = OUT_CKPT / "peb_pre_phase7_pinn_bounded.pt"
            torch.save({
                "state_dict": pinn.state_dict(),
                "config": {
                    "DH_nm2_s": DH, "kloss_s_inv": KLOSS, "kdep_s_inv": KDEP,
                    "x_range_nm": X_RANGE_NM, "t_range_s": T_RANGE_S,
                    "sigma_nm": SIGMA_NM, "Hmax_mol_dm3": HMAX,
                    "eta": ETA, "dose": DOSE,
                    "hidden": 64, "n_hidden_layers": 4, "n_fourier": 16,
                    "fourier_scale": 1.0,
                    "weight_bound": w_bound,
                },
                "training": {
                    "train_time_sec": history.train_time_sec,
                    "final_loss_pde_H": history.loss_pde_H[-1],
                    "final_loss_pde_P": history.loss_pde_P[-1],
                    "final_loss_bound": history.loss_bound[-1],
                    "max_abs_err_P_vs_FD": float(err_P.max().item()),
                    "max_abs_err_H_vs_FD": float(err_H.max().item()),
                    "P_min": float(P_pinn.min().item()),
                    "P_max": float(P_pinn.max().item()),
                    "area_pinn": area_pinn,
                    "area_fd": area_fd,
                },
            }, ckpt)
            print(f"    saved {ckpt}")

    # Pick the "best bounded" run as the one minimizing
    # max_abs_err_P_vs_FD subject to P_min >= -0.05.
    candidate = None
    for w, _, P_pinn, _ in pinn_results:
        if w == 0.0:
            continue
        if P_pinn.min().item() < -0.05:
            continue
        max_err = float((P_pinn - P_fd).abs().max().item())
        if candidate is None or max_err < candidate[1]:
            candidate = (w, max_err)
    if candidate is not None:
        print(f"\n  best bounded weight: {candidate[0]:g} "
              f"(max|P-P_FD|={candidate[1]:.4f})")

    # FD reference row at the end of the table for easy diff
    rows.append([
        "FD truth", "—", "—",
        f"{P_fd.min().item():.4f}",
        f"{P_fd.max().item():.4f}",
        f"{area_fd}",
        "0.0000",
        "0.0000",
        "0.0000",
        "0.0000",
    ])

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_pre_phase7_pinn_bound_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\n  wrote {log}")

    # ---------------------------------------------------------------
    # Comparison figure: (1 + N_RUNS) rows × 3 cols
    #   Row 0 (FD truth): H_FD | P_FD | thresholded
    #   Row r (run r):    H | P | |P - P_FD|
    # ---------------------------------------------------------------
    n_rows = 1 + len(pinn_results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 4 * n_rows))
    extent_nm = GRID_SIZE * DX_NM
    real_ext = (-extent_nm / 2, extent_nm / 2, -extent_nm / 2, extent_nm / 2)
    H_max_val = max(
        float(H0.max().item()), float(H_fd.max().item()),
        max(float(H.max().item()) for _, H, _, _ in pinn_results),
        1e-12,
    )

    def _to_np(x: torch.Tensor) -> np.ndarray:
        return x.detach().cpu().numpy()

    # Row 0 — FD truth
    im = axes[0, 0].imshow(_to_np(H_fd), cmap="magma", extent=real_ext,
                           origin="lower", vmin=0, vmax=H_max_val)
    axes[0, 0].set_title(f"FD truth  H  peak={H_fd.max().item():.4f}")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)
    im = axes[0, 1].imshow(_to_np(P_fd), cmap="Greens", extent=real_ext,
                           origin="lower", vmin=0, vmax=1)
    axes[0, 1].set_title(f"FD truth  P  P_max={P_fd.max().item():.4f}")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    im = axes[0, 2].imshow(_to_np((P_fd > P_THRESHOLD).float()),
                           cmap="Greens", extent=real_ext,
                           origin="lower", vmin=0, vmax=1)
    axes[0, 2].set_title(f"FD  P > {P_THRESHOLD}  (area {area_fd})")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    for axrow in axes:
        for ax in axrow:
            ax.set_xlabel("x [nm]"); ax.set_ylabel("y [nm]")

    for r, (w_bound, H_pinn, P_pinn, info) in enumerate(pinn_results, start=1):
        label = (f"PINN baseline (w_bound=0)" if w_bound == 0
                 else f"PINN bounded w_bound={w_bound:g}")
        err_P_map = (P_pinn - P_fd).abs()
        err_max = max(float(err_P_map.max().item()), 1e-12)

        im = axes[r, 0].imshow(_to_np(H_pinn), cmap="magma", extent=real_ext,
                               origin="lower", vmin=0, vmax=H_max_val)
        axes[r, 0].set_title(f"{label}\nH PINN  peak={H_pinn.max().item():.4f}")
        fig.colorbar(im, ax=axes[r, 0], fraction=0.046, pad=0.04)
        im = axes[r, 1].imshow(_to_np(P_pinn), cmap="Greens",
                               extent=real_ext, origin="lower",
                               vmin=min(0.0, info["P_min"]),
                               vmax=max(1.0, info["P_max"]))
        axes[r, 1].set_title(
            f"P PINN  P_min={info['P_min']:.3f}  P_max={info['P_max']:.3f}\n"
            f"area(P>{P_THRESHOLD}) = {info['area']}"
        )
        fig.colorbar(im, ax=axes[r, 1], fraction=0.046, pad=0.04)
        im = axes[r, 2].imshow(_to_np(err_P_map), cmap="inferno",
                               extent=real_ext, origin="lower",
                               vmin=0, vmax=err_max)
        axes[r, 2].set_title(f"|P_PINN - P_FD|  max={err_max:.3f}")
        fig.colorbar(im, ax=axes[r, 2], fraction=0.046, pad=0.04)

    fig.suptitle(
        "PEB pre-Phase-7: PINN bound penalty sweep\n"
        "(top row FD truth; subsequent rows weight_bound = "
        + ", ".join(f"{w:g}" for w in WEIGHT_BOUND_RUNS) + ")"
    )
    fig.tight_layout()
    out = save_figure(fig, OUT_FIG / "peb_pre_phase7_pinn_bound_compare.png")
    print(f"  wrote {out}")

    print()
    print("metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

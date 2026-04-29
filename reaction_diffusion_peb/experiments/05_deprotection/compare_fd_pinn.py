"""PEB Phase-5 demo: FD vs PINN side-by-side.

Run:
    python reaction_diffusion_peb/experiments/05_deprotection/compare_fd_pinn.py

Loads the Phase-5 PINN checkpoint, runs the FD evolver on the same
problem, and writes a 2x4 comparison figure plus a metrics CSV.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
    thresholded_area,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import gaussian_spot_acid_callable
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.train_pinn_deprotection import pinn_deprotection_to_grid
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_deprotection_fd_vs_pinn,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
CKPT = Path("reaction_diffusion_peb/outputs/checkpoints/peb_phase5_pinn_deprotection.pt")

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _time_call(fn, *args, n_warmup: int = 1, n_runs: int = 5, **kwargs):
    for _ in range(n_warmup):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_runs):
        out = fn(*args, **kwargs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, (time.time() - t0) / n_runs


def _load_pinn() -> PINNDeprotection:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"PINN checkpoint not found at {CKPT}. Run "
            "reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_pinn.py first."
        )
    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    cfg = blob["config"]
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=cfg["sigma_nm"], Hmax=cfg["Hmax_mol_dm3"],
        eta=cfg["eta"], dose=cfg["dose"],
    )
    pinn = PINNDeprotection(
        DH_nm2_s=cfg["DH_nm2_s"], kloss_s_inv=cfg["kloss_s_inv"],
        kdep_s_inv=cfg["kdep_s_inv"],
        hard_ic=True, H0_callable=H0_fn,
        x_range_nm=cfg["x_range_nm"], t_range_s=cfg["t_range_s"],
        hidden=cfg["hidden"], n_hidden_layers=cfg["n_hidden_layers"],
        n_fourier=cfg["n_fourier"], fourier_scale=cfg["fourier_scale"],
    ).to(DEVICE)
    pinn.load_state_dict(blob["state_dict"])
    pinn.eval()
    train_info = blob.get("training", {})
    print(f"  loaded PINN from {CKPT}")
    print(f"    train time {train_info.get('train_time_sec', '?'):.1f} s, "
          f"final loss_pde_H {train_info.get('final_loss_pde_H', float('nan')):.3e}, "
          f"final loss_pde_P {train_info.get('final_loss_pde_P', float('nan')):.3e}")
    return pinn


def main() -> None:
    pinn = _load_pinn()

    H0 = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)

    (H_fd, P_fd), fd_wall = _time_call(
        evolve_acid_loss_deprotection_fd, H0, DH, KLOSS, KDEP, T_END, DX_NM,
        n_runs=3,
    )
    (H_pinn, P_pinn), pinn_wall = _time_call(
        pinn_deprotection_to_grid, pinn, GRID_SIZE, DX_NM, T_END, n_runs=20,
    )

    err_H = (H_pinn - H_fd).abs()
    err_P = (P_pinn - P_fd).abs()
    area_fd = thresholded_area(P_fd, P_threshold=P_THRESHOLD)
    area_pinn = thresholded_area(P_pinn, P_threshold=P_THRESHOLD)

    rows = [["solver", "max_abs_err_H_vs_FD", "max_abs_err_P_vs_FD",
             "P_max", "P_min", f"area(P>{P_THRESHOLD})_px",
             "wallclock_per_call_ms", "extra_info"]]
    rows.append([
        "FD (truth)", "0.000e+00", "0.000e+00",
        f"{P_fd.max().item():.4f}", f"{P_fd.min().item():.4f}",
        f"{area_fd}",
        f"{fd_wall * 1000:.3f}",
        f"dx={DX_NM} nm, periodic BC",
    ])
    rows.append([
        "PINN",
        f"{float(err_H.max().item()):.3e}",
        f"{float(err_P.max().item()):.3e}",
        f"{P_pinn.max().item():.4f}", f"{P_pinn.min().item():.4f}",
        f"{area_pinn}",
        f"{pinn_wall * 1000:.3f}",
        "after training (see CSV)",
    ])

    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    train_t = blob.get("training", {}).get("train_time_sec", float("nan"))
    rows[-1][-1] = f"after {train_t:.1f} s training"

    fig = show_deprotection_fd_vs_pinn(
        H0, H_fd, P_fd, H_pinn, P_pinn,
        dx_nm=DX_NM, t_end_s=T_END, P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 5: FD (truth) vs PINN   DH={DH}   "
                  f"kloss={KLOSS}   kdep={KDEP}   t={T_END} s"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase5_compare_fd_pinn.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase5_compare_fd_pinn_metrics.csv"
    with open(log, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  wrote {log}")

    print()
    print("metrics:")
    widths = [max(len(str(r[c])) for r in rows) for c in range(len(rows[0]))]
    for r in rows:
        print("  " + "  ".join(str(v).ljust(w) for v, w in zip(r, widths)))


if __name__ == "__main__":
    main()

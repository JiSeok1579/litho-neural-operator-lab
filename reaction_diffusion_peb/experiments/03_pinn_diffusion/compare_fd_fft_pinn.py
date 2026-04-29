"""PEB Phase-3 demo: FD vs FFT vs PINN on the same diffusion problem.

Run:
    python reaction_diffusion_peb/experiments/03_pinn_diffusion/compare_fd_fft_pinn.py

Loads the checkpoint produced by ``run_pinn_diffusion.py`` and runs all
three solvers on the same Gaussian-spot exposure, then writes:

    outputs/figures/peb_phase3_compare_fd_fft_pinn.png
    outputs/logs/peb_phase3_compare_fd_fft_pinn_metrics.csv

FFT is treated as the ground-truth heat-equation solution (it is exact
modulo float precision); FD and PINN errors are measured against it.

Headline lesson (matches study plan §6.8 in the main project): on a
smooth, closed-form-tractable diffusion problem, FFT >> FD >> PINN
on accuracy and speed. The PINN is only "competitive" once the
problem geometry / boundaries / unknown parameters make FD / FFT
inconvenient — which is not the case here.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.diffusion_fd import diffuse_fd
from reaction_diffusion_peb.src.diffusion_fft import diffuse_fft
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import (
    PINNDiffusion,
    gaussian_spot_acid_callable,
    pinn_to_grid,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_fd_fft_pinn,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
CKPT = Path("reaction_diffusion_peb/outputs/checkpoints/peb_phase3_pinn_diffusion.pt")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_NM = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
T_END = 60.0

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


def _load_pinn() -> PINNDiffusion:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"PINN checkpoint not found at {CKPT}. "
            "Run reaction_diffusion_peb/experiments/03_pinn_diffusion/"
            "run_pinn_diffusion.py first."
        )
    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    cfg = blob["config"]
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=cfg["sigma_nm"], Hmax=cfg["Hmax_mol_dm3"],
        eta=cfg["eta"], dose=cfg["dose"],
    )
    pinn = PINNDiffusion(
        DH_nm2_s=cfg["DH_nm2_s"], hard_ic=True, H0_callable=H0_fn,
        x_range_nm=cfg["x_range_nm"], t_range_s=cfg["t_range_s"],
        hidden=cfg["hidden"], n_hidden_layers=cfg["n_hidden_layers"],
        n_fourier=cfg["n_fourier"], fourier_scale=cfg["fourier_scale"],
    ).to(DEVICE)
    pinn.load_state_dict(blob["state_dict"])
    pinn.eval()
    train_info = blob.get("training", {})
    print(f"  loaded PINN from {CKPT}")
    print(f"    train time {train_info.get('train_time_sec', '?'):.1f} s, "
          f"final loss_pde {train_info.get('final_loss_pde', float('nan')):.3e}")
    return pinn


def main() -> None:
    pinn = _load_pinn()

    H0 = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)

    H_fd, fd_wall = _time_call(
        diffuse_fd, H0, DH_nm2_s=DH, t_end_s=T_END, dx_nm=DX_NM, n_runs=3,
    )
    H_fft, fft_wall = _time_call(
        diffuse_fft, H0, DH_nm2_s=DH, t_end_s=T_END, dx_nm=DX_NM, n_runs=20,
    )
    H_pinn, pinn_wall = _time_call(
        pinn_to_grid, pinn, GRID_SIZE, DX_NM, T_END, n_runs=20,
    )

    err_fd = (H_fd - H_fft).abs()
    err_pinn = (H_pinn - H_fft).abs()

    rows = [["solver", "max_abs_err_vs_fft", "mean_abs_err_vs_fft",
             "L2_rel_err_vs_fft", "wallclock_per_call_ms", "extra_info"]]

    def l2_rel(diff: torch.Tensor) -> float:
        num = float((diff ** 2).sum().sqrt().item())
        den = float((H_fft ** 2).sum().sqrt().item()) + 1e-12
        return num / den

    rows.append([
        "FFT", "0.000e+00", "0.000e+00", "0.000e+00",
        f"{fft_wall * 1000:.4f}", "exact heat kernel",
    ])
    rows.append([
        "FD",
        f"{float(err_fd.max().item()):.3e}",
        f"{float(err_fd.mean().item()):.3e}",
        f"{l2_rel(H_fd - H_fft):.3e}",
        f"{fd_wall * 1000:.3f}",
        f"dx={DX_NM} nm, periodic BC",
    ])
    rows.append([
        "PINN",
        f"{float(err_pinn.max().item()):.3e}",
        f"{float(err_pinn.mean().item()):.3e}",
        f"{l2_rel(H_pinn - H_fft):.3e}",
        f"{pinn_wall * 1000:.3f}",
        "after ~10s training",  # filled in below
    ])

    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    train_t = blob.get("training", {}).get("train_time_sec", float("nan"))
    rows[-1][-1] = f"after {train_t:.1f} s training"

    fig = show_fd_fft_pinn(
        H0, H_fd, H_fft, H_pinn, dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 3: FD vs FFT vs PINN   "
                  f"DH={DH} nm^2/s   t={T_END} s   sigma={SIGMA_NM} nm"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase3_compare_fd_fft_pinn.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase3_compare_fd_fft_pinn_metrics.csv"
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

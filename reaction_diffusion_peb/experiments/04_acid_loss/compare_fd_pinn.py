"""PEB Phase-4 demo: FD vs FFT vs PINN with the acid-loss term.

Run:
    python reaction_diffusion_peb/experiments/04_acid_loss/compare_fd_pinn.py

Loads the Phase-4 PINN checkpoint and runs all three solvers
(FD, FFT, PINN) on the same Gaussian-spot exposure with kloss=0.005.
FFT is the truth reference (still exact in closed form for this
linear-loss equation).
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import (
    gaussian_spot_acid_callable,
    pinn_to_grid,
)
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDiffusionLoss
from reaction_diffusion_peb.src.reaction_diffusion import (
    diffuse_acid_loss_fd,
    diffuse_acid_loss_fft,
    expected_mass_decay_factor,
    total_mass,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_fd_fft_pinn,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_LOG = Path("reaction_diffusion_peb/outputs/logs")
CKPT = Path("reaction_diffusion_peb/outputs/checkpoints/peb_phase4_pinn_acid_loss.pt")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_NM = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
KLOSS = 0.005
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


def _load_pinn() -> PINNDiffusionLoss:
    if not CKPT.exists():
        raise FileNotFoundError(
            f"PINN checkpoint not found at {CKPT}. Run "
            "reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_pinn.py first."
        )
    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    cfg = blob["config"]
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=cfg["sigma_nm"], Hmax=cfg["Hmax_mol_dm3"],
        eta=cfg["eta"], dose=cfg["dose"],
    )
    pinn = PINNDiffusionLoss(
        DH_nm2_s=cfg["DH_nm2_s"], kloss_s_inv=cfg["kloss_s_inv"],
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
          f"final loss_pde {train_info.get('final_loss_pde', float('nan')):.3e}")
    return pinn


def main() -> None:
    pinn = _load_pinn()

    H0 = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)

    H_fd, fd_wall = _time_call(
        diffuse_acid_loss_fd, H0, DH, KLOSS, T_END, DX_NM, n_runs=3,
    )
    H_fft, fft_wall = _time_call(
        diffuse_acid_loss_fft, H0, DH, KLOSS, T_END, DX_NM, n_runs=20,
    )
    H_pinn, pinn_wall = _time_call(
        pinn_to_grid, pinn, GRID_SIZE, DX_NM, T_END, n_runs=20,
    )

    err_fd = (H_fd - H_fft).abs()
    err_pinn = (H_pinn - H_fft).abs()

    def l2_rel(diff: torch.Tensor) -> float:
        num = float((diff ** 2).sum().sqrt().item())
        den = float((H_fft ** 2).sum().sqrt().item()) + 1e-12
        return num / den

    rows = [["solver", "max_abs_err_vs_fft", "mean_abs_err_vs_fft",
             "L2_rel_err_vs_fft", "wallclock_per_call_ms",
             "mass", "mass_expected_decay", "extra_info"]]

    m0 = total_mass(H0, DX_NM)
    m_expected = m0 * expected_mass_decay_factor(KLOSS, T_END)

    for label, H, wall, info in [
        ("FFT", H_fft, fft_wall, "exact heat kernel + decay"),
        ("FD", H_fd, fd_wall, f"dx={DX_NM} nm, periodic BC"),
        ("PINN", H_pinn, pinn_wall, "after training (see CSV)"),
    ]:
        diff = (H - H_fft).abs()
        m = total_mass(H, DX_NM)
        rows.append([
            label,
            f"{float(diff.max().item()):.3e}",
            f"{float(diff.mean().item()):.3e}",
            f"{l2_rel(H - H_fft):.3e}",
            f"{wall * 1000:.3f}",
            f"{m:.4f}",
            f"{m_expected:.4f}",
            info,
        ])

    blob = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    train_t = blob.get("training", {}).get("train_time_sec", float("nan"))
    rows[3][-1] = f"after {train_t:.1f} s training"

    fig = show_fd_fft_pinn(
        H0, H_fd, H_fft, H_pinn, dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 4: FD vs FFT vs PINN   "
                  f"DH={DH} nm^2/s   kloss={KLOSS} 1/s   t={T_END} s   "
                  f"M decay {expected_mass_decay_factor(KLOSS, T_END):.3f}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase4_compare_fd_fft_pinn.png")
    print(f"  wrote {out}")

    OUT_LOG.mkdir(parents=True, exist_ok=True)
    log = OUT_LOG / "peb_phase4_compare_fd_pinn_metrics.csv"
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

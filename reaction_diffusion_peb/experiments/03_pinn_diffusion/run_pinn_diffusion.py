"""PEB Phase-3 demo: train a PINN against the diffusion equation.

Run:
    python reaction_diffusion_peb/experiments/03_pinn_diffusion/run_pinn_diffusion.py

Setup matches Phases 1 + 2:
    Gaussian-spot exposure, sigma=12 nm, Hmax=0.2 mol/dm^3
    DH = 0.8 nm^2/s, t_end = 60 s, dx = 1 nm
    PINN: hidden=64, 4 hidden layers, n_fourier=16, fourier_scale=1.0
    hard_ic=True (so IC is exact at t=0 by construction)
    5000 iters of Adam at lr=1e-3 with step LR every 2000 iters

Saves:
    outputs/figures/peb_phase3_pinn_training.png
    outputs/figures/peb_phase3_pinn_vs_fft.png   (3-way comparison row)
    outputs/checkpoints/peb_phase3_pinn_diffusion.pt

The actual FD/FFT/PINN three-way comparison numbers live in the
companion script ``compare_fd_fft_pinn.py``; this script's job is to
train the network and save the checkpoint.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.diffusion_fft import diffuse_fft
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import (
    PINNDiffusion,
    gaussian_spot_acid_callable,
    pinn_to_grid,
    train_pinn_diffusion,
)
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_diffusion_chain,
    show_pinn_training,
)

OUT_FIG = Path("reaction_diffusion_peb/outputs/figures")
OUT_CKPT = Path("reaction_diffusion_peb/outputs/checkpoints")

GRID_SIZE = 128
DX_NM = 1.0
SIGMA_NM = 12.0           # spot sigma in nm (= sigma_px * dx_nm)
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8                  # nm^2/s
T_END = 60.0              # s

X_HALF = GRID_SIZE * DX_NM / 2.0
X_RANGE_NM = (-X_HALF, X_HALF)
T_RANGE_S = (0.0, T_END)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=SIGMA_NM, Hmax=HMAX, eta=ETA, dose=DOSE,
    )

    pinn = PINNDiffusion(
        DH_nm2_s=DH, hard_ic=True, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        hidden=64, n_hidden_layers=4, n_fourier=16,
        fourier_scale=1.0, activation="tanh", seed=0,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)
    print(f"  PINN parameters: {n_params:,}")

    print(f"  training PINN  (DH={DH}, t_end={T_END}, hard_ic=True)...")
    history = train_pinn_diffusion(
        pinn, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        n_iters=5000, lr=1e-3,
        n_collocation=4096, n_ic=512, n_ic_grid_side=16,
        weight_pde=1.0, weight_ic=0.0,
        lr_decay_step=2000, lr_decay_gamma=0.5,
        log_every=50, seed=0,
    )
    print(f"  trained in {history.train_time_sec:.1f} s, "
          f"final loss_total={history.loss_total[-1]:.4e}, "
          f"loss_pde={history.loss_pde[-1]:.4e}")

    fig = show_pinn_training(
        history,
        suptitle=(f"PEB phase 3 PINN training   DH={DH} nm^2/s   "
                  f"sigma={SIGMA_NM} nm   t_end={T_END} s"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase3_pinn_training.png")
    print(f"  wrote {out}")

    # Compare PINN(t_end) against FFT (treated as truth) on the same grid.
    H0_grid = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM), dose=DOSE,
        eta=ETA, Hmax=HMAX,
    ).to(DEVICE)
    t0 = time.time()
    H_pinn = pinn_to_grid(pinn, GRID_SIZE, DX_NM, T_END, device=DEVICE)
    pinn_inf = time.time() - t0
    H_fft = diffuse_fft(H0_grid, DH_nm2_s=DH, t_end_s=T_END, dx_nm=DX_NM)
    err_fft = (H_pinn - H_fft).abs()
    print(f"  PINN inference {pinn_inf*1000:.2f} ms, "
          f"max|PINN - FFT|={float(err_fft.max().item()):.3e}, "
          f"mean|PINN - FFT|={float(err_fft.mean().item()):.3e}")

    fig = show_diffusion_chain(
        H0_grid, H_pinn, dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 3 PINN prediction at t={T_END} s   "
                  f"max|PINN - FFT|={float(err_fft.max().item()):.3e}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase3_pinn_vs_fft.png")
    print(f"  wrote {out}")

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / "peb_phase3_pinn_diffusion.pt"
    torch.save({
        "state_dict": pinn.state_dict(),
        "config": {
            "DH_nm2_s": DH,
            "x_range_nm": X_RANGE_NM,
            "t_range_s": T_RANGE_S,
            "sigma_nm": SIGMA_NM,
            "Hmax_mol_dm3": HMAX,
            "eta": ETA,
            "dose": DOSE,
            "hidden": 64,
            "n_hidden_layers": 4,
            "n_fourier": 16,
            "fourier_scale": 1.0,
        },
        "training": {
            "train_time_sec": history.train_time_sec,
            "final_loss_total": history.loss_total[-1],
            "final_loss_pde": history.loss_pde[-1],
            "max_abs_err_vs_fft": float(err_fft.max().item()),
            "mean_abs_err_vs_fft": float(err_fft.mean().item()),
        },
    }, ckpt_path)
    print(f"  wrote {ckpt_path}")


if __name__ == "__main__":
    main()

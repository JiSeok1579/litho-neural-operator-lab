"""PEB Phase-4 demo: train a PINN against the acid-loss equation.

Run:
    python reaction_diffusion_peb/experiments/04_acid_loss/run_acid_loss_pinn.py

Same problem as ``compare_fd_pinn.py`` but this script's job is to
train and save the checkpoint to
``outputs/checkpoints/peb_phase4_pinn_acid_loss.pt``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import (
    gaussian_spot_acid_callable,
    pinn_to_grid,
    train_pinn_diffusion,
)
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDiffusionLoss
from reaction_diffusion_peb.src.reaction_diffusion import diffuse_acid_loss_fft
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
SIGMA_NM = 12.0
HMAX = 0.2
ETA = 1.0
DOSE = 1.0
DH = 0.8
KLOSS = 0.005
T_END = 60.0

X_HALF = GRID_SIZE * DX_NM / 2.0
X_RANGE_NM = (-X_HALF, X_HALF)
T_RANGE_S = (0.0, T_END)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=SIGMA_NM, Hmax=HMAX, eta=ETA, dose=DOSE,
    )

    pinn = PINNDiffusionLoss(
        DH_nm2_s=DH, kloss_s_inv=KLOSS,
        hard_ic=True, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        hidden=64, n_hidden_layers=4, n_fourier=16,
        fourier_scale=1.0, activation="tanh", seed=0,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)
    print(f"  PINN parameters: {n_params:,}")
    print(f"  training PINN  (DH={DH}, kloss={KLOSS}, t_end={T_END}, hard_ic=True)...")

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
          f"final loss_pde={history.loss_pde[-1]:.4e}")

    fig = show_pinn_training(
        history,
        suptitle=(f"PEB phase 4 PINN training   DH={DH}   kloss={KLOSS}   "
                  f"sigma={SIGMA_NM} nm   t_end={T_END} s"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase4_pinn_training.png")
    print(f"  wrote {out}")

    H0_grid = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)
    H_pinn = pinn_to_grid(pinn, GRID_SIZE, DX_NM, T_END, device=DEVICE)
    H_fft = diffuse_acid_loss_fft(H0_grid, DH, KLOSS, T_END, dx_nm=DX_NM)
    err_fft = (H_pinn - H_fft).abs()
    print(f"  max|PINN - FFT|={float(err_fft.max().item()):.3e}, "
          f"mean|PINN - FFT|={float(err_fft.mean().item()):.3e}")

    fig = show_diffusion_chain(
        H0_grid, H_pinn, dx_nm=DX_NM, t_end_s=T_END, Hmax=HMAX,
        suptitle=(f"PEB phase 4 PINN at t={T_END} s   "
                  f"max|PINN - FFT|={float(err_fft.max().item()):.3e}"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase4_pinn_vs_fft.png")
    print(f"  wrote {out}")

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / "peb_phase4_pinn_acid_loss.pt"
    torch.save({
        "state_dict": pinn.state_dict(),
        "config": {
            "DH_nm2_s": DH,
            "kloss_s_inv": KLOSS,
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
            "final_loss_pde": history.loss_pde[-1],
            "max_abs_err_vs_fft": float(err_fft.max().item()),
            "mean_abs_err_vs_fft": float(err_fft.mean().item()),
        },
    }, ckpt_path)
    print(f"  wrote {ckpt_path}")


if __name__ == "__main__":
    main()

"""PEB Phase-5 demo: train PINNDeprotection at kdep=0.5.

Run:
    python reaction_diffusion_peb/experiments/05_deprotection/run_deprotection_pinn.py

Saves the trained model to
``outputs/checkpoints/peb_phase5_pinn_deprotection.pt`` plus a loss
curve and a quick PINN-vs-FD comparison figure.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from reaction_diffusion_peb.src.deprotection import (
    evolve_acid_loss_deprotection_fd,
)
from reaction_diffusion_peb.src.exposure import acid_generation
from reaction_diffusion_peb.src.pinn_diffusion import gaussian_spot_acid_callable
from reaction_diffusion_peb.src.pinn_reaction_diffusion import PINNDeprotection
from reaction_diffusion_peb.src.synthetic_aerial import gaussian_spot
from reaction_diffusion_peb.src.train_pinn_deprotection import (
    pinn_deprotection_to_grid,
    train_pinn_deprotection,
)
from reaction_diffusion_peb.src.visualization import (
    save_figure,
    show_deprotection_fd_vs_pinn,
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
KDEP = 0.5
T_END = 60.0
P_THRESHOLD = 0.5

X_HALF = GRID_SIZE * DX_NM / 2.0
X_RANGE_NM = (-X_HALF, X_HALF)
T_RANGE_S = (0.0, T_END)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    H0_fn = gaussian_spot_acid_callable(
        sigma_nm=SIGMA_NM, Hmax=HMAX, eta=ETA, dose=DOSE,
    )

    pinn = PINNDeprotection(
        DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        hard_ic=True, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        hidden=64, n_hidden_layers=4, n_fourier=16,
        fourier_scale=1.0, activation="tanh", seed=0,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in pinn.parameters() if p.requires_grad)
    print(f"  PINN parameters: {n_params:,}")
    print(f"  training PINNDeprotection (DH={DH}, kloss={KLOSS}, "
          f"kdep={KDEP}, t_end={T_END})...")

    history = train_pinn_deprotection(
        pinn, H0_callable=H0_fn,
        x_range_nm=X_RANGE_NM, t_range_s=T_RANGE_S,
        n_iters=5000, lr=1e-3,
        n_collocation=4096, n_ic=512, n_ic_grid_side=16,
        weight_pde_H=1.0, weight_pde_P=1.0, weight_ic=0.0,
        lr_decay_step=2000, lr_decay_gamma=0.5,
        log_every=50, seed=0,
    )
    print(f"  trained in {history.train_time_sec:.1f} s, "
          f"final loss_pde_H={history.loss_pde_H[-1]:.4e}, "
          f"loss_pde_P={history.loss_pde_P[-1]:.4e}")

    # Build a TrainingHistory-compatible view for the existing plot helper.
    class _HistoryView:
        iters = history.iters
        loss_total = history.loss_total
        loss_pde = history.loss_pde_H  # show H residual on the plot
        loss_ic = history.loss_pde_P   # repurpose IC slot for P residual

    fig = show_pinn_training(
        _HistoryView,
        suptitle=(f"PEB phase 5 PINN training   DH={DH}   kloss={KLOSS}   "
                  f"kdep={KDEP}   "
                  f"(red dashed = P residual, blue = H residual)"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase5_pinn_training.png")
    print(f"  wrote {out}")

    # Compare to FD on the same grid
    H0_grid = acid_generation(
        gaussian_spot(GRID_SIZE, sigma_px=SIGMA_NM / DX_NM),
        dose=DOSE, eta=ETA, Hmax=HMAX,
    ).to(DEVICE)
    H_fd, P_fd = evolve_acid_loss_deprotection_fd(
        H0_grid, DH_nm2_s=DH, kloss_s_inv=KLOSS, kdep_s_inv=KDEP,
        t_end_s=T_END, dx_nm=DX_NM,
    )
    H_pinn, P_pinn = pinn_deprotection_to_grid(
        pinn, GRID_SIZE, DX_NM, T_END, device=DEVICE,
    )
    err_H = (H_pinn - H_fd).abs()
    err_P = (P_pinn - P_fd).abs()
    print(f"  max|H PINN - H FD|={float(err_H.max().item()):.3e}, "
          f"max|P PINN - P FD|={float(err_P.max().item()):.3e}")

    fig = show_deprotection_fd_vs_pinn(
        H0_grid, H_fd, P_fd, H_pinn, P_pinn,
        dx_nm=DX_NM, t_end_s=T_END, P_threshold=P_THRESHOLD,
        suptitle=(f"PEB phase 5 PINN vs FD   DH={DH}   kloss={KLOSS}   "
                  f"kdep={KDEP}   t={T_END} s"),
    )
    out = save_figure(fig, OUT_FIG / "peb_phase5_pinn_vs_fd.png")
    print(f"  wrote {out}")

    OUT_CKPT.mkdir(parents=True, exist_ok=True)
    ckpt_path = OUT_CKPT / "peb_phase5_pinn_deprotection.pt"
    torch.save({
        "state_dict": pinn.state_dict(),
        "config": {
            "DH_nm2_s": DH, "kloss_s_inv": KLOSS, "kdep_s_inv": KDEP,
            "x_range_nm": X_RANGE_NM, "t_range_s": T_RANGE_S,
            "sigma_nm": SIGMA_NM, "Hmax_mol_dm3": HMAX,
            "eta": ETA, "dose": DOSE,
            "hidden": 64, "n_hidden_layers": 4, "n_fourier": 16,
            "fourier_scale": 1.0,
        },
        "training": {
            "train_time_sec": history.train_time_sec,
            "final_loss_pde_H": history.loss_pde_H[-1],
            "final_loss_pde_P": history.loss_pde_P[-1],
            "max_abs_err_H_vs_FD": float(err_H.max().item()),
            "max_abs_err_P_vs_FD": float(err_P.max().item()),
        },
    }, ckpt_path)
    print(f"  wrote {ckpt_path}")


if __name__ == "__main__":
    main()
